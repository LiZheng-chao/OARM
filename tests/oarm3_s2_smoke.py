import argparse
import json
import math
import os
import tempfile

import numpy as np
import torch

from OARM.config import get_oarm_training_preset
from OARM.eval.check_episode_splits import check_episode_splits
from OARM.eval.fit_risk_calibration import fit_calibration_from_jsonl
from OARM.loss import OARMLoss
from OARM.policy.oarm_brake import constrained_brake_command, deterministic_brake_endpoint, evaluate_brake_trajectory
from OARM.policy.oarm_intervention_selector import InterventionSelectorConfig, OARMInterventionSelector
from OARM.policy.oarm_latency_model import OARMLatencyModel
from OARM.policy.oarm_network import OARMNetwork
from OARM.policy.oarm_risk_calibrator import TemperatureCalibration
from OARM.policy.oarm_rm_critic import (
    CandidateRMCritic,
    hazard_cdf_from_logits,
    risk_probability_from_window,
    two_stage_risk_probability,
)
from OARM.policy.oarm_trainer import OARMTrainer
from OARM.train_oarm import parser, resolve_training_options
from OARM.utils.checkpoint import make_oarm_checkpoint, validate_checkpoint_metadata
from OARM.visibility.first_visible_time import first_visible_time
from OARM.visibility.reaction_margin_labeler import ReactionMarginLabeler


def check_preset_route():
    preset = get_oarm_training_preset("oarm3_s2_prob_rm")
    assert preset.candidate_mode == "yopo_preserve"
    assert preset.backbone_mode == "yopo_original"
    assert preset.train_probabilistic_rm_critic is True
    assert preset.rm_critic_hazard_bins == 8
    assert preset.train_reaction_margin is False
    assert preset.train_margin_ranking is False

    args = parser().parse_args(["--stage", "oarm3_s2_prob_rm"])
    options = resolve_training_options(args)
    assert options["train_probabilistic_rm_critic"] is True
    assert options["rm_critic_hazard_bins"] == 8
    assert options["train_reaction_margin"] is False


def check_bad_route_rejected():
    args = parser().parse_args(["--train-probabilistic-rm-critic"])
    try:
        resolve_training_options(args)
    except ValueError as exc:
        assert "OARM3 S2" in str(exc)
        return
    raise AssertionError("misconfigured --train-probabilistic-rm-critic should fail instead of using the old path")


def check_checkpoint_metadata():
    state = {"x": torch.zeros(1)}
    ckpt = make_oarm_checkpoint(
        state,
        "yopo_preserve",
        "yopo_original",
        {
            "stage": "oarm3_s2_prob_rm",
            "train_probabilistic_rm_critic": True,
            "risk_label_source": "gt_pointcloud",
            "yopo_preserve_utility_delta_scale": 0.35,
        },
    )
    assert ckpt["oarm_version"] == "oarm3"
    assert ckpt["training_route"] == "oarm3_s2_prob_rm"
    assert ckpt["checkpoint_format"] == "oarm_policy_v3"
    validate_checkpoint_metadata(
        ckpt,
        "yopo_preserve",
        "yopo_original",
        risk_label_source="gt_pointcloud",
        yopo_preserve_utility_delta_scale=0.35,
        train_probabilistic_rm_critic=True,
    )
    old_metadata = {
        "candidate_mode": "yopo_preserve",
        "backbone_mode": "yopo_original",
        "risk_label_source": "gt_pointcloud",
    }
    try:
        validate_checkpoint_metadata(
            old_metadata,
            "yopo_preserve",
            "yopo_original",
            risk_label_source="gt_pointcloud",
            train_probabilistic_rm_critic=True,
        )
    except ValueError as exc:
        assert "train_probabilistic_rm_critic" in str(exc)
    else:
        raise AssertionError("OARM3 S2 resume should reject checkpoints without S2 metadata")


def check_labeler_window_semantics():
    labeler = ReactionMarginLabeler(reaction_time=0.3, risk_arrival_radius_m=0.05)
    sampled_time = torch.tensor([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]], dtype=torch.float32)
    sampled_pos = torch.zeros((2, 3, 3), dtype=torch.float32)
    sampled_pos[:, :, 0] = sampled_time
    yaw_ref = torch.full((2, 3), torch.pi, dtype=torch.float32)
    risk_points = torch.tensor([[[1.0, 0.0, 0.0]], [[10.0, 0.0, 0.0]]], dtype=torch.float32)
    risk_weight = torch.ones((2, 1), dtype=torch.float32)
    visibility_mask = torch.zeros((2, 3, 1), dtype=torch.bool)
    labels = labeler(sampled_pos, sampled_time, yaw_ref, risk_points, risk_weight, visibility_mask=visibility_mask)
    assert bool(labels["reaction_margin_valid"][0])
    assert not bool(labels["rm_event_valid_gt"][0])
    assert bool(labels["rm_right_censored_gt"][0])
    assert bool(labels["rm_blind_at_entry_gt"][0])
    assert not bool(labels["rm_no_entry_gt"][0])
    assert torch.allclose(labels["reaction_window_gt"][0], torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(
        labels["reaction_margin_gt"][0],
        labels["reaction_window_gt"][0] - torch.tensor(0.3),
        atol=1e-5,
    )
    assert bool(labels["rm_no_entry_gt"][1])
    mask_sum = (
        labels["rm_event_valid_gt"].int()
        + labels["rm_right_censored_gt"].int()
        + labels["rm_no_entry_gt"].int()
    )
    assert bool((mask_sum <= 1).all())

    origin = torch.zeros((1, 1, 3), dtype=torch.float32)
    t0 = torch.zeros((1, 1), dtype=torch.float32)
    yaw0 = torch.zeros((1, 1), dtype=torch.float32)
    far_point = torch.tensor([[[6.0, 0.0, 0.0]]], dtype=torch.float32)
    visible_limited = first_visible_time(origin, t0, yaw0, far_point, math.radians(90.0), math.radians(90.0), max_range_m=5.0)
    visible_unlimited = first_visible_time(origin, t0, yaw0, far_point, math.radians(90.0), math.radians(90.0), max_range_m=None)
    assert torch.isinf(visible_limited).all()
    assert torch.isfinite(visible_unlimited).all()

    interp_pos = torch.zeros((1, 2, 3), dtype=torch.float32)
    interp_time = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    interp_yaw = torch.tensor([[0.0, math.pi / 2.0]], dtype=torch.float32)
    side_point = torch.tensor([[[1.0, 2.0, 0.0]]], dtype=torch.float32)
    visible_interp = first_visible_time(interp_pos, interp_time, interp_yaw, side_point, math.radians(90.0), math.radians(90.0), max_range_m=None)
    assert 0.0 < float(visible_interp[0, 0]) < 1.0


def check_loss_fail_fast():
    loss = OARMLoss(enable_probabilistic_rm_critic=True)
    candidate_flat = {
        "traj_time": torch.ones(2),
        "reaction_window_mean": torch.zeros(2),
        "reaction_window_logvar": torch.zeros(2),
        "validity_logit": torch.zeros(2),
    }
    try:
        loss.probabilistic_rm_critic_loss(candidate_flat, {})
    except RuntimeError as exc:
        assert "reaction_window" in str(exc)
        return
    raise AssertionError("RM critic loss should fail when reaction_window labels are missing")


def check_two_stage_hazard_risk_model():
    critic = CandidateRMCritic(candidate_feature_dim=3, hazard_bins=4)
    critic_out = critic(torch.zeros(2, 5, 3))
    assert critic_out["hazard_logits"].shape == (2, 5, 4)
    assert torch.allclose(critic_out["zero_window_logit"], critic_out["insufficient_margin_logit"])

    geom_critic = CandidateRMCritic(candidate_feature_dim=3, geometry_feature_dim=2, hidden_dim=4)
    with torch.no_grad():
        for param in geom_critic.parameters():
            param.zero_()
        geom_critic.mlp[0].weight[0, 4] = 1.0
        geom_critic.mlp[2].weight[0, 0] = 1.0
        geom_critic.mlp[-1].weight[0, 0] = 1.0
    same_feature = torch.zeros(1, 1, 3)
    same_cost = torch.zeros(1, 1)
    out_a = geom_critic(same_feature, yopo_cost=same_cost, candidate_geometry=torch.zeros(1, 1, 2))["reaction_window_mean"]
    out_b = geom_critic(same_feature, yopo_cost=same_cost, candidate_geometry=torch.ones(1, 1, 2))["reaction_window_mean"]
    assert not torch.allclose(out_a, out_b)

    hazard_logits = torch.tensor([[-4.0, 3.0, -4.0, -4.0]], dtype=torch.float32)
    cdf_early = hazard_cdf_from_logits(hazard_logits, 0.25, max_time_s=2.0)
    cdf_late = hazard_cdf_from_logits(hazard_logits, 1.25, max_time_s=2.0)
    assert bool(torch.all(cdf_late >= cdf_early))

    interaction = torch.tensor([4.0], dtype=torch.float32)
    zero_low = torch.tensor([-4.0], dtype=torch.float32)
    zero_high = torch.tensor([4.0], dtype=torch.float32)
    risk_low_zero = two_stage_risk_probability(interaction, zero_low, hazard_logits, 0.75, hazard_max_time_s=2.0)
    risk_high_zero = two_stage_risk_probability(interaction, zero_high, hazard_logits, 0.75, hazard_max_time_s=2.0)
    assert bool(torch.all(risk_high_zero >= risk_low_zero))

    loss = OARMLoss(
        enable_probabilistic_rm_critic=True,
        rm_critic_hazard_bins=4,
        rm_critic_hazard_max_time_s=2.0,
    )
    candidate_flat = {
        "traj_time": torch.ones(4),
        "reaction_window_mean": torch.tensor([0.0, 0.5, 1.2, 0.0]),
        "reaction_window_logvar": torch.zeros(4),
        "validity_logit": torch.zeros(4),
        "zero_window_logit": torch.zeros(4),
        "hazard_logits": torch.zeros(4, 4),
    }
    labels = {
        "reaction_window": torch.tensor([0.0, 0.5, 1.2, 0.0]),
        "rm_interaction_valid": torch.tensor([True, True, True, False]),
        "rm_timely_visible": torch.tensor([False, True, True, False]),
        "rm_event_valid": torch.tensor([False, True, True, False]),
        "rm_right_censored": torch.tensor([True, False, False, False]),
        "rm_blind_at_entry": torch.tensor([True, False, False, False]),
        "rm_no_entry": torch.tensor([False, False, False, True]),
        "risk_visible_at_t0": torch.tensor([False, False, False, False]),
    }
    loss_dict = loss.probabilistic_rm_critic_loss(candidate_flat, labels)
    assert loss_dict["rm_critic_zero_bce"] > 0.0
    assert loss_dict["rm_critic_hazard_bce"] > 0.0
    assert loss_dict["rm_critic_positive_event_rate"] > 0.0
    assert loss_dict["rm_critic_two_stage_risk_mean"] > 0.0


def check_latency_budget_margin():
    model = OARMLatencyModel(
        brake_accel_mps2=2.0,
        sensor_age_s=0.01,
        queue_latency_s=0.02,
        selector_latency_s=0.03,
        control_latency_s=0.04,
        actuation_latency_s=0.05,
        reaction_margin_s=0.10,
    )
    budget = model.estimate(speed_parallel_mps=2.0, inference_latency_s=0.06)
    assert abs(budget.maneuver_latency_s - 1.0) < 1e-6
    assert abs(budget.tau_fixed_s - 0.21) < 1e-6
    assert abs(budget.tau_total_s - 1.31) < 1e-6
    override = model.estimate(speed_parallel_mps=2.0, inference_latency_s=0.01, maneuver_latency_s=1.7, brake_distance_m=3.4)
    assert abs(override.maneuver_latency_s - 1.7) < 1e-6
    assert abs(override.brake_distance_m - 3.4) < 1e-6
    assert budget.log_fields_ms()["reaction_margin_ms"] == 100.0


def check_intervention_selector_excludes_top1_rerank():
    selector = OARMInterventionSelector(InterventionSelectorConfig(delta_keep=0.10, delta_safe=0.20, risk_improvement_min=0.02))
    decision = selector.select(
        risk_upper_bound=[0.15, 0.35, 0.40],
        yopo_cost=[0.0, 0.1, 0.2],
        geometry_admissible=[True, True, True],
        top1_index=0,
    )
    assert decision.intervention_type == "KEEP"
    assert decision.intervention_reason == "KEEP_GRAY_NO_RISK_IMPROVEMENT"
    assert decision.risk_after <= decision.risk_before

    decision = selector.select(
        risk_upper_bound=[0.15, 0.19, 0.40],
        yopo_cost=[0.0, -10.0, 0.2],
        geometry_admissible=[True, True, True],
        top1_index=0,
    )
    assert decision.intervention_type == "KEEP"
    assert decision.risk_after <= decision.risk_before

    decision = selector.select(
        risk_upper_bound=[0.15, 0.12, 0.40],
        yopo_cost=[0.0, 0.3, 0.2],
        geometry_admissible=[True, True, True],
        top1_index=0,
    )
    assert decision.intervention_type == "RERANK"
    assert decision.selected_index == 1
    assert decision.risk_after <= decision.risk_before - 0.02

    risk_weighted = OARMInterventionSelector(
        InterventionSelectorConfig(delta_keep=0.10, delta_safe=0.20, risk_improvement_min=0.02, lambda_risk=10.0)
    )
    decision = risk_weighted.select(
        risk_upper_bound=[0.18, 0.14, 0.05],
        yopo_cost=[0.0, 0.0, 0.5],
        geometry_admissible=[True, True, True],
        top1_index=0,
    )
    assert decision.intervention_type == "RERANK"
    assert decision.selected_index == 2
    assert decision.risk_after < decision.risk_before

    decision = selector.select(
        risk_upper_bound=[0.70, 0.55, 0.60],
        yopo_cost=[0.0, 0.2, 0.1],
        geometry_admissible=[True, True, True],
        brake_feasible=False,
        brake_risk_upper_bound=1.0,
        top1_index=0,
    )
    assert decision.intervention_type == "DEGRADED"
    assert decision.intervention_reason == "NO_VERIFIED_SAFE_ACTION"
    assert decision.selected_index == 1
    assert decision.metadata["brake_feasible"] is False


def check_deterministic_brake_endpoint():
    end_pos, end_vel, end_acc = deterministic_brake_endpoint(
        start_pos=np.array([1.0, 2.0, 1.5], dtype=np.float32),
        start_vel=np.array([2.0, 0.0, -0.5], dtype=np.float32),
        goal=np.array([5.0, 2.0, 2.0], dtype=np.float32),
        selected_time=0.5,
        distance_scale=0.0,
        retreat_distance=0.25,
        target_z=1.8,
        z_rate=0.4,
        min_command_z=1.0,
        max_command_z=2.0,
    )
    assert np.allclose(end_vel, np.zeros(3), atol=1e-6)
    assert np.allclose(end_acc, np.zeros(3), atol=1e-6)
    assert end_pos[0] < 1.0
    assert 1.0 <= float(end_pos[2]) <= 2.0


def check_constrained_brake_trajectory():
    slow = constrained_brake_command(
        start_pos=np.array([0.0, 0.0, 1.5], dtype=np.float32),
        start_vel=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        start_acc=np.zeros(3, dtype=np.float32),
        goal=np.array([10.0, 0.0, 1.5], dtype=np.float32),
        min_time=0.3,
        brake_accel=2.0,
        max_time=4.0,
        max_accel=10.0,
        max_jerk=80.0,
        max_thrust_accel=25.0,
        max_tilt_deg=80.0,
    )
    fast = constrained_brake_command(
        start_pos=np.array([0.0, 0.0, 1.5], dtype=np.float32),
        start_vel=np.array([3.0, 0.0, 0.0], dtype=np.float32),
        start_acc=np.zeros(3, dtype=np.float32),
        goal=np.array([10.0, 0.0, 1.5], dtype=np.float32),
        min_time=0.3,
        brake_accel=2.0,
        max_time=6.0,
        max_accel=10.0,
        max_jerk=80.0,
        max_thrust_accel=25.0,
        max_tilt_deg=80.0,
    )
    assert fast.diagnostics.stop_distance > slow.diagnostics.stop_distance
    assert fast.duration >= slow.duration
    assert np.allclose(fast.end_vel, np.zeros(3), atol=1e-6)
    assert np.allclose(fast.end_acc, np.zeros(3), atol=1e-6)
    metrics = evaluate_brake_trajectory(
        np.array([0.0, 0.0, 1.5], dtype=np.float32),
        np.array([3.0, 0.0, 0.0], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        fast.end_pos,
        fast.end_vel,
        fast.end_acc,
        fast.duration,
        max_accel=10.0,
        max_jerk=80.0,
        max_thrust_accel=25.0,
        max_tilt_deg=80.0,
    )
    assert metrics["feasible"]
    assert fast.diagnostics.peak_accel <= fast.diagnostics.max_accel + 1e-6
    assert fast.diagnostics.peak_jerk <= fast.diagnostics.max_jerk + 1e-6
    assert fast.diagnostics.max_jerk == 80.0


def check_fit_risk_calibration_cli_core():
    rows = [
        {
            "split": "calibration",
            "episode_id": "ep0",
            "candidates": [
                {"raw_risk_prob": 0.05, "validity_prob": 0.9, "insufficient_reaction_gt": 0, "reaction_window_gt": 1.0, "reaction_budget_s": 0.5},
                {"raw_risk_prob": 0.80, "validity_prob": 0.8, "insufficient_reaction_gt": 1, "reaction_window_gt": 0.2, "reaction_budget_s": 0.5},
            ]
        },
        {
            "split": "calibration",
            "episode_id": "ep1",
            "candidates": [
                {"raw_risk_prob": 0.20, "validity_prob": 0.7, "insufficient_reaction_gt": 0, "reaction_window_gt": 0.8, "reaction_budget_s": 0.5},
                {"raw_risk_prob": 0.65, "validity_prob": 0.9, "insufficient_reaction_gt": 1, "reaction_window_gt": 0.1, "reaction_budget_s": 0.5},
            ]
        },
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "calib.jsonl")
        out_path = os.path.join(tmpdir, "calibration_fit.json")
        with open(in_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        payload = fit_calibration_from_jsonl([in_path], out_path, label_key="insufficient_reaction_gt", empirical_upper_alpha=0.25, max_iter=10)
        assert payload["sample_count"] == 4
        assert payload["validity_fusion"] is True
        assert payload["label_key"] == "insufficient_reaction_gt"
        assert payload["input_stats"]["episode_count"] == 2
        assert payload["conformal_slack"] >= 0.0
        loaded = TemperatureCalibration.from_file(out_path)
        assert loaded.temperature == payload["temperature"]
        assert loaded.conformal_slack == payload["conformal_slack"]
        assert os.path.exists(out_path)

        derived_path = os.path.join(tmpdir, "derived_calibration.json")
        derived = fit_calibration_from_jsonl([in_path], derived_path, label_key="reaction_window_lt_budget", empirical_upper_alpha=0.25, max_iter=10)
        assert derived["sample_count"] == 4
        assert derived["label_key"] == "reaction_window_lt_budget"
        assert derived["input_stats"]["missing_reaction_window"] == 0
        assert derived["input_stats"]["missing_reaction_budget"] == 0

        hazard_path = os.path.join(tmpdir, "hazard_calib.jsonl")
        with open(hazard_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"split": "calibration", "episode_id": "ep_h0", "candidates": [{"hazard_risk_prob": 0.05, "validity_prob": 0.1, "insufficient_reaction_gt": 0}]}) + "\n")
            f.write(json.dumps({"split": "calibration", "episode_id": "ep_h1", "candidates": [{"hazard_risk_prob": 0.90, "validity_prob": 0.1, "insufficient_reaction_gt": 1}]}) + "\n")
        hazard_payload = fit_calibration_from_jsonl([hazard_path], os.path.join(tmpdir, "hazard.json"), label_key="insufficient_reaction_gt", empirical_upper_alpha=0.25, max_iter=5)
        assert hazard_payload["input_stats"]["validity_fusion_skipped_two_stage"] == 2
        assert hazard_payload["input_stats"]["missing_validity"] == 0

        bad_path = os.path.join(tmpdir, "bad_collision.jsonl")
        with open(bad_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"split": "calibration", "episode_id": "ep_bad", "candidates": [{"raw_risk_prob": 0.4, "validity_prob": 1.0, "collision": True}]}) + "\n")
        try:
            fit_calibration_from_jsonl([bad_path], os.path.join(tmpdir, "bad.json"), label_key="collision")
        except ValueError as exc:
            assert "episode-level" in str(exc) or "candidate-level" in str(exc)
        else:
            raise AssertionError("candidate-level calibration must reject episode collision labels")

        generic_path = os.path.join(tmpdir, "bad_generic_label.jsonl")
        with open(generic_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"split": "calibration", "episode_id": "ep_generic", "candidates": [{"raw_risk_prob": 0.4, "validity_prob": 1.0, "risk_label": 1}]}) + "\n")
        try:
            fit_calibration_from_jsonl([generic_path], os.path.join(tmpdir, "bad_generic.json"), label_key="risk_label")
        except ValueError as exc:
            assert "too generic" in str(exc)
        else:
            raise AssertionError("candidate-level calibration must reject generic risk_label labels")

        missing_split = os.path.join(tmpdir, "missing_split.jsonl")
        with open(missing_split, "w", encoding="utf-8") as f:
            f.write(json.dumps({"episode_id": "ep2", "candidates": [{"raw_risk_prob": 0.4, "validity_prob": 1.0, "insufficient_reaction_gt": 0}]}) + "\n")
        try:
            fit_calibration_from_jsonl([missing_split], os.path.join(tmpdir, "missing.json"), label_key="insufficient_reaction_gt")
        except ValueError as exc:
            assert "split" in str(exc)
        else:
            raise AssertionError("formal calibration should require split metadata")


def check_episode_split_manifest_guard():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifests = {}
        split_rows = {
            "train": [{"episode_id": "train_ep", "map_id": "map_train"}],
            "val": [{"episode_id": "val_ep", "map_id": "map_val"}],
            "calibration": [{"episode_id": "cal_ep", "map_id": "map_cal"}],
            "test": [{"episode_id": "test_ep", "map_id": "map_test"}],
        }
        for split, rows in split_rows.items():
            path = os.path.join(tmpdir, f"{split}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"episodes": rows}, f)
            manifests[split] = path
        result = check_episode_splits(manifests)
        assert result["ok"] is True
        assert result["splits"]["calibration"]["episode_count"] == 1

        data_path = os.path.join(tmpdir, "calibration_data.jsonl")
        out_path = os.path.join(tmpdir, "calibration_fit_manifest.json")
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"split": "calibration", "episode_id": "cal_ep", "map_id": "map_cal", "raw_risk_prob": 0.2, "validity_prob": 1.0, "reaction_window_gt": 1.0, "reaction_budget_s": 0.5}) + "\n")
            f.write(json.dumps({"split": "calibration", "episode_id": "cal_ep", "map_id": "map_cal", "raw_risk_prob": 0.8, "validity_prob": 1.0, "reaction_window_gt": 0.1, "reaction_budget_s": 0.5}) + "\n")
        payload = fit_calibration_from_jsonl(
            [data_path],
            out_path,
            label_key="reaction_window_lt_budget",
            train_manifest=manifests["train"],
            val_manifest=manifests["val"],
            calibration_manifest=manifests["calibration"],
            test_manifest=manifests["test"],
            max_iter=5,
        )
        assert payload["split_manifest_check"]["ok"] is True

        with open(manifests["test"], "w", encoding="utf-8") as f:
            json.dump({"episodes": [{"episode_id": "test_ep", "map_id": "map_cal"}]}, f)
        try:
            check_episode_splits(manifests)
        except ValueError as exc:
            assert "leakage" in str(exc)
        else:
            raise AssertionError("split checker should reject map overlap between calibration and test")


def check_trainable_contract():
    trainer = OARMTrainer.__new__(OARMTrainer)
    trainer.candidate_mode = "yopo_preserve"
    trainer.backbone_mode = "yopo_original"
    trainer.train_yield_head_only = False
    trainer.train_probabilistic_rm_critic = True
    trainer.yopo_preserve_freeze_margin_risk_head = True
    trainer.policy = OARMNetwork(
        candidate_mode="yopo_preserve",
        backbone_mode="yopo_original",
        enable_rm_critic=True,
    )
    trainer.configure_trainable_parameters()
    trainer.assert_trainable_parameter_contract()
    trainable = trainer.trainable_parameter_names()
    assert trainable
    assert all(name.startswith("preserve_network.rm_critic.") for name in trainable)


def check_end_to_end_mini_batch():
    if not torch.cuda.is_available():
        print("oarm3_s2 full mini-batch skipped: CUDA is unavailable")
        return
    torch.manual_seed(7)
    device = torch.device("cuda")
    trainer = OARMTrainer.__new__(OARMTrainer)
    trainer.candidate_mode = "yopo_preserve"
    trainer.backbone_mode = "yopo_original"
    trainer.train_yield_head_only = False
    trainer.train_probabilistic_rm_critic = True
    trainer.yopo_preserve_freeze_margin_risk_head = True
    trainer.policy = OARMNetwork(
        candidate_mode="yopo_preserve",
        backbone_mode="yopo_original",
        enable_rm_critic=True,
    ).to(device)
    trainer.configure_trainable_parameters()
    trainer.assert_trainable_parameter_contract()

    trainable_names = trainer.trainable_parameter_names()
    optimizer = torch.optim.AdamW(trainer.trainable_parameters(), lr=1e-3)
    loss_fn = OARMLoss(enable_probabilistic_rm_critic=True)

    depth = torch.zeros((1, 1, 96, 160), dtype=torch.float32, device=device)
    obs = torch.zeros((1, 9), dtype=torch.float32, device=device)
    obs[:, 6] = 10.0

    trainer.policy.train()
    candidate = trainer.policy.inference(depth, obs)
    flat = candidate.flatten()
    window_shape = flat["reaction_window_mean"].shape
    labels = {
        "reaction_window": torch.linspace(0.0, 1.0, flat["reaction_window_mean"].numel(), device=device).reshape(window_shape),
        "rm_interaction_valid": torch.ones(window_shape, dtype=torch.bool, device=device),
        "rm_event_valid": torch.ones(window_shape, dtype=torch.bool, device=device),
        "rm_timely_visible": torch.ones(window_shape, dtype=torch.bool, device=device),
        "rm_right_censored": torch.zeros(window_shape, dtype=torch.bool, device=device),
        "rm_no_entry": torch.zeros(window_shape, dtype=torch.bool, device=device),
    }
    labels["rm_event_valid"].reshape(-1)[0] = False
    labels["rm_timely_visible"].reshape(-1)[0] = False
    labels["rm_right_censored"].reshape(-1)[0] = True
    labels["reaction_window"].reshape(-1)[0] = 0.0

    noncritic_before = {
        name: param.detach().clone()
        for name, param in trainer.policy.named_parameters()
        if not name.startswith("preserve_network.rm_critic.")
    }
    critic_before = {
        name: param.detach().clone()
        for name, param in trainer.policy.named_parameters()
        if name.startswith("preserve_network.rm_critic.")
    }

    loss_dict = loss_fn.probabilistic_rm_critic_loss(flat, labels)
    assert torch.isfinite(loss_dict["rm_critic_loss"])
    assert loss_dict["rm_interaction_valid_rate"].item() > 0.99
    assert loss_dict["rm_critic_positive_event_rate"].item() > 0.0
    assert loss_dict["rm_zero_window_rate"].item() > 0.0
    optimizer.zero_grad(set_to_none=True)
    loss_dict["rm_critic_loss"].backward()

    grad_names = [name for name, param in trainer.policy.named_parameters() if param.grad is not None]
    assert grad_names
    assert all(name.startswith("preserve_network.rm_critic.") for name in grad_names)
    optimizer.step()

    changed = False
    for name, param in trainer.policy.named_parameters():
        if name.startswith("preserve_network.rm_critic."):
            changed = changed or not torch.allclose(param.detach(), critic_before[name])
        else:
            assert torch.allclose(param.detach(), noncritic_before[name]), name
    assert changed

    trainer.policy.eval()
    with torch.inference_mode():
        out_before = trainer.policy.inference(depth, obs).flatten()["reaction_window_mean"].detach().cpu()
    ckpt = make_oarm_checkpoint(
        trainer.policy.state_dict(),
        "yopo_preserve",
        "yopo_original",
        {"stage": "oarm3_s2_prob_rm", "train_probabilistic_rm_critic": True},
    )
    with tempfile.NamedTemporaryFile(suffix=".pth") as f:
        torch.save(ckpt, f.name)
        payload = torch.load(f.name, map_location=device, weights_only=True)
    reloaded = OARMNetwork(
        candidate_mode="yopo_preserve",
        backbone_mode="yopo_original",
        enable_rm_critic=True,
    ).to(device)
    reloaded.load_state_dict(payload["state_dict"])
    reloaded.eval()
    with torch.inference_mode():
        out_after = reloaded.inference(depth, obs).flatten()["reaction_window_mean"].detach().cpu()
    assert torch.allclose(out_before, out_after, atol=1e-6)

    mu = flat["reaction_window_mean"].detach()
    logvar = flat["reaction_window_logvar"].detach()
    r1 = risk_probability_from_window(mu, logvar, 0.1)
    r2 = risk_probability_from_window(mu, logvar, 0.5)
    assert bool(torch.all(r2 >= r1))


def main():
    check_preset_route()
    check_bad_route_rejected()
    check_checkpoint_metadata()
    check_labeler_window_semantics()
    check_loss_fail_fast()
    check_two_stage_hazard_risk_model()
    check_latency_budget_margin()
    check_intervention_selector_excludes_top1_rerank()
    check_deterministic_brake_endpoint()
    check_constrained_brake_trajectory()
    check_fit_risk_calibration_cli_core()
    check_episode_split_manifest_guard()
    check_trainable_contract()
    check_end_to_end_mini_batch()
    print("oarm3_s2_smoke ok")


if __name__ == "__main__":
    main()

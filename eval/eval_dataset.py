import argparse
import json
import math
import os
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from OARM.dataset import OARMDataset
from OARM.config import get_oarm_training_preset, oarm_cfg
from OARM.eval.metrics_backup_feasibility import backup_feasibility_metrics
from OARM.eval.metrics_reaction_margin import (
    margin_disentanglement_metrics,
    margin_prediction_metrics,
    matched_pairwise_ranking_accuracy,
    pairwise_ranking_accuracy,
    reaction_margin_metrics,
    risk_calibration_metrics,
)
from OARM.loss import OARMLoss
from OARM.loss.reaction_margin_loss import weak_margin_label_from_risk
from OARM.policy.oarm_candidate_generator import OARMCandidateGenerator
from OARM.policy.oarm_poly_solver import quintic_coefficients, sample_polynomial
from OARM.policy.oarm_network import OARMNetwork
from OARM.policy.oarm_state_transform import rotate_body2world, state_body2world
from OARM.utils.checkpoint import load_oarm_checkpoint, validate_checkpoint_metadata
from OARM.utils.gt_clearance import candidate_min_clearance_gt
from OARM.utils.visible_free_distance import visible_free_distance_from_depth
from OARM.utils.yopo_compat import ensure_yopo_path
from OARM.utils.yopo_dataset_context import yopo_dataset_cfg
from OARM.visibility.reaction_margin_labeler import ReactionMarginLabeler
from OARM.visibility.reaction_margin_targets import generate_reaction_margin_labels

ensure_yopo_path()
from config.config import cfg


TYPE_NAMES = {
    OARMCandidateGenerator.PROGRESS: "progress",
    OARMCandidateGenerator.PROBE: "probe",
    OARMCandidateGenerator.BRAKE: "brake",
    OARMCandidateGenerator.YIELD: "yield",
}


EVAL_STAGE_MAP = {
    "eval_occlusion_risk": "train_occlusion_risk",
    "eval_risk_point_guidance": "train_risk_point_guidance",
    "eval_reaction_margin": "train_reaction_margin",
    "eval_margin_ranking": "train_margin_ranking",
    "eval_backup_feasibility": "train_yield_feasibility",
    "eval_yield_feasibility": "train_yield_feasibility",
    "use_weak_margin_label": "use_weak_margin_label",
    "use_esdf_collision": "use_esdf_collision",
    "use_occlusion_aware_visibility": "use_occlusion_aware_visibility",
    "use_privileged_risk_filter": "use_privileged_risk_filter",
}


GT_SAMPLER_ARG_KEYS = (
    "gt_risk_point_count",
    "gt_hidden_depth_margin_m",
    "gt_min_forward_m",
    "gt_max_forward_m",
    "gt_horizon_fov_expand_deg",
    "gt_vertical_fov_expand_deg",
    "gt_depth_metric",
    "gt_reachable_forward_center_m",
    "gt_reachable_forward_sigma_m",
    "gt_reachable_lateral_sigma_m",
    "gt_reachable_vertical_sigma_m",
    "gt_reachable_score_weight",
    "gt_side_score_weight",
    "gt_risk_nms_radius_m",
    "gt_risk_voxel_size_m",
    "risk_assoc_distance_m",
    "risk_assoc_sigma_m",
    "risk_arrival_radius_m",
)


def gt_sampler_options_from_args(args):
    return {
        "point_count": args.gt_risk_point_count,
        "hidden_depth_margin_m": args.gt_hidden_depth_margin_m,
        "min_forward_m": args.gt_min_forward_m,
        "max_forward_m": args.gt_max_forward_m,
        "horizon_fov_expand_deg": args.gt_horizon_fov_expand_deg,
        "vertical_fov_expand_deg": args.gt_vertical_fov_expand_deg,
        "depth_metric": args.gt_depth_metric,
        "reachable_forward_center_m": args.gt_reachable_forward_center_m,
        "reachable_forward_sigma_m": args.gt_reachable_forward_sigma_m,
        "reachable_lateral_sigma_m": args.gt_reachable_lateral_sigma_m,
        "reachable_vertical_sigma_m": args.gt_reachable_vertical_sigma_m,
        "reachable_score_weight": args.gt_reachable_score_weight,
        "side_score_weight": args.gt_side_score_weight,
        "nms_radius_m": args.gt_risk_nms_radius_m,
        "voxel_size_m": args.gt_risk_voxel_size_m,
    }


def tensor_scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def add_metric(accumulator, key, value, weight=1):
    value = tensor_scalar(value)
    if weight <= 0 or not math.isfinite(value):
        return
    accumulator[key].append((value, weight))


def add_quantile_metrics(accumulator, prefix, values, weight=1):
    flat = values.reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return
    probs = torch.tensor([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99], device=flat.device, dtype=flat.dtype)
    quantiles = torch.quantile(flat, probs)
    names = ("p01", "p05", "p25", "p50", "p75", "p95", "p99")
    metric_weight = int(flat.numel()) if weight is None else weight
    for name, value in zip(names, quantiles):
        add_metric(accumulator, f"{prefix}_{name}", value, metric_weight)


def finalize_metrics(accumulator):
    metrics = {}
    for key, values in accumulator.items():
        finite_values = [(v, w) for v, w in values if math.isfinite(v) and w > 0]
        if not finite_values:
            continue
        total_weight = sum(w for _, w in finite_values)
        metrics[key] = sum(v * w for v, w in finite_values) / max(total_weight, 1)
    return metrics


def checkpoint_training_option(metadata, key, default=None):
    if not metadata:
        return default
    if key in metadata and metadata[key] is not None:
        return metadata[key]
    training_options = metadata.get("training_options") or {}
    return training_options.get(key, default)


def resolve_utility_delta_scale(args, checkpoint_metadata):
    if args.yopo_preserve_utility_delta_scale is not None:
        return float(args.yopo_preserve_utility_delta_scale)
    stored = checkpoint_training_option(checkpoint_metadata, "yopo_preserve_utility_delta_scale")
    if stored is not None:
        return float(stored)
    preset = get_oarm_training_preset(args.stage)
    return float(getattr(preset, "yopo_preserve_utility_delta_scale", oarm_cfg.yopo_preserve_utility_delta_scale))



EVAL_PROTOCOL_ARG_KEYS = (
    "yopo_preserve_safety_cost_threshold",
    "yopo_preserve_safe_cost_threshold",
    "yopo_preserve_geometry_oracle_source",
    "yopo_preserve_unsafe_clearance_m",
    "yopo_preserve_safe_clearance_m",
    "yopo_preserve_safe_margin_m",
    "yopo_preserve_oracle_min_progress",
)


def apply_checkpoint_eval_protocol(args, checkpoint_metadata):
    training_options = (checkpoint_metadata or {}).get("training_options") or {}
    if checkpoint_metadata:
        for key in ("candidate_mode", "backbone_mode", "deployed_yaw_mode", "risk_label_source"):
            stored = checkpoint_metadata.get(key)
            if stored is None:
                stored = training_options.get(key)
            if stored not in {None, ""}:
                setattr(args, key, stored)
    for eval_key, preset_key in EVAL_STAGE_MAP.items():
        if training_options.get(preset_key) and not getattr(args, eval_key):
            setattr(args, eval_key, True)
    for bool_key in (
        "use_esdf_collision",
        "use_occlusion_aware_visibility",
        "use_privileged_risk_filter",
        "enable_yield_candidates",
    ):
        if training_options.get(bool_key) and not getattr(args, bool_key):
            setattr(args, bool_key, True)
    for key in GT_SAMPLER_ARG_KEYS:
        stored = training_options.get(key)
        if stored is not None:
            setattr(args, key, stored)
    preset = get_oarm_training_preset(args.stage)
    for key in EVAL_PROTOCOL_ARG_KEYS:
        stored = training_options.get(key)
        if stored is not None:
            setattr(args, key, stored)
            continue
        current = getattr(args, key)
        if current is not None and current != "":
            continue
        setattr(args, key, getattr(preset, key, getattr(oarm_cfg, key)))

def flatten_labels(labels, flat, device, args):
    flat_labels = {}
    if args.eval_occlusion_risk and "occlusion_risk" in labels:
        flat_labels["occlusion_risk"] = labels["occlusion_risk"].to(device).reshape(-1)
    if args.eval_reaction_margin and "reaction_margin" in labels:
        flat_labels["reaction_margin"] = labels["reaction_margin"].to(device).reshape(-1)
        if "reaction_margin_valid" in labels:
            flat_labels["reaction_margin_valid"] = labels["reaction_margin_valid"].to(device).reshape(-1)
        if "reaction_margin_censored" in labels:
            flat_labels["reaction_margin_censored"] = labels["reaction_margin_censored"].to(device).reshape(-1)
    elif args.eval_reaction_margin and args.use_weak_margin_label and "occlusion_risk" in flat_labels:
        flat_labels["reaction_margin"] = weak_margin_label_from_risk(flat["traj_time"], flat_labels["occlusion_risk"])
    if args.eval_yield_feasibility and "yield_feasible" in labels:
        flat_labels["backup_feasible"] = labels["yield_feasible"].to(device).reshape(-1)
    elif args.eval_yield_feasibility and "backup_feasible" in labels:
        flat_labels["backup_feasible"] = labels["backup_feasible"].to(device).reshape(-1)
    for source_key in (
        "uses_gt_reaction_margin",
        "uses_proxy_reaction_margin",
        "reaction_margin_label_source_id",
        "hidden_risk_gt",
        "raw_gt_risk_point_valid_rate",
        "raw_gt_risk_point_weight_sum",
        "raw_gt_risk_point_weight_mean",
    ):
        if source_key in labels:
            flat_labels[source_key] = labels[source_key].to(device)
    if args.eval_risk_point_guidance and "risk_points_w" in labels:
        flat_labels["risk_points_w"] = labels["risk_points_w"].to(device)
        if "risk_weight" in labels:
            flat_labels["risk_weight"] = labels["risk_weight"].to(device)
        if "yaw0" in labels:
            flat_labels["yaw0"] = labels["yaw0"].to(device)
        if "yaw_rate0" in labels:
            flat_labels["yaw_rate0"] = labels["yaw_rate0"].to(device)
    return flat_labels


def build_world_states(pos, rot, obs_b, flat):
    goal_w = rotate_body2world(rot, obs_b[:, 6:9])
    start_vel_w = rotate_body2world(rot, obs_b[:, 0:3])
    start_acc_w = rotate_body2world(rot, obs_b[:, 3:6])
    start_state_w = torch.stack([pos, start_vel_w, start_acc_w], dim=1)

    traj_num = cfg["traj_num"]
    endstate_flat = flat["end_state_b"]
    pos_expanded = pos.repeat_interleave(traj_num, dim=0)
    rot_expanded = rot.repeat_interleave(traj_num, dim=0)
    start_state_w = start_state_w.repeat_interleave(traj_num, dim=0)
    goal_w = goal_w.repeat_interleave(traj_num, dim=0)

    end_pos_w, end_vel_w, end_acc_w = state_body2world(
        pos_expanded,
        rot_expanded,
        endstate_flat[:, 0:3],
        endstate_flat[:, 3:6],
        endstate_flat[:, 6:9],
    )
    end_state_w = torch.stack([end_pos_w, end_vel_w, end_acc_w], dim=1)
    return start_state_w, end_state_w, goal_w


def expand_candidate_label(label, candidate_count, like):
    label = label.to(device=like.device, dtype=like.dtype)
    if label.shape[0] == candidate_count:
        return label
    if candidate_count % label.shape[0] != 0:
        raise ValueError(f"Cannot expand label with first dim {label.shape[0]} to {candidate_count} candidates")
    return label.repeat_interleave(candidate_count // label.shape[0], dim=0)


def sampled_time_grid(traj_time, eval_points, include_zero=True):
    start = 0.0 if include_zero else 1.0 / eval_points
    tau = torch.linspace(start, 1.0, eval_points, device=traj_time.device, dtype=traj_time.dtype)
    return traj_time[:, None] * tau[None, :]


def maybe_generate_reaction_margin_labels(
    flat_labels,
    flat,
    start_state_w,
    end_state_w,
    map_id_expanded,
    goal_w,
    args,
    labeler,
    line_of_sight,
    yaw_helper,
):
    return generate_reaction_margin_labels(
        flat_labels,
        flat,
        start_state_w,
        end_state_w,
        map_id_expanded,
        goal_w,
        enabled=args.eval_reaction_margin,
        labeler=labeler,
        line_of_sight=line_of_sight,
        yaw_helper=yaw_helper,
    )


def selected_candidate_stats(candidate, accumulator, args, flat_labels=None, safety_cost=None, progress=None, min_clearance_gt=None):
    utility = candidate.utility_score.reshape(candidate.utility_score.shape[0], -1)
    best_id = utility.argmax(dim=1)
    batch_size = utility.shape[0]
    add_metric(accumulator, "selected_utility", utility.gather(1, best_id[:, None]).mean(), batch_size)

    flat_time = candidate.traj_time.reshape(batch_size, -1)
    add_metric(accumulator, "selected_time", flat_time.gather(1, best_id[:, None]).mean(), batch_size)

    flat_safety = None
    finite_safety = None
    geom_unsafe = None
    geom_safe = None
    if safety_cost is not None:
        flat_safety = safety_cost.reshape(batch_size, -1).to(device=utility.device, dtype=utility.dtype)
        finite_safety = torch.isfinite(flat_safety)
        esdf_unsafe = finite_safety & (flat_safety > args.yopo_preserve_safety_cost_threshold)
        esdf_safe = finite_safety & (flat_safety <= args.yopo_preserve_safe_cost_threshold)
        selected_safety = flat_safety.gather(1, best_id[:, None]).squeeze(1)
        selected_safety_valid = torch.isfinite(selected_safety)
        if bool(finite_safety.any()):
            add_quantile_metrics(accumulator, "esdf_safety_cost", flat_safety[finite_safety], int(finite_safety.sum().item()))
        if bool(selected_safety_valid.any()):
            add_metric(accumulator, "selected_safety_cost", selected_safety[selected_safety_valid].mean(), int(selected_safety_valid.sum().item()))
            add_metric(
                accumulator,
                "selected_esdf_unsafe_candidate_rate",
                (selected_safety[selected_safety_valid] > args.yopo_preserve_safety_cost_threshold).float().mean(),
                int(selected_safety_valid.sum().item()),
            )
        add_metric(accumulator, "esdf_unsafe_candidate_rate", esdf_unsafe.float().mean(), batch_size)
        add_metric(accumulator, "esdf_safe_candidate_rate", esdf_safe.float().mean(), batch_size)
        add_metric(accumulator, "esdf_safe_candidate_available_rate", esdf_safe.any(dim=1).float().mean(), batch_size)
        if args.yopo_preserve_geometry_oracle_source == "esdf_cost":
            geom_unsafe = esdf_unsafe
            geom_safe = esdf_safe
            if bool(selected_safety_valid.any()):
                add_metric(
                    accumulator,
                    "selected_geom_unsafe_candidate_rate",
                    (selected_safety[selected_safety_valid] > args.yopo_preserve_safety_cost_threshold).float().mean(),
                    int(selected_safety_valid.sum().item()),
                )
            add_metric(accumulator, "geom_unsafe_candidate_rate", geom_unsafe.float().mean(), batch_size)
            add_metric(accumulator, "geom_safe_candidate_rate", geom_safe.float().mean(), batch_size)
            add_metric(accumulator, "geom_safe_candidate_available_rate", geom_safe.any(dim=1).float().mean(), batch_size)

    if min_clearance_gt is not None:
        flat_clearance = min_clearance_gt.reshape(batch_size, -1).to(device=utility.device, dtype=utility.dtype)
        finite_clearance = torch.isfinite(flat_clearance)
        gt_unsafe = finite_clearance & (flat_clearance < args.yopo_preserve_unsafe_clearance_m)
        gt_safe = finite_clearance & (flat_clearance > args.yopo_preserve_safe_clearance_m)
        if bool(finite_clearance.any()):
            add_metric(accumulator, "gt_min_clearance_mean", flat_clearance[finite_clearance].mean(), int(finite_clearance.sum().item()))
            add_quantile_metrics(accumulator, "gt_min_clearance", flat_clearance[finite_clearance], int(finite_clearance.sum().item()))
        selected_clearance = flat_clearance.gather(1, best_id[:, None]).squeeze(1)
        selected_clearance_valid = torch.isfinite(selected_clearance)
        if bool(selected_clearance_valid.any()):
            weight = int(selected_clearance_valid.sum().item())
            add_metric(accumulator, "selected_gt_min_clearance", selected_clearance[selected_clearance_valid].mean(), weight)
            add_metric(accumulator, "selected_gt_clearance_collision_rate", (selected_clearance[selected_clearance_valid] < args.yopo_preserve_unsafe_clearance_m).float().mean(), weight)
        add_metric(accumulator, "gt_clearance_valid_rate", finite_clearance.float().mean(), batch_size)
        add_metric(accumulator, "gt_clearance_unsafe_candidate_rate", gt_unsafe.float().mean(), batch_size)
        add_metric(accumulator, "gt_clearance_safe_candidate_rate", gt_safe.float().mean(), batch_size)
        add_metric(accumulator, "gt_clearance_safe_candidate_available_rate", gt_safe.any(dim=1).float().mean(), batch_size)
        if args.yopo_preserve_geometry_oracle_source == "gt_clearance":
            geom_unsafe = gt_unsafe
            geom_safe = gt_safe
            if bool(selected_clearance_valid.any()):
                add_metric(accumulator, "selected_geom_unsafe_candidate_rate", (selected_clearance[selected_clearance_valid] < args.yopo_preserve_unsafe_clearance_m).float().mean(), int(selected_clearance_valid.sum().item()))
            add_metric(accumulator, "geom_unsafe_candidate_rate", geom_unsafe.float().mean(), batch_size)
            add_metric(accumulator, "geom_safe_candidate_rate", geom_safe.float().mean(), batch_size)
            add_metric(accumulator, "geom_safe_candidate_available_rate", geom_safe.any(dim=1).float().mean(), batch_size)

    if candidate.risk_logit is not None:
        flat_risk = torch.sigmoid(candidate.risk_logit.reshape(batch_size, -1))
        add_metric(accumulator, "selected_pred_risk", flat_risk.gather(1, best_id[:, None]).mean(), batch_size)

    if candidate.backup_logit is not None:
        flat_backup = torch.sigmoid(candidate.backup_logit.reshape(batch_size, -1))
        selected_backup = flat_backup.gather(1, best_id[:, None]).mean()
        add_metric(accumulator, "selected_pred_backup", selected_backup, batch_size)
        add_metric(accumulator, "selected_pred_yield", selected_backup, batch_size)

    if candidate.candidate_type is not None:
        flat_type = candidate.candidate_type.reshape(batch_size, -1)
        selected_type = flat_type.gather(1, best_id[:, None]).squeeze(1)
        for type_id, name in TYPE_NAMES.items():
            add_metric(accumulator, f"selected_{name}_rate", (selected_type == type_id).float().mean(), batch_size)
            add_metric(accumulator, f"all_{name}_rate", (flat_type == type_id).float().mean(), batch_size)

    if flat_labels is not None and "reaction_margin" in flat_labels:
        flat_margin = flat_labels["reaction_margin"].reshape(batch_size, -1)
        finite_margin = torch.isfinite(flat_margin)
        valid_margin = flat_labels.get("reaction_margin_valid")
        if valid_margin is not None:
            finite_margin = finite_margin & valid_margin.reshape_as(flat_margin).bool()
        add_metric(accumulator, "selected_margin_finite_rate", finite_margin.float().mean(), batch_size)
        add_metric(accumulator, "selected_margin_valid_rate", finite_margin.float().mean(), batch_size)
        selected_margin = flat_margin.gather(1, best_id[:, None]).squeeze(1)
        selected_finite = torch.isfinite(selected_margin)
        if valid_margin is not None:
            selected_valid_mask = valid_margin.reshape_as(flat_margin).bool().gather(1, best_id[:, None]).squeeze(1)
            selected_finite = selected_finite & selected_valid_mask
        if bool(selected_finite.any()):
            selected_valid = selected_margin[selected_finite]
            selected_weight = int(selected_valid.numel())
            add_metric(accumulator, "selected_reaction_margin", selected_valid.mean(), selected_weight)
            add_metric(accumulator, "selected_reaction_margin_violation_rate", (selected_valid < 0.0).float().mean(), selected_weight)
            add_metric(accumulator, "selected_rmvr", (selected_valid < 0.0).float().mean(), selected_weight)

        oracle_valid = finite_margin.any(dim=1)
        oracle_source = flat_margin.masked_fill(~finite_margin, -torch.inf)
        oracle_margin, oracle_id = oracle_source.max(dim=1)
        positive_margin_available = (oracle_margin > 0.0) & oracle_valid
        add_metric(accumulator, "safe_candidate_available_rate", positive_margin_available.float().mean(), batch_size)
        add_metric(accumulator, "positive_margin_candidate_available_rate", positive_margin_available.float().mean(), batch_size)
        if bool(oracle_valid.any()):
            oracle_valid_margin = oracle_margin[oracle_valid]
            oracle_weight = int(oracle_valid_margin.numel())
            add_metric(accumulator, "oracle_best_reaction_margin", oracle_valid_margin.mean(), oracle_weight)
            add_metric(accumulator, "oracle_margin_selected_rate", (best_id[oracle_valid] == oracle_id[oracle_valid]).float().mean(), oracle_weight)
        gap_mask = oracle_valid & selected_finite
        if bool(gap_mask.any()):
            selected_positive = selected_margin[gap_mask] > 0.0
            gap_weight = int(gap_mask.float().sum().item())
            oracle_gap = oracle_margin[gap_mask] - selected_margin[gap_mask]
            add_metric(accumulator, "margin_oracle_gap", oracle_gap.mean(), gap_weight)
            missed = positive_margin_available[gap_mask] & ~selected_positive
            add_metric(accumulator, "safe_candidate_missed_rate", missed.float().mean(), gap_weight)
            add_metric(accumulator, "positive_margin_candidate_missed_rate", missed.float().mean(), gap_weight)

        if geom_safe is None:
            geom_safe_for_oracle = torch.ones_like(finite_margin, dtype=torch.bool)
            geom_unsafe_for_oracle = torch.zeros_like(finite_margin, dtype=torch.bool)
        else:
            geom_safe_for_oracle = geom_safe
            geom_unsafe_for_oracle = geom_unsafe
        margin_unsafe = finite_margin & (flat_margin < 0.0)
        margin_safe = finite_margin & (flat_margin > args.yopo_preserve_safe_margin_m)
        final_unsafe = geom_unsafe_for_oracle | margin_unsafe
        safety_primary_mask = geom_safe_for_oracle & margin_safe & ~final_unsafe
        fallback_mask = geom_safe_for_oracle & finite_margin & ~final_unsafe
        if progress is not None and args.yopo_preserve_oracle_min_progress > 0.0:
            flat_progress = progress.reshape(batch_size, -1).to(device=utility.device, dtype=utility.dtype)
            progress_ok = torch.isfinite(flat_progress) & (flat_progress > args.yopo_preserve_oracle_min_progress)
            safety_primary_mask = safety_primary_mask & progress_ok
            fallback_mask = fallback_mask & progress_ok
            add_metric(accumulator, "safety_oracle_progress_ok_rate", progress_ok.float().mean(), batch_size)
        primary_available = safety_primary_mask.any(dim=1)
        fallback_available = fallback_mask.any(dim=1) & ~primary_available
        safety_oracle_mask = torch.where(primary_available[:, None], safety_primary_mask, fallback_mask)
        if geom_unsafe is not None:
            add_metric(accumulator, "geom_unsafe_and_margin_safe_rate", (geom_unsafe_for_oracle & margin_safe).float().mean(), batch_size)
            add_metric(accumulator, "geom_safe_and_margin_unsafe_rate", (geom_safe_for_oracle & margin_unsafe).float().mean(), batch_size)
        add_metric(accumulator, "safety_oracle_primary_available_rate", primary_available.float().mean(), batch_size)
        add_metric(accumulator, "safety_oracle_fallback_available_rate", fallback_available.float().mean(), batch_size)
        add_metric(accumulator, "safety_oracle_safe_mask_rate", safety_oracle_mask.float().mean(), batch_size)
        add_metric(accumulator, "safety_oracle_unsafe_mask_rate", final_unsafe.float().mean(), batch_size)
        add_metric(accumulator, "safety_oracle_overlap_rate", (safety_oracle_mask & final_unsafe).float().mean(), batch_size)
        safety_available = safety_oracle_mask.any(dim=1)
        add_metric(accumulator, "safety_constrained_candidate_available_rate", safety_available.float().mean(), batch_size)
        if bool(safety_available.any()):
            safety_source = flat_margin.masked_fill(~safety_oracle_mask, -torch.inf)
            safety_oracle_margin, safety_oracle_id = safety_source.max(dim=1)
            available_weight = int(safety_available.sum().item())
            selected_safety_ok = safety_oracle_mask.gather(1, best_id[:, None]).squeeze(1)
            add_metric(
                accumulator,
                "safety_constrained_oracle_selected_rate",
                (best_id[safety_available] == safety_oracle_id[safety_available]).float().mean(),
                available_weight,
            )
            add_metric(
                accumulator,
                "safety_constrained_safe_missed_rate",
                (~selected_safety_ok[safety_available]).float().mean(),
                available_weight,
            )
            safety_gap_mask = safety_available & selected_finite
            if bool(safety_gap_mask.any()):
                gap_weight = int(safety_gap_mask.sum().item())
                add_metric(
                    accumulator,
                    "safety_constrained_oracle_gap",
                    (safety_oracle_margin[safety_gap_mask] - selected_margin[safety_gap_mask]).mean(),
                    gap_weight,
                )

        if candidate.candidate_type is not None and bool(oracle_valid.any()):
            flat_type = candidate.candidate_type.reshape(batch_size, -1)
            oracle_type = flat_type.gather(1, oracle_id[:, None]).squeeze(1)[oracle_valid]
            oracle_weight = int(oracle_type.numel())
            for type_id, name in TYPE_NAMES.items():
                add_metric(accumulator, f"oracle_{name}_rate", (oracle_type == type_id).float().mean(), oracle_weight)

def evaluate(args):
    apply_eval_stage(args)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)

    state_dict = None
    checkpoint_metadata = {}
    if args.checkpoint:
        state_dict, checkpoint_metadata = load_oarm_checkpoint(args.checkpoint, map_location=device)
    args.yopo_preserve_utility_delta_scale = resolve_utility_delta_scale(args, checkpoint_metadata)
    apply_checkpoint_eval_protocol(args, checkpoint_metadata)

    gt_sampler_options = gt_sampler_options_from_args(args)
    dataset = OARMDataset(
        mode=args.mode,
        dataset_root=args.dataset_root or None,
        use_privileged_risk_filter=args.use_privileged_risk_filter,
        risk_label_source=args.risk_label_source,
        gt_sampler_options=gt_sampler_options,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    policy = OARMNetwork(
        candidate_mode=args.candidate_mode,
        backbone_mode=args.backbone_mode,
        enable_yield_candidates=args.enable_yield_candidates,
        utility_delta_scale=args.yopo_preserve_utility_delta_scale,
    ).to(device)
    if args.checkpoint:
        if args.candidate_mode in {"yopo_preserve", "yopo_preserve_rerank"}:
            is_oarm_preserve_checkpoint = any(key.startswith("preserve_network.") for key in state_dict)
            if is_oarm_preserve_checkpoint:
                validate_checkpoint_metadata(
                    checkpoint_metadata,
                    args.candidate_mode,
                    args.backbone_mode,
                    allow_mismatch=args.allow_checkpoint_mismatch,
                    enable_yield_candidates=args.enable_yield_candidates,
                    deployed_yaw_mode=args.deployed_yaw_mode,
                    risk_label_source=args.risk_label_source,
                    yopo_preserve_utility_delta_scale=args.yopo_preserve_utility_delta_scale,
                )
                policy.load_state_dict(state_dict)
                print(f"Loaded OARM {args.candidate_mode} checkpoint: {args.checkpoint}")
            else:
                policy.preserve_network.load_yopo_state_dict(state_dict, strict=True)
                print(f"Loaded official YOPO checkpoint into {args.candidate_mode} policy: {args.checkpoint}")
        else:
            validate_checkpoint_metadata(
                checkpoint_metadata,
                args.candidate_mode,
                args.backbone_mode,
                allow_mismatch=args.allow_checkpoint_mismatch,
                enable_yield_candidates=args.enable_yield_candidates,
                deployed_yaw_mode=args.deployed_yaw_mode,
                risk_label_source=args.risk_label_source,
                yopo_preserve_utility_delta_scale=args.yopo_preserve_utility_delta_scale,
            )
            policy.load_state_dict(state_dict)
            print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint provided; evaluating randomly initialized OARMNetwork.")
    policy.eval()

    with yopo_dataset_cfg(args.dataset_root or None):
        loss_fn = OARMLoss(
            use_esdf_collision=args.use_esdf_collision,
            use_occlusion_aware_visibility=args.use_occlusion_aware_visibility,
            enable_occlusion_risk=args.eval_occlusion_risk,
            enable_risk_point_guidance=args.eval_risk_point_guidance,
            enable_reaction_margin=args.eval_reaction_margin,
            enable_margin_ranking=args.eval_margin_ranking,
            enable_yaw_visibility=args.eval_yaw_visibility,
            deployed_yaw_mode=args.deployed_yaw_mode,
            enable_yield_feasibility=args.eval_backup_feasibility,
            risk_assoc_distance_m=args.risk_assoc_distance_m,
            risk_assoc_sigma_m=args.risk_assoc_sigma_m,
            risk_arrival_radius_m=args.risk_arrival_radius_m,
        )
        line_of_sight = loss_fn.line_of_sight if args.use_occlusion_aware_visibility else None
    margin_labeler = ReactionMarginLabeler(risk_arrival_radius_m=args.risk_arrival_radius_m)
    accumulator = defaultdict(list)
    seen_batches = 0

    with torch.inference_mode():
        for batch_id, (depth, pos, rot, obs_b, map_id, labels) in enumerate(loader):
            if args.max_batches is not None and batch_id >= args.max_batches:
                break

            depth = depth.to(device)
            pos = pos.to(device)
            rot = rot.to(device)
            obs_b = obs_b.to(device)
            map_id = map_id.to(device)

            candidate = policy.inference(depth, obs_b)
            flat = candidate.flatten()
            flat_labels = flatten_labels(labels, flat, device, args)

            start_state_w, end_state_w, goal_w = build_world_states(pos, rot, obs_b, flat)
            if args.eval_yield_feasibility:
                flat_labels["visible_free_distance"] = visible_free_distance_from_depth(
                    depth,
                    flat["end_state_b"][:, 0:3],
                )
            map_id_expanded = map_id.repeat_interleave(cfg["traj_num"], dim=0)
            flat_labels = maybe_generate_reaction_margin_labels(
                flat_labels,
                flat,
                start_state_w,
                end_state_w,
                map_id_expanded,
                goal_w,
                args,
                margin_labeler,
                line_of_sight,
                loss_fn,
            )
            loss_dict = loss_fn(start_state_w, end_state_w, flat, goal_w, flat_labels, map_id_expanded)
            min_clearance_gt = None
            if args.yopo_preserve_geometry_oracle_source == "gt_clearance":
                sampled_pos_w = loss_dict.get("sampled_pos_w")
                if sampled_pos_w is None:
                    raise RuntimeError("GT clearance eval requires sampled_pos_w from OARMLoss")
                min_clearance_gt = candidate_min_clearance_gt(sampled_pos_w, map_id_expanded, dataset.dataset_dir)

            batch_size = depth.shape[0]
            for key, value in loss_dict.items():
                if torch.is_tensor(value) and value.dim() == 0:
                    add_metric(accumulator, key, value, batch_size)

            selected_progress = -OARMLoss.goal_progress_cost(start_state_w, end_state_w, goal_w, flat["traj_time"]).detach()
            selected_candidate_stats(
                candidate,
                accumulator,
                args,
                flat_labels,
                loss_dict.get("safety_cost_per_candidate"),
                selected_progress,
                min_clearance_gt,
            )

            if "reaction_margin" in flat_labels:
                margin_valid = flat_labels.get("reaction_margin_valid")
                margin_metrics = reaction_margin_metrics(flat_labels["reaction_margin"], valid_mask=margin_valid)
                margin_censored = flat_labels.get("reaction_margin_censored")
                if margin_censored is not None:
                    add_metric(accumulator, "reaction_margin_censored_rate", margin_censored.float().mean(), margin_censored.numel())
                    add_metric(accumulator, "reaction_margin_valid_rate", margin_valid.float().mean() if margin_valid is not None else torch.ones_like(margin_censored, dtype=torch.float32).mean(), margin_censored.numel())
                for key, value in margin_metrics.items():
                    add_metric(accumulator, key, value, flat_labels["reaction_margin"].numel())
                pred_metrics = margin_prediction_metrics(flat["margin_pred"], flat_labels["reaction_margin"], valid_mask=margin_valid)
                for key, value in pred_metrics.items():
                    add_metric(accumulator, key, value, flat_labels["reaction_margin"].numel())
                ranking_metrics = pairwise_ranking_accuracy(
                    flat["utility_score"],
                    flat_labels["reaction_margin"],
                    cfg["traj_num"],
                    margin_delta=oarm_cfg.ranking_margin_delta,
                    valid_mask=margin_valid,
                )
                for key, value in ranking_metrics.items():
                    add_metric(accumulator, key, value, flat_labels["reaction_margin"].numel())

                ranking_progress = -OARMLoss.goal_progress_cost(start_state_w, end_state_w, goal_w, flat['traj_time']).detach()
                ranking_base_cost = OARMLoss.ranking_base_cost_proxy(
                    start_state_w,
                    end_state_w,
                    goal_w,
                    flat['traj_time'],
                ).detach()
                ranking_coeff = quintic_coefficients(start_state_w, end_state_w, flat['traj_time'])
                _, ranking_vel, _, _ = sample_polynomial(ranking_coeff, flat['traj_time'], 30, include_zero=True)
                ranking_speed = ranking_vel.norm(dim=-1).mean(dim=-1).detach()
                matched_ranking_metrics = matched_pairwise_ranking_accuracy(
                    flat["utility_score"],
                    flat_labels["reaction_margin"],
                    cfg["traj_num"],
                    margin_delta=oarm_cfg.ranking_margin_delta,
                    valid_mask=margin_valid,
                    progress=ranking_progress,
                    base_cost=ranking_base_cost,
                    mean_speed=ranking_speed,
                    traj_time=flat["traj_time"],
                    progress_eps=oarm_cfg.ranking_progress_eps,
                    base_cost_eps=oarm_cfg.ranking_base_cost_eps,
                    speed_eps=oarm_cfg.ranking_speed_eps,
                    time_eps=oarm_cfg.ranking_time_eps,
                )
                for key, value in matched_ranking_metrics.items():
                    add_metric(accumulator, key, value, flat_labels["reaction_margin"].numel())
                frontier_score = flat.get("frontier_score")
                disentanglement = margin_disentanglement_metrics(
                    flat_labels["reaction_margin"],
                    flat["utility_score"],
                    valid_mask=margin_valid,
                    frontier_score=frontier_score,
                    duration=flat['traj_time'],
                    progress=ranking_progress,
                )
                for key, value in disentanglement.items():
                    add_metric(accumulator, key, value, flat_labels["reaction_margin"].numel())

            if "occlusion_risk" in flat_labels:
                risk_metrics = risk_calibration_metrics(flat["risk_logit"], flat_labels["occlusion_risk"])
                for key, value in risk_metrics.items():
                    add_metric(accumulator, key, value, flat_labels["occlusion_risk"].numel())

            if "backup_feasible" in flat_labels:
                backup_metrics = backup_feasibility_metrics(
                    flat["backup_logit"],
                    flat_labels["backup_feasible"],
                    flat_labels.get("reaction_margin"),
                )
                for key, value in backup_metrics.items():
                    add_metric(accumulator, key, value, flat_labels["backup_feasible"].numel())

            seen_batches += 1

    metrics = finalize_metrics(accumulator)
    metrics["batches"] = seen_batches
    metrics["samples"] = seen_batches * args.batch_size
    metrics["stage"] = args.stage
    metrics["dataset_root"] = args.dataset_root
    metrics["online_inputs"] = ["depth", "state", "goal"]
    metrics["yield_feasibility_eval"] = bool(args.eval_yield_feasibility)
    metrics["enable_yield_candidates"] = bool(args.enable_yield_candidates)
    metrics["deployed_yaw_mode"] = args.deployed_yaw_mode
    metrics["risk_label_source"] = args.risk_label_source
    metrics["yopo_preserve_utility_delta_scale"] = args.yopo_preserve_utility_delta_scale
    for key in EVAL_PROTOCOL_ARG_KEYS:
        metrics[key] = getattr(args, key)
    for key in GT_SAMPLER_ARG_KEYS:
        metrics[key] = getattr(args, key)
    metrics["eval_yaw_visibility"] = bool(args.eval_yaw_visibility)
    metrics["privileged_training"] = bool(
        args.use_privileged_risk_filter
        or args.use_occlusion_aware_visibility
        or args.use_esdf_collision
        or args.eval_reaction_margin
        or args.eval_risk_point_guidance
    )
    metrics["mapless_online_inference"] = True
    print(json.dumps(metrics, indent=2, sort_keys=True))

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        print(f"Wrote metrics: {args.output}")


def parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage",
        choices=["v0", "v1_occ", "v2_margin", "v3_yield", "a3h", "full"],
        default="v0",
        help="named evaluation preset matching OARM/train_oarm.py",
    )
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--allow-checkpoint-mismatch", action="store_true")
    p.add_argument("--candidate-mode", choices=["yopo", "typed_frontier", "yopo_preserve", "yopo_preserve_rerank"], default="")
    p.add_argument("--backbone-mode", choices=["oarm_light", "yopo_original"], default="")
    p.add_argument("--enable-yield-candidates", action="store_true")
    p.add_argument("--deployed-yaw-mode", choices=["goal", "hold", "predicted"], default="")
    p.add_argument("--risk-label-source", choices=["proxy", "proxy_esdf", "gt_pointcloud"], default="")
    p.add_argument("--yopo-preserve-utility-delta-scale", type=float, default=None)
    p.add_argument("--yopo-preserve-safety-cost-threshold", type=float, default=None)
    p.add_argument("--yopo-preserve-safe-cost-threshold", type=float, default=None)
    p.add_argument("--yopo-preserve-geometry-oracle-source", choices=["esdf_cost", "gt_clearance"], default="")
    p.add_argument("--yopo-preserve-unsafe-clearance-m", type=float, default=None)
    p.add_argument("--yopo-preserve-safe-clearance-m", type=float, default=None)
    p.add_argument("--yopo-preserve-safe-margin-m", type=float, default=None)
    p.add_argument("--yopo-preserve-oracle-min-progress", type=float, default=None)
    p.add_argument("--gt-risk-point-count", type=int, default=None)
    p.add_argument("--gt-hidden-depth-margin-m", type=float, default=None)
    p.add_argument("--gt-min-forward-m", type=float, default=None)
    p.add_argument("--gt-max-forward-m", type=float, default=None)
    p.add_argument("--gt-horizon-fov-expand-deg", type=float, default=None)
    p.add_argument("--gt-vertical-fov-expand-deg", type=float, default=None)
    p.add_argument("--gt-depth-metric", choices=["", "forward", "ray"], default="")
    p.add_argument("--gt-reachable-forward-center-m", type=float, default=None)
    p.add_argument("--gt-reachable-forward-sigma-m", type=float, default=None)
    p.add_argument("--gt-reachable-lateral-sigma-m", type=float, default=None)
    p.add_argument("--gt-reachable-vertical-sigma-m", type=float, default=None)
    p.add_argument("--gt-reachable-score-weight", type=float, default=None)
    p.add_argument("--gt-side-score-weight", type=float, default=None)
    p.add_argument("--gt-risk-nms-radius-m", type=float, default=None)
    p.add_argument("--gt-risk-voxel-size-m", type=float, default=None)
    p.add_argument("--risk-assoc-distance-m", type=float, default=None)
    p.add_argument("--risk-assoc-sigma-m", type=float, default=None)
    p.add_argument("--risk-arrival-radius-m", type=float, default=None)
    p.add_argument("--mode", choices=["train", "valid"], default="valid")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--dataset-root", type=str, default="")
    p.add_argument("--eval-occlusion-risk", action="store_true")
    p.add_argument("--eval-reaction-margin", action="store_true")
    p.add_argument("--eval-margin-ranking", action="store_true")
    p.add_argument("--eval-risk-point-guidance", action="store_true")
    p.add_argument("--eval-yaw-visibility", action="store_true")
    p.add_argument("--use-weak-margin-label", action="store_true")
    p.add_argument("--eval-backup-feasibility", action="store_true")
    p.add_argument("--eval-yield-feasibility", action="store_true")
    p.add_argument("--use-esdf-collision", action="store_true")
    p.add_argument("--use-occlusion-aware-visibility", action="store_true")
    p.add_argument("--use-privileged-risk-filter", action="store_true")
    p.add_argument("--output", type=str, default="")
    return p


def apply_eval_stage(args):
    preset = get_oarm_training_preset(args.stage)
    if not args.candidate_mode:
        args.candidate_mode = preset.candidate_mode
    if not args.backbone_mode:
        args.backbone_mode = preset.backbone_mode
    if not args.deployed_yaw_mode:
        args.deployed_yaw_mode = preset.deployed_yaw_mode
    if not args.risk_label_source:
        args.risk_label_source = preset.risk_label_source
    for key in GT_SAMPLER_ARG_KEYS:
        value = getattr(args, key)
        if value is None or value == "":
            setattr(args, key, getattr(preset, key, getattr(oarm_cfg, key)))
    if preset.enable_yield_candidates and not args.enable_yield_candidates:
        args.enable_yield_candidates = True
    for eval_key, preset_key in EVAL_STAGE_MAP.items():
        if getattr(preset, preset_key) and not getattr(args, eval_key):
            setattr(args, eval_key, True)
    if preset.train_yaw_visibility and not args.eval_yaw_visibility:
        args.eval_yaw_visibility = True
    if args.eval_backup_feasibility:
        args.eval_yield_feasibility = True
    if args.eval_yield_feasibility:
        args.eval_backup_feasibility = True


if __name__ == "__main__":
    evaluate(parser().parse_args())

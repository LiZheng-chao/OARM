import argparse
import tempfile

import torch

from OARM.config import get_oarm_training_preset
from OARM.loss import OARMLoss
from OARM.policy.oarm_network import OARMNetwork
from OARM.policy.oarm_rm_critic import risk_probability_from_window
from OARM.policy.oarm_trainer import OARMTrainer
from OARM.train_oarm import parser, resolve_training_options
from OARM.utils.checkpoint import make_oarm_checkpoint, validate_checkpoint_metadata
from OARM.visibility.reaction_margin_labeler import ReactionMarginLabeler


def check_preset_route():
    preset = get_oarm_training_preset("oarm3_s2_prob_rm")
    assert preset.candidate_mode == "yopo_preserve"
    assert preset.backbone_mode == "yopo_original"
    assert preset.train_probabilistic_rm_critic is True
    assert preset.train_reaction_margin is False
    assert preset.train_margin_ranking is False

    args = parser().parse_args(["--stage", "oarm3_s2_prob_rm"])
    options = resolve_training_options(args)
    assert options["train_probabilistic_rm_critic"] is True
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
    assert loss_dict["rm_critic_valid_rate"].item() > 0.99
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
    check_trainable_contract()
    check_end_to_end_mini_batch()
    print("oarm3_s2_smoke ok")


if __name__ == "__main__":
    main()

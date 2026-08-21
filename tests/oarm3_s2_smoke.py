import argparse

import torch

from OARM.config import get_oarm_training_preset
from OARM.loss import OARMLoss
from OARM.policy.oarm_network import OARMNetwork
from OARM.policy.oarm_trainer import OARMTrainer
from OARM.train_oarm import parser, resolve_training_options
from OARM.utils.checkpoint import make_oarm_checkpoint, validate_checkpoint_metadata


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


def main():
    check_preset_route()
    check_bad_route_rejected()
    check_checkpoint_metadata()
    check_loss_fail_fast()
    check_trainable_contract()
    print("oarm3_s2_smoke ok")


if __name__ == "__main__":
    main()

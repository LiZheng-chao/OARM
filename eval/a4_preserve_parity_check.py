import argparse
import json
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
YOPO_DIR = os.path.join(REPO_ROOT, "YOPO")
if YOPO_DIR not in sys.path:
    sys.path.insert(0, YOPO_DIR)

from OARM.policy.oarm_candidate_generator import OARMCandidateGenerator
from OARM.policy.oarm_network import OARMNetwork
from OARM.utils.yopo_compat import ensure_yopo_path

ensure_yopo_path()
from config.config import cfg


def parser():
    p = argparse.ArgumentParser(description="Check A4a preserve parity and deterministic brake semantics.")
    p.add_argument("--yopo-checkpoint", required=True)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--output", default="")
    p.add_argument("--atol", type=float, default=1e-6)
    return p


def load_yopo(policy, checkpoint, device):
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    policy.preserve_network.load_yopo_state_dict(state_dict, strict=True)


def max_abs(a, b):
    return float((a - b).abs().max().detach().cpu())


def main(args):
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    torch.manual_seed(7)
    batch = args.batch_size
    depth = torch.rand(batch, 1, cfg["image_height"], cfg["image_width"], device=device)
    obs = torch.randn(batch, 9, device=device)
    obs[:, 0] = torch.linspace(0.5, 4.0, batch, device=device)
    obs[:, 1:6] *= 0.1
    obs[:, 6:9] = torch.tensor([cfg["goal_length"], 0.0, 0.0], device=device)

    yopo = OARMNetwork(candidate_mode="yopo_preserve", backbone_mode="yopo_original").to(device).eval()
    a4a = OARMNetwork(candidate_mode="a4_preserve_brake", backbone_mode="yopo_original").to(device).eval()
    load_yopo(yopo, args.yopo_checkpoint, device)
    load_yopo(a4a, args.yopo_checkpoint, device)

    with torch.inference_mode():
        base = yopo.inference(depth, obs)
        aug = a4a.inference(depth, obs)

    n_base = base.utility_score.numel() // batch
    n_aug = aug.utility_score.numel() // batch
    base_flat = base.flatten()
    aug_flat = aug.flatten()

    progress = {
        "end_state_b_max_abs": max_abs(
            aug_flat["end_state_b"].reshape(batch, n_aug, 9)[:, :n_base],
            base_flat["end_state_b"].reshape(batch, n_base, 9),
        ),
        "traj_time_max_abs": max_abs(
            aug_flat["traj_time"].reshape(batch, n_aug)[:, :n_base],
            base_flat["traj_time"].reshape(batch, n_base),
        ),
        "yaw_terminal_max_abs": max_abs(
            aug_flat["yaw_terminal"].reshape(batch, n_aug)[:, :n_base],
            base_flat["yaw_terminal"].reshape(batch, n_base),
        ),
        "utility_base_max_abs": max_abs(
            aug_flat["utility_base"].reshape(batch, n_aug)[:, :n_base],
            base_flat["utility_base"].reshape(batch, n_base),
        ),
        "utility_score_initial_max_abs": max_abs(
            aug_flat["utility_score"].reshape(batch, n_aug)[:, :n_base],
            base_flat["utility_score"].reshape(batch, n_base),
        ),
    }
    type_group = aug_flat["candidate_type"].reshape(batch, n_aug)
    brake = aug_flat["end_state_b"].reshape(batch, n_aug, 9)[:, -1]
    brake_type = type_group[:, -1]
    metrics = {
        "candidate_count_base": n_base,
        "candidate_count_a4a": n_aug,
        "progress_candidate_count": n_base,
        "appended_candidate_count": n_aug - n_base,
        "progress_parity": progress,
        "progress_types_all_progress": bool((type_group[:, :n_base] == OARMCandidateGenerator.PROGRESS).all().item()),
        "brake_type_all_brake": bool((brake_type == OARMCandidateGenerator.BRAKE).all().item()),
        "brake_terminal_speed_max": float(brake[:, 3:6].norm(dim=1).max().detach().cpu()),
        "brake_terminal_acc_max": float(brake[:, 6:9].norm(dim=1).max().detach().cpu()),
        "brake_time_min": float(aug_flat["traj_time"].reshape(batch, n_aug)[:, -1].min().detach().cpu()),
        "brake_time_max": float(aug_flat["traj_time"].reshape(batch, n_aug)[:, -1].max().detach().cpu()),
    }
    metrics["passed"] = (
        n_base == int(cfg["traj_num"])
        and n_aug == int(cfg["traj_num"]) + 1
        and all(value <= args.atol for value in progress.values())
        and metrics["progress_types_all_progress"]
        and metrics["brake_type_all_brake"]
        and metrics["brake_terminal_speed_max"] <= args.atol
        and metrics["brake_terminal_acc_max"] <= args.atol
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
    if not metrics["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main(parser().parse_args())

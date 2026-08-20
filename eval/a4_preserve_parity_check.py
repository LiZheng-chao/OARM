import argparse
import json
import os
import sys

import numpy as np
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
from policy.poly_solver import Poly5Solver


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


def max_abs_first15(base, aug, batch, n_base, n_aug):
    base_flat = base.flatten()
    aug_flat = aug.flatten()
    return {
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


def set_brake_gate_bias(policy, bias):
    head = policy.preserve_network.brake_gate_head[-1]
    with torch.no_grad():
        head.weight.zero_()
        head.bias.fill_(float(bias))


def selected_ids(candidate, batch):
    n = candidate.utility_score.numel() // batch
    return candidate.flatten()["utility_score"].reshape(batch, n).argmax(dim=1)


class _SpatialRampGate(torch.nn.Module):
    def forward(self, features):
        b, _c, v, h = features.shape
        ramp = torch.arange(v * h, device=features.device, dtype=features.dtype).reshape(1, 1, v, h)
        return ramp.expand(b, -1, -1, -1)


def check_top1_gate_gather(policy, depth, obs, batch):
    original_head = policy.preserve_network.brake_gate_head
    try:
        policy.preserve_network.brake_gate_head = _SpatialRampGate().to(depth.device)
        with torch.inference_mode():
            candidate = policy.inference(depth, obs)
        n = candidate.utility_score.numel() // batch
        flat = candidate.flatten()
        utility_base = flat["utility_base"].reshape(batch, n)
        top1_id = utility_base[:, : n - 1].argmax(dim=1)
        brake_gate = flat["utility_delta"].reshape(batch, n)[:, -1]
        expected = top1_id.to(device=brake_gate.device, dtype=brake_gate.dtype)
        max_abs_error = float((brake_gate - expected).abs().max().detach().cpu())
        return {
            "passed": max_abs_error <= 1e-6,
            "max_abs_error": max_abs_error,
            "top1_id": top1_id.detach().cpu().tolist(),
            "brake_gate": brake_gate.detach().cpu().tolist(),
        }
    finally:
        policy.preserve_network.brake_gate_head = original_head


def check_brake_physics(policy, device):
    speeds = torch.tensor([0.0, 0.05, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0], device=device)
    obs = torch.zeros(speeds.numel(), 9, device=device)
    obs[:, 0] = speeds
    end_state, brake_time = policy.preserve_network.deterministic_brake_candidate(obs, torch.float32, device)
    end_state = end_state.reshape(speeds.numel(), 9).detach().cpu().numpy()
    brake_time = brake_time.reshape(speeds.numel()).detach().cpu().numpy()
    acc_max = float(policy.preserve_network.lattice_primitive.acc_max)
    worst = {
        "min_forward_velocity": float("inf"),
        "max_acceleration": 0.0,
        "max_overshoot": 0.0,
        "terminal_speed_max": 0.0,
        "terminal_acc_max": 0.0,
    }
    for i, speed in enumerate(speeds.detach().cpu().numpy()):
        tf = float(brake_time[i])
        px = Poly5Solver(0.0, float(speed), 0.0, float(end_state[i, 0]), 0.0, 0.0, tf)
        ts = np.linspace(0.0, tf, 201)
        x = px.get_position(ts)
        v = px.get_velocity(ts)
        a = px.get_acceleration(ts)
        worst["min_forward_velocity"] = min(worst["min_forward_velocity"], float(v.min()))
        worst["max_acceleration"] = max(worst["max_acceleration"], float(np.abs(a).max()))
        worst["max_overshoot"] = max(worst["max_overshoot"], float(x.max() - end_state[i, 0]))
        worst["terminal_speed_max"] = max(worst["terminal_speed_max"], abs(float(v[-1])))
        worst["terminal_acc_max"] = max(worst["terminal_acc_max"], abs(float(a[-1])))
    worst["passed"] = (
        worst["min_forward_velocity"] >= -1e-5
        and worst["max_overshoot"] <= 1e-5
        and worst["max_acceleration"] <= acc_max * 1.05
        and worst["terminal_speed_max"] <= 1e-5
        and worst["terminal_acc_max"] <= 1e-5
    )
    return worst


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

    progress = max_abs_first15(base, aug, batch, n_base, n_aug)
    type_group = aug_flat["candidate_type"].reshape(batch, n_aug)
    brake = aug_flat["end_state_b"].reshape(batch, n_aug, 9)[:, -1]
    brake_type = type_group[:, -1]
    base_selected_id = selected_ids(base, batch)
    aug_selected_id = selected_ids(aug, batch)
    selected_id_parity = bool((base_selected_id == aug_selected_id).all().item())
    set_brake_gate_bias(a4a, -10.0)
    with torch.inference_mode():
        gate_negative = a4a.inference(depth, obs)
    negative_ids = selected_ids(gate_negative, batch)
    negative_chooses_progress = bool((negative_ids < n_base).all().item())

    set_brake_gate_bias(a4a, 10.0)
    with torch.inference_mode():
        gate_positive = a4a.inference(depth, obs)
    positive_ids = selected_ids(gate_positive, batch)
    positive_chooses_brake = bool((positive_ids == n_aug - 1).all().item())

    top1_gate_gather = check_top1_gate_gather(a4a, depth, obs, batch)

    optimizer = torch.optim.SGD(a4a.preserve_network.brake_gate_head.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    dummy_loss = torch.zeros((), device=device)
    for param in a4a.preserve_network.brake_gate_head.parameters():
        dummy_loss = dummy_loss + param.sum()
    dummy_loss.backward()
    optimizer.step()
    with torch.inference_mode():
        aug_after_step = a4a.inference(depth, obs)
    optimizer_step_progress = max_abs_first15(base, aug_after_step, batch, n_base, n_aug)
    brake_physics = check_brake_physics(a4a, device)

    metrics = {
        "candidate_count_base": n_base,
        "candidate_count_a4a": n_aug,
        "progress_candidate_count": n_base,
        "appended_candidate_count": n_aug - n_base,
        "progress_parity": progress,
        "progress_types_all_progress": bool((type_group[:, :n_base] == OARMCandidateGenerator.PROGRESS).all().item()),
        "selected_id_parity": selected_id_parity,
        "base_selected_id": base_selected_id.detach().cpu().tolist(),
        "a4a_initial_selected_id": aug_selected_id.detach().cpu().tolist(),
        "brake_type_all_brake": bool((brake_type == OARMCandidateGenerator.BRAKE).all().item()),
        "brake_terminal_speed_max": float(brake[:, 3:6].norm(dim=1).max().detach().cpu()),
        "brake_terminal_acc_max": float(brake[:, 6:9].norm(dim=1).max().detach().cpu()),
        "brake_time_min": float(aug_flat["traj_time"].reshape(batch, n_aug)[:, -1].min().detach().cpu()),
        "brake_time_max": float(aug_flat["traj_time"].reshape(batch, n_aug)[:, -1].max().detach().cpu()),
        "gate_negative_chooses_progress": negative_chooses_progress,
        "gate_positive_chooses_brake": positive_chooses_brake,
        "top1_gate_gather": top1_gate_gather,
        "optimizer_step_progress_parity": optimizer_step_progress,
        "brake_physics": brake_physics,
    }
    metrics["passed"] = (
        n_base == int(cfg["traj_num"])
        and n_aug == int(cfg["traj_num"]) + 1
        and all(value <= args.atol for value in progress.values())
        and metrics["progress_types_all_progress"]
        and metrics["selected_id_parity"]
        and metrics["brake_type_all_brake"]
        and metrics["brake_terminal_speed_max"] <= args.atol
        and metrics["brake_terminal_acc_max"] <= args.atol
        and metrics["gate_negative_chooses_progress"]
        and metrics["gate_positive_chooses_brake"]
        and metrics["top1_gate_gather"]["passed"]
        and all(value <= args.atol for value in metrics["optimizer_step_progress_parity"].values())
        and metrics["brake_physics"]["passed"]
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

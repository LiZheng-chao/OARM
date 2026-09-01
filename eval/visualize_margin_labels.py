import argparse
import json
import math
import os

import torch

from OARM.config import oarm_cfg
from OARM.dataset import OARMDataset
from OARM.policy.oarm_candidate_generator import OARMCandidateGenerator
from OARM.policy.oarm_network import OARMNetwork
from OARM.loss import OARMLoss
from OARM.policy.oarm_poly_solver import quintic_coefficients, sample_polynomial
from OARM.policy.oarm_state_transform import rotate_body2world, state_body2world
from OARM.utils.checkpoint import load_oarm_checkpoint, validate_checkpoint_metadata
from OARM.visibility.esdf_visibility import ESDFLineOfSight
from OARM.visibility.reaction_margin_labeler import ReactionMarginLabeler
from OARM.utils.yopo_compat import ensure_yopo_path

ensure_yopo_path()
from config.config import cfg


TYPE_NAMES = {
    OARMCandidateGenerator.PROGRESS: "progress",
    OARMCandidateGenerator.PROBE: "probe",
    OARMCandidateGenerator.BRAKE: "brake",
    OARMCandidateGenerator.YIELD: "yield",
}


def checkpoint_training_option(metadata, key, default=None):
    training_options = (metadata or {}).get("training_options") or {}
    if key in training_options and training_options[key] is not None:
        return training_options[key]
    if metadata and metadata.get(key) is not None:
        return metadata[key]
    return default


def load_policy(
    checkpoint,
    device,
    candidate_mode,
    backbone_mode,
    allow_checkpoint_mismatch,
    enable_yield_candidates=False,
    deployed_yaw_mode="goal",
    enable_rm_critic=False,
):
    state_dict = None
    checkpoint_metadata = {}
    if checkpoint:
        state_dict, checkpoint_metadata = load_oarm_checkpoint(checkpoint, map_location=device)
    rm_critic_hazard_bins = int(checkpoint_training_option(
        checkpoint_metadata,
        "rm_critic_hazard_bins",
        oarm_cfg.rm_critic_hazard_bins,
    ))
    rm_critic_hazard_max_time_s = float(checkpoint_training_option(
        checkpoint_metadata,
        "rm_critic_hazard_max_time_s",
        oarm_cfg.rm_critic_hazard_max_time_s,
    ))
    policy = OARMNetwork(
        candidate_mode=candidate_mode,
        backbone_mode=backbone_mode,
        enable_yield_candidates=enable_yield_candidates,
        enable_rm_critic=enable_rm_critic,
        rm_critic_hazard_bins=rm_critic_hazard_bins,
        rm_critic_hazard_max_time_s=rm_critic_hazard_max_time_s,
    ).to(device)
    if checkpoint:
        validate_checkpoint_metadata(
            checkpoint_metadata,
            candidate_mode,
            backbone_mode,
            allow_mismatch=allow_checkpoint_mismatch,
            enable_yield_candidates=enable_yield_candidates,
            deployed_yaw_mode=deployed_yaw_mode,
        )
        policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def sampled_time_grid(traj_time, eval_points, include_zero=True):
    start = 0.0 if include_zero else 1.0 / eval_points
    tau = torch.linspace(start, 1.0, eval_points, device=traj_time.device, dtype=traj_time.dtype)
    return traj_time[:, None] * tau[None, :]


def build_margin_labels(policy, dataset, sample_id, device, args):
    depth, pos, rot, obs_b, map_id, labels = dataset[sample_id]
    depth = depth.to(device).unsqueeze(0)
    pos = torch.as_tensor(pos, dtype=torch.float32, device=device).unsqueeze(0)
    rot = torch.as_tensor(rot, dtype=torch.float32, device=device).unsqueeze(0)
    obs_b = torch.as_tensor(obs_b, dtype=torch.float32, device=device).unsqueeze(0)
    map_id = torch.as_tensor(map_id, dtype=torch.long, device=device).reshape(1)

    with torch.inference_mode():
        candidate = policy.inference(depth, obs_b)
        flat = candidate.flatten()

        traj_num = cfg["traj_num"]
        goal_w_single = rotate_body2world(rot, obs_b[:, 6:9])
        start_vel_w = rotate_body2world(rot, obs_b[:, 0:3])
        start_acc_w = rotate_body2world(rot, obs_b[:, 3:6])
        start_state_w = torch.stack([pos, start_vel_w, start_acc_w], dim=1).repeat_interleave(traj_num, dim=0)
        goal_w = goal_w_single.repeat_interleave(traj_num, dim=0)

        pos_expanded = pos.repeat_interleave(traj_num, dim=0)
        rot_expanded = rot.repeat_interleave(traj_num, dim=0)
        end_pos_w, end_vel_w, end_acc_w = state_body2world(
            pos_expanded,
            rot_expanded,
            flat["end_state_b"][:, 0:3],
            flat["end_state_b"][:, 3:6],
            flat["end_state_b"][:, 6:9],
        )
        end_state_w = torch.stack([end_pos_w, end_vel_w, end_acc_w], dim=1)

        coeff = quintic_coefficients(start_state_w, end_state_w, flat["traj_time"])
        sampled_pos_w, sampled_vel_w, _, _ = sample_polynomial(coeff, flat["traj_time"], args.eval_points, include_zero=True)
        sampled_time = sampled_time_grid(flat["traj_time"], args.eval_points, include_zero=True)

        risk_points_w = labels["risk_points_w"].to(device).unsqueeze(0).repeat_interleave(traj_num, dim=0)
        risk_weight = labels["risk_weight"].to(device).unsqueeze(0).repeat_interleave(traj_num, dim=0)
        yaw0 = labels.get("yaw0", torch.zeros((), dtype=torch.float32)).to(device).reshape(1).repeat_interleave(traj_num)
        yaw_rate0 = labels.get("yaw_rate0", torch.zeros((), dtype=torch.float32)).to(device).reshape(1).repeat_interleave(traj_num)
        yaw_helper = OARMLoss(deployed_yaw_mode=args.deployed_yaw_mode)
        yaw_ref, _ = yaw_helper.deployed_yaw_reference(
            yaw0,
            yaw_rate0,
            flat["yaw_terminal"],
            flat["traj_time"],
            sampled_pos_w,
            sampled_vel_w,
            sampled_time,
            goal_w,
        )

        visibility_mask = None
        if args.use_occlusion_aware_visibility:
            line_of_sight = ESDFLineOfSight(device=device)
            visibility_mask = line_of_sight(sampled_pos_w, risk_points_w, map_id.repeat_interleave(traj_num))

        labeler = ReactionMarginLabeler()
        margin_labels = labeler(sampled_pos_w, sampled_time, yaw_ref, risk_points_w, risk_weight, visibility_mask=visibility_mask)
        first_vis = margin_labels["first_visible_time"]
        arrival = margin_labels["first_entry_time"]

    return {
        "depth": depth.squeeze(0).detach().cpu(),
        "frontier_map": labels["frontier_map"].detach().cpu(),
        "risk_points_w": labels["risk_points_w"].detach().cpu(),
        "risk_weight": labels["risk_weight"].detach().cpu(),
        "start_pos_w": pos.squeeze(0).detach().cpu(),
        "sampled_pos_w": sampled_pos_w.detach().cpu(),
        "utility": flat["utility_score"].detach().cpu(),
        "candidate_type": flat.get("candidate_type", torch.zeros(traj_num, dtype=torch.long, device=device)).detach().cpu(),
        "traj_time": flat["traj_time"].detach().cpu(),
        "margin_pred": flat["margin_pred"].detach().cpu(),
        "margin_label": margin_labels["reaction_margin_softmin"].detach().cpu(),
        "margin_min": margin_labels["reaction_margin_min"].detach().cpu(),
        "margin_valid": margin_labels["reaction_margin_valid"].detach().cpu(),
        "margin_censored": margin_labels["reaction_margin_censored"].detach().cpu(),
        "first_visible_time_min": first_vis.detach().cpu().amin(dim=-1),
        "arrival_time_min": arrival.detach().cpu().amin(dim=-1),
        "use_occlusion_aware_visibility": bool(args.use_occlusion_aware_visibility),
        "risk_label_source": args.risk_label_source,
        "deployed_yaw_mode": args.deployed_yaw_mode,
    }
def render_sample(data, output_png, output_json, top_k):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    utility = data["utility"]
    selected = int(torch.argmax(utility))
    top_ids = torch.topk(utility, k=min(top_k, utility.numel())).indices.tolist()
    candidate_ids = sorted(set(top_ids + [selected]))
    start = data["start_pos_w"]
    risk_xy = data["risk_points_w"][:, :2] - start[:2]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].imshow(data["depth"].squeeze(0), cmap="gray")
    axes[0].set_title("depth")
    axes[0].axis("off")

    axes[1].imshow(data["frontier_map"].squeeze(0), cmap="magma")
    axes[1].set_title("frontier / risk proxy")
    axes[1].axis("off")

    type_colors = {
        "progress": "tab:blue",
        "probe": "tab:green",
        "brake": "tab:orange",
        "yield": "tab:red",
    }
    for idx in candidate_ids:
        ctype = TYPE_NAMES.get(int(data["candidate_type"][idx]), str(int(data["candidate_type"][idx])))
        xy = data["sampled_pos_w"][idx, :, :2] - start[:2]
        linewidth = 2.8 if idx == selected else 1.2
        label = f"{idx}:{ctype} m={float(data['margin_label'][idx]):.2f}"
        axes[2].plot(xy[:, 0], xy[:, 1], color=type_colors.get(ctype, "k"), linewidth=linewidth, label=label)
    if risk_xy.numel() > 0:
        axes[2].scatter(
            risk_xy[:, 0],
            risk_xy[:, 1],
            c=data["risk_weight"],
            cmap="Reds",
            edgecolors="black",
            linewidths=0.3,
            s=35,
            label="risk points",
        )
    axes[2].scatter([0.0], [0.0], marker="x", color="black", label="start")
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].set_title("candidate trajectories")
    axes[2].set_xlabel("x rel. [m]")
    axes[2].set_ylabel("y rel. [m]")
    axes[2].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)

    rows = []
    for idx in range(utility.numel()):
        ctype = TYPE_NAMES.get(int(data["candidate_type"][idx]), str(int(data["candidate_type"][idx])))
        rows.append(
            {
                "candidate_id": int(idx),
                "candidate_type": ctype,
                "selected": bool(idx == selected),
                "utility": float(data["utility"][idx]),
                "traj_time": float(data["traj_time"][idx]),
                "margin_pred": float(data["margin_pred"][idx]),
                "reaction_margin": float(data["margin_label"][idx]),
                "reaction_margin_min": float(data["margin_min"][idx]),
                "reaction_margin_valid": bool(data["margin_valid"][idx]),
                "reaction_margin_censored": bool(data["margin_censored"][idx]),
                "first_visible_time_min": float(data["first_visible_time_min"][idx]),
                "arrival_time_min": float(data["arrival_time_min"][idx]),
            }
        )
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({
            "selected_id": selected,
            "risk_label_source": data["risk_label_source"],
            "deployed_yaw_mode": data["deployed_yaw_mode"],
            "use_occlusion_aware_visibility": data["use_occlusion_aware_visibility"],
            "candidates": rows,
        }, f, indent=2, sort_keys=True)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--candidate-mode", choices=["yopo", "typed_frontier", "yopo_preserve", "yopo_preserve_rerank", "a4_preserve_brake"], default="typed_frontier")
    p.add_argument("--backbone-mode", choices=["oarm_light", "yopo_original"], default="yopo_original")
    p.add_argument("--enable-yield-candidates", action="store_true")
    p.add_argument("--enable-rm-critic", action="store_true")
    p.add_argument("--deployed-yaw-mode", choices=["goal", "hold", "predicted"], default="goal")
    p.add_argument("--allow-checkpoint-mismatch", action="store_true")
    p.add_argument("--mode", choices=["train", "valid"], default="valid")
    p.add_argument("--dataset-root", type=str, default="")
    p.add_argument("--risk-label-source", choices=["proxy", "proxy_esdf", "gt_pointcloud"], default="gt_pointcloud")
    p.add_argument("--use-occlusion-aware-visibility", action="store_true")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--eval-points", type=int, default=40)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--use-privileged-risk-filter", action="store_true")
    p.add_argument("--output-dir", type=str, default="OARM/results/margin_label_viz")
    return p


def main(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = OARMDataset(
        mode=args.mode,
        dataset_root=args.dataset_root or None,
        use_privileged_risk_filter=args.use_privileged_risk_filter,
        risk_label_source=args.risk_label_source,
    )
    policy = load_policy(
        args.checkpoint,
        device,
        args.candidate_mode,
        args.backbone_mode,
        args.allow_checkpoint_mismatch,
        enable_yield_candidates=args.enable_yield_candidates,
        deployed_yaw_mode=args.deployed_yaw_mode,
        enable_rm_critic=args.enable_rm_critic,
    )
    end = min(args.sample + args.count, len(dataset))
    for sample_id in range(args.sample, end):
        data = build_margin_labels(policy, dataset, sample_id, device, args)
        stem = f"sample_{sample_id:06d}"
        render_sample(
            data,
            os.path.join(args.output_dir, f"{stem}.png"),
            os.path.join(args.output_dir, f"{stem}.json"),
            args.top_k,
        )
        print(f"wrote {stem}")


if __name__ == "__main__":
    main(parser().parse_args())

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from OARM.config import oarm_cfg
from OARM.dataset import OARMDataset
from OARM.eval.eval_dataset import build_world_states, flatten_labels
from OARM.loss import OARMLoss
from OARM.policy.oarm_network import OARMNetwork
from OARM.utils.checkpoint import load_oarm_checkpoint
from OARM.utils.gt_clearance import candidate_min_clearance_gt
from OARM.utils.yopo_compat import ensure_yopo_path
from OARM.utils.yopo_dataset_context import resolve_dataset_dir, yopo_dataset_cfg

ensure_yopo_path()
from config.config import cfg


def _training_option(metadata, key, default=None):
    opts = (metadata or {}).get("training_options") or {}
    if key in opts and opts[key] is not None:
        return opts[key]
    return (metadata or {}).get(key, default)


def _load_policy(path, candidate_mode, device):
    state_dict, metadata = load_oarm_checkpoint(path, map_location=device)
    scale = float(_training_option(metadata, "yopo_preserve_utility_delta_scale", oarm_cfg.yopo_preserve_utility_delta_scale))
    policy = OARMNetwork(
        candidate_mode=candidate_mode,
        backbone_mode="yopo_original",
        utility_delta_scale=scale,
    ).to(device)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy, metadata


def _to_device(batch, device):
    depth, pos, rot, obs_b, map_id, labels = batch
    labels = {k: v.to(device) if torch.is_tensor(v) else v for k, v in labels.items()}
    return depth.to(device), pos.to(device), rot.to(device), obs_b.to(device), map_id.to(device), labels


def _forward_bundle(policy, depth, pos, rot, obs_b, map_id, labels, loss_fn, dataset_root):
    candidate = policy.inference(depth, obs_b)
    flat = candidate.flatten()
    start_state_w, end_state_w, goal_w = build_world_states(pos, rot, obs_b, flat)
    traj_num = cfg["traj_num"]
    map_id_expanded = map_id.repeat_interleave(traj_num)
    flat_labels = argparse.Namespace(
        eval_occlusion_risk=True,
        eval_reaction_margin=True,
        use_weak_margin_label=False,
        eval_yield_feasibility=False,
        eval_backup_feasibility=False,
        eval_risk_point_guidance=True,
    )
    label_dict = flatten_labels(labels, flat, depth.device, flat_labels)
    loss_dict = loss_fn(start_state_w, end_state_w, flat, goal_w, label_dict, map_id_expanded)
    sampled_pos_w = loss_dict["sampled_pos_w"]
    min_clearance = candidate_min_clearance_gt(sampled_pos_w, map_id_expanded, dataset_root).reshape(-1, traj_num)
    utility_score = flat.get("utility_score").reshape(-1, traj_num)
    utility_base = flat.get("utility_base", flat.get("utility_score")).reshape(-1, traj_num)
    utility_delta = flat.get("utility_delta", torch.zeros_like(flat["traj_time"])).reshape(-1, traj_num)
    return {
        "flat": flat,
        "end_state_b": flat["end_state_b"].reshape(-1, traj_num, 9),
        "traj_time": flat["traj_time"].reshape(-1, traj_num),
        "margin_pred": flat.get("margin_pred", torch.zeros_like(flat["traj_time"])).reshape(-1, traj_num),
        "risk_logit": flat.get("risk_logit", torch.zeros_like(flat["traj_time"])).reshape(-1, traj_num),
        "utility_base": utility_base,
        "utility_delta": utility_delta,
        "utility_score": utility_score,
        "selected_id": utility_score.argmax(dim=1),
        "gt_min_clearance": min_clearance,
        "reaction_margin_label": label_dict.get("reaction_margin"),
        "reaction_margin_valid": label_dict.get("reaction_margin_valid"),
    }


def _max_abs(a, b):
    diff = (a - b).detach()
    finite = torch.isfinite(diff)
    if not bool(finite.any()):
        return None
    return float(diff[finite].abs().max().cpu())


def _mean_abs(a):
    finite = torch.isfinite(a)
    if not bool(finite.any()):
        return None
    return float(a[finite].abs().mean().cpu())


def _rate(mask):
    return float(mask.float().mean().detach().cpu()) if mask.numel() else None


def run(args):
    dataset_root = resolve_dataset_dir(args.dataset_root)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    with yopo_dataset_cfg(dataset_root):
        loader = DataLoader(
            OARMDataset(mode=args.mode, dataset_root=dataset_root, use_privileged_risk_filter=True, risk_label_source="gt_pointcloud"),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        batch = None
        for idx, item in enumerate(loader):
            if idx == args.batch_index:
                batch = item
                break
        if batch is None:
            raise ValueError(f"batch_index {args.batch_index} is outside the {args.mode} loader")
        depth, pos, rot, obs_b, map_id, labels = _to_device(batch, device)
        a1, a1_meta = _load_policy(args.a1_checkpoint, "yopo_preserve", device)
        a3h, a3h_meta = _load_policy(args.a3h_checkpoint, "yopo_preserve_rerank", device)
        loss_fn = OARMLoss(enable_occlusion_risk=True, enable_risk_point_guidance=True, enable_reaction_margin=True, enable_margin_ranking=True).to(device)
        with torch.inference_mode():
            a1_out = _forward_bundle(a1, depth, pos, rot, obs_b.clone(), map_id, labels, loss_fn, dataset_root)
            a3h_out = _forward_bundle(a3h, depth, pos, rot, obs_b.clone(), map_id, labels, loss_fn, dataset_root)

    label = a1_out["reaction_margin_label"]
    label_valid = a1_out["reaction_margin_valid"]
    if label is not None:
        if label_valid is not None:
            label_valid = label_valid.bool() & torch.isfinite(label)
        else:
            label_valid = torch.isfinite(label)
    result = {
        "dataset_root": str(dataset_root),
        "mode": args.mode,
        "batch_index": args.batch_index,
        "batch_size": args.batch_size,
        "traj_num": cfg["traj_num"],
        "a1_checkpoint": args.a1_checkpoint,
        "a3h_checkpoint": args.a3h_checkpoint,
        "a1_candidate_mode": _training_option(a1_meta, "candidate_mode"),
        "a3h_candidate_mode": _training_option(a3h_meta, "candidate_mode"),
        "end_state_b_max_abs_diff": _max_abs(a1_out["end_state_b"], a3h_out["end_state_b"]),
        "traj_time_max_abs_diff": _max_abs(a1_out["traj_time"], a3h_out["traj_time"]),
        "margin_pred_max_abs_diff": _max_abs(a1_out["margin_pred"], a3h_out["margin_pred"]),
        "risk_logit_max_abs_diff": _max_abs(a1_out["risk_logit"], a3h_out["risk_logit"]),
        "utility_base_max_abs_diff": _max_abs(a1_out["utility_base"], a3h_out["utility_base"]),
        "gt_min_clearance_max_abs_diff": _max_abs(a1_out["gt_min_clearance"], a3h_out["gt_min_clearance"]),
        "a3h_utility_delta_mean_abs": _mean_abs(a3h_out["utility_delta"]),
        "a3h_utility_delta_max_abs": _max_abs(a3h_out["utility_delta"], torch.zeros_like(a3h_out["utility_delta"])),
        "selected_id_match_rate": _rate(a1_out["selected_id"] == a3h_out["selected_id"]),
        "a1_selected_gt_clearance_mean": float(a1_out["gt_min_clearance"].gather(1, a1_out["selected_id"][:, None]).mean().detach().cpu()),
        "a3h_selected_gt_clearance_mean": float(a3h_out["gt_min_clearance"].gather(1, a3h_out["selected_id"][:, None]).mean().detach().cpu()),
    }
    if label is not None and bool(label_valid.any()):
        valid_label = label[label_valid]
        result.update({
            "reaction_margin_label_valid_rate": _rate(label_valid),
            "reaction_margin_label_mean": float(valid_label.mean().detach().cpu()),
            "reaction_margin_label_negative_rate": _rate(valid_label < 0.0),
        })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


def parser():
    p = argparse.ArgumentParser(description="Check A1/A3h parity on a fixed OARM validation batch.")
    p.add_argument("--a1-checkpoint", required=True)
    p.add_argument("--a3h-checkpoint", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--mode", default="valid", choices=("train", "valid"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--batch-index", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="")
    p.add_argument("--output", required=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())

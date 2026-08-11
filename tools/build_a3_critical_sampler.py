import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch.utils.data import DataLoader

from OARM.config import oarm_cfg
from OARM.dataset import OARMDataset
from OARM.eval.eval_dataset import build_world_states, flatten_labels, maybe_generate_reaction_margin_labels
from OARM.loss import OARMLoss
from OARM.policy.oarm_network import OARMNetwork
from OARM.utils.checkpoint import load_oarm_checkpoint
from OARM.utils.gt_clearance import candidate_min_clearance_gt
from OARM.utils.yopo_dataset_context import resolve_dataset_dir, yopo_dataset_cfg
from OARM.utils.yopo_compat import ensure_yopo_path

ensure_yopo_path()
from config.config import cfg


def tensor_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def safe_mean(values: torch.Tensor):
    finite = torch.isfinite(values)
    if not bool(finite.any()):
        return None
    return float(values[finite].mean().detach().cpu())


def load_policy(args, device):
    if args.checkpoint:
        state_dict, metadata = load_oarm_checkpoint(args.checkpoint, map_location=device)
        scale = ((metadata or {}).get("training_options") or {}).get(
            "yopo_preserve_utility_delta_scale", oarm_cfg.yopo_preserve_utility_delta_scale
        )
        policy = OARMNetwork(
            candidate_mode=args.candidate_mode,
            backbone_mode="yopo_original",
            utility_delta_scale=float(scale),
        ).to(device)
        policy.load_state_dict(state_dict)
    elif args.yopo_checkpoint:
        policy = OARMNetwork(candidate_mode="yopo_preserve", backbone_mode="yopo_original").to(device)
        state_dict = torch.load(args.yopo_checkpoint, map_location=device, weights_only=True)
        policy.preserve_network.load_yopo_state_dict(state_dict, strict=True)
    else:
        raise ValueError("Pass either --checkpoint A1/OARM checkpoint or --yopo-checkpoint YOPO checkpoint")
    policy.eval()
    return policy


def to_device(batch, device):
    depth, pos, rot, obs_b, map_id, labels = batch
    labels = {k: v.to(device) if torch.is_tensor(v) else v for k, v in labels.items()}
    return depth.to(device), pos.to(device), rot.to(device), obs_b.to(device), map_id.to(device), labels


def metric_args():
    return argparse.Namespace(
        eval_occlusion_risk=True,
        eval_reaction_margin=True,
        use_weak_margin_label=False,
        eval_yield_feasibility=False,
        eval_backup_feasibility=False,
        eval_risk_point_guidance=True,
    )


def masked_min(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    fill = torch.full_like(values, float("inf"))
    return torch.where(mask, values, fill).amin(dim=1)


def masked_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    fill = torch.full_like(values, -float("inf"))
    return torch.where(mask, values, fill).amax(dim=1)


def batch_records(args, batch_start: int, batch, policy, loss_fn, dataset_root: str, device):
    depth, pos, rot, obs_b, map_id, labels = to_device(batch, device)
    traj_num = int(cfg["traj_num"])
    with torch.inference_mode():
        candidate = policy.inference(depth, obs_b)
        flat = candidate.flatten()
        start_state_w, end_state_w, goal_w = build_world_states(pos, rot, obs_b, flat)
        map_id_expanded = map_id.repeat_interleave(traj_num)
        eval_args = metric_args()
        flat_labels = flatten_labels(labels, flat, device, eval_args)
        flat_labels = maybe_generate_reaction_margin_labels(
            flat_labels,
            flat,
            start_state_w,
            end_state_w,
            map_id_expanded,
            goal_w,
            eval_args,
            loss_fn.margin_labeler,
            loss_fn.line_of_sight,
            loss_fn,
        )
        loss_dict = loss_fn(start_state_w, end_state_w, flat, goal_w, flat_labels, map_id_expanded)
        margin = flat_labels.get("reaction_margin")
        valid = flat_labels.get("reaction_margin_valid")
        if margin is None or valid is None:
            raise RuntimeError("Reaction-margin labels were not generated; check loss/dataset settings")
        margin = margin.reshape(-1, traj_num).float()
        valid = valid.reshape(-1, traj_num).bool() & torch.isfinite(margin)
        min_clearance = candidate_min_clearance_gt(
            loss_dict["sampled_pos_w"], map_id_expanded, dataset_root
        ).reshape(-1, traj_num)
        clearance_valid = torch.isfinite(min_clearance)
        geom_safe = clearance_valid & (min_clearance > args.safe_clearance_m)
        geom_unsafe = clearance_valid & (min_clearance < args.unsafe_clearance_m)
        progress_score = -OARMLoss.goal_progress_cost(start_state_w, end_state_w, goal_w, flat["traj_time"]).reshape(-1, traj_num)
        if args.oracle_min_progress > 0.0:
            progress_ok = torch.isfinite(progress_score) & (progress_score > args.oracle_min_progress)
        else:
            progress_ok = torch.isfinite(progress_score)
        geom_margin_valid = geom_safe & valid & progress_ok
        valid_count = valid.sum(dim=1)
        geom_valid_count = geom_margin_valid.sum(dim=1)
        progress_ok_count = progress_ok.sum(dim=1)
        positive_count = (geom_margin_valid & (margin > 0.0)).sum(dim=1)
        negative_count = (geom_margin_valid & (margin < 0.0)).sum(dim=1)
        margin_min = masked_min(margin, geom_margin_valid)
        margin_max = masked_max(margin, geom_margin_valid)
        spread = margin_max - margin_min
        spread = torch.where(torch.isfinite(spread), spread, torch.zeros_like(spread))
        mixed_sign = (positive_count > 0) & (negative_count > 0)
        oracle_available = geom_valid_count > 0
        high_spread = spread >= args.min_margin_spread
        valid_rich = geom_valid_count >= args.min_valid_candidates
        rank_informative = oracle_available & (mixed_sign | (high_spread & valid_rich))
        all_negative = oracle_available & (positive_count == 0) & (negative_count > 0)
        weights = torch.ones_like(spread, dtype=torch.float32) * args.normal_weight
        weights = weights + oracle_available.float() * args.oracle_bonus
        weights = weights + mixed_sign.float() * args.mixed_sign_bonus
        weights = weights + high_spread.float() * args.high_spread_bonus
        weights = weights + valid_rich.float() * args.valid_rich_bonus
        weights = weights + all_negative.float() * args.all_negative_bonus
        weights = weights.clamp(max=args.max_weight)
        records = []
        for i in range(depth.shape[0]):
            item = batch_start + i
            records.append(
                {
                    "dataset_index": item,
                    "map_id": int(map_id[i].detach().cpu()),
                    "valid_candidate_count": int(valid_count[i].detach().cpu()),
                    "geom_margin_valid_count": int(geom_valid_count[i].detach().cpu()),
                    "positive_count": int(positive_count[i].detach().cpu()),
                    "negative_count": int(negative_count[i].detach().cpu()),
                    "geom_safe_count": int(geom_safe[i].sum().detach().cpu()),
                    "progress_ok_count": int(progress_ok_count[i].detach().cpu()),
                    "geom_unsafe_count": int(geom_unsafe[i].sum().detach().cpu()),
                    "margin_min": None if not bool(oracle_available[i]) else float(margin_min[i].detach().cpu()),
                    "margin_max": None if not bool(oracle_available[i]) else float(margin_max[i].detach().cpu()),
                    "margin_spread": float(spread[i].detach().cpu()),
                    "oracle_available": bool(oracle_available[i].detach().cpu()),
                    "mixed_sign": bool(mixed_sign[i].detach().cpu()),
                    "high_spread": bool(high_spread[i].detach().cpu()),
                    "valid_rich": bool(valid_rich[i].detach().cpu()),
                    "rank_informative": bool(rank_informative[i].detach().cpu()),
                    "all_negative": bool(all_negative[i].detach().cpu()),
                    "sample_weight": float(weights[i].detach().cpu()),
                    "raw_gt_risk_point_valid_rate": tensor_float(labels["raw_gt_risk_point_valid_rate"][i]),
                    "raw_gt_risk_point_weight_sum": tensor_float(labels["raw_gt_risk_point_weight_sum"][i]),
                }
            )
    return records, weights.detach().cpu()


def summarize(records: List[Dict]) -> Dict:
    n = max(len(records), 1)
    def rate(key):
        return sum(1 for r in records if r.get(key)) / n
    def mean(key):
        vals = [r[key] for r in records if r.get(key) is not None and math.isfinite(float(r[key]))]
        return sum(float(v) for v in vals) / max(len(vals), 1)
    weight_sum = sum(float(r.get("sample_weight", 0.0)) for r in records)
    weight_sum = max(weight_sum, 1e-9)
    def weighted_rate(key):
        return sum(float(r.get("sample_weight", 0.0)) for r in records if r.get(key)) / weight_sum
    def weighted_mean(key):
        vals = [(float(r.get("sample_weight", 0.0)), r.get(key)) for r in records]
        vals = [(w, float(v)) for w, v in vals if v is not None and math.isfinite(float(v)) and w > 0.0]
        denom = sum(w for w, _v in vals)
        return sum(w * v for w, v in vals) / max(denom, 1e-9)
    hist = {}
    weighted_hist = {}
    for r in records:
        c = int(r["geom_margin_valid_count"])
        key = str(c)
        hist[key] = hist.get(key, 0) + 1
        weighted_hist[key] = weighted_hist.get(key, 0.0) + float(r.get("sample_weight", 0.0)) / weight_sum
    return {
        "sample_count": len(records),
        "oracle_available_rate": rate("oracle_available"),
        "rank_informative_rate": rate("rank_informative"),
        "mixed_sign_rate": rate("mixed_sign"),
        "high_spread_rate": rate("high_spread"),
        "valid_rich_rate": rate("valid_rich"),
        "all_negative_rate": rate("all_negative"),
        "weighted_oracle_available_rate": weighted_rate("oracle_available"),
        "weighted_rank_informative_rate": weighted_rate("rank_informative"),
        "weighted_mixed_sign_rate": weighted_rate("mixed_sign"),
        "weighted_high_spread_rate": weighted_rate("high_spread"),
        "weighted_valid_rich_rate": weighted_rate("valid_rich"),
        "weighted_all_negative_rate": weighted_rate("all_negative"),
        "valid_candidate_count_mean": mean("valid_candidate_count"),
        "geom_margin_valid_count_mean": mean("geom_margin_valid_count"),
        "weighted_geom_margin_valid_count_mean": weighted_mean("geom_margin_valid_count"),
        "positive_count_mean": mean("positive_count"),
        "negative_count_mean": mean("negative_count"),
        "margin_spread_mean": mean("margin_spread"),
        "weighted_margin_spread_mean": weighted_mean("margin_spread"),
        "sample_weight_mean": mean("sample_weight"),
        "sample_weight_max": max((float(r["sample_weight"]) for r in records), default=0.0),
        "geom_margin_valid_count_hist": hist,
        "weighted_geom_margin_valid_count_hist": weighted_hist,
    }


def dataset_identity(dataset_root: str, mode: str, sample_count: int) -> Dict:
    root = Path(dataset_root).resolve()
    pose_files = sorted(root.glob("pose-*.csv"), key=lambda item: item.name)
    pointcloud_files = sorted(root.glob("pointcloud-*.ply"), key=lambda item: item.name)
    payload = {
        "dataset_root": str(root),
        "mode": mode,
        "sample_count": int(sample_count),
        "pose_file_count": len(pose_files),
        "pointcloud_file_count": len(pointcloud_files),
        "pose_files": [item.name for item in pose_files[:8]],
        "pointcloud_files": [item.name for item in pointcloud_files[:8]],
    }
    return payload


def write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def write_csv(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    keys = list(records[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def main():
    args = parser().parse_args()
    dataset_root = resolve_dataset_dir(args.dataset_root)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    records: List[Dict] = []
    weight_chunks: List[torch.Tensor] = []
    with yopo_dataset_cfg(dataset_root):
        dataset = OARMDataset(
            mode=args.mode,
            dataset_root=dataset_root,
            use_privileged_risk_filter=True,
            risk_label_source="gt_pointcloud",
            gt_sampler_options={
                "point_count": args.gt_risk_point_count,
                "max_forward_m": args.gt_max_forward_m,
                "horizon_fov_expand_deg": args.gt_horizon_fov_expand_deg,
                "vertical_fov_expand_deg": args.gt_vertical_fov_expand_deg,
                "reachable_score_weight": args.gt_reachable_score_weight,
            },
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
        policy = load_policy(args, device)
        loss_fn = OARMLoss(
            enable_occlusion_risk=True,
            enable_risk_point_guidance=True,
            enable_reaction_margin=True,
            enable_margin_ranking=True,
            use_esdf_collision=args.use_esdf_collision,
            use_occlusion_aware_visibility=True,
            deployed_yaw_mode=args.deployed_yaw_mode,
            risk_assoc_distance_m=args.risk_assoc_distance_m,
            risk_assoc_sigma_m=args.risk_assoc_sigma_m,
            risk_arrival_radius_m=args.risk_arrival_radius_m,
        ).to(device)
        seen = 0
        for step, batch in enumerate(loader):
            if args.max_batches is not None and step >= args.max_batches:
                break
            batch_records_out, batch_weights = batch_records(args, seen, batch, policy, loss_fn, dataset_root, device)
            records.extend(batch_records_out)
            weight_chunks.append(batch_weights)
            seen += len(batch_records_out)
            if args.print_every and (step + 1) % args.print_every == 0:
                print(f"scanned_batches={step + 1} samples={seen}")
    weights = torch.cat(weight_chunks, dim=0) if weight_chunks else torch.empty(0, dtype=torch.float32)
    summary = summarize(records)
    identity = dataset_identity(dataset_root, args.mode, len(records))
    payload = {
        "dataset_identity": identity,
        "dataset_root": str(dataset_root),
        "mode": args.mode,
        "candidate_mode": args.candidate_mode,
        "checkpoint": args.checkpoint,
        "yopo_checkpoint": args.yopo_checkpoint,
        "thresholds": {
            "unsafe_clearance_m": args.unsafe_clearance_m,
            "safe_clearance_m": args.safe_clearance_m,
            "min_valid_candidates": args.min_valid_candidates,
            "min_margin_spread": args.min_margin_spread,
            "oracle_min_progress": args.oracle_min_progress,
        },
        "weighting": {
            "normal_weight": args.normal_weight,
            "oracle_bonus": args.oracle_bonus,
            "mixed_sign_bonus": args.mixed_sign_bonus,
            "high_spread_bonus": args.high_spread_bonus,
            "valid_rich_bonus": args.valid_rich_bonus,
            "all_negative_bonus": args.all_negative_bonus,
            "max_weight": args.max_weight,
        },
        "summary": summary,
    }
    if args.output_weights:
        out = Path(args.output_weights)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"weights": weights, "records": records, "summary": payload}, out)
    if args.output_jsonl:
        write_jsonl(Path(args.output_jsonl), records)
    if args.output_csv:
        write_csv(Path(args.output_csv), records)
    if args.output_summary:
        out = Path(args.output_summary)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def parser():
    p = argparse.ArgumentParser(description="Build A3 critical-frame sampling weights from candidate-level reaction-margin density.")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--mode", default="train", choices=("train", "valid"))
    p.add_argument("--checkpoint", default="", help="A1/OARM checkpoint used to generate YOPO-preserve candidates")
    p.add_argument("--yopo-checkpoint", default="", help="Official YOPO checkpoint alternative")
    p.add_argument("--candidate-mode", default="yopo_preserve", choices=("yopo_preserve", "yopo_preserve_rerank"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--device", default="")
    p.add_argument("--deployed-yaw-mode", default="goal", choices=("goal", "hold", "predicted"))
    p.add_argument("--use-esdf-collision", action="store_true")
    p.add_argument("--gt-risk-point-count", type=int, default=oarm_cfg.gt_risk_point_count)
    p.add_argument("--gt-max-forward-m", type=float, default=oarm_cfg.gt_max_forward_m)
    p.add_argument("--gt-horizon-fov-expand-deg", type=float, default=oarm_cfg.gt_horizon_fov_expand_deg)
    p.add_argument("--gt-vertical-fov-expand-deg", type=float, default=oarm_cfg.gt_vertical_fov_expand_deg)
    p.add_argument("--gt-reachable-score-weight", type=float, default=oarm_cfg.gt_reachable_score_weight)
    p.add_argument("--risk-assoc-distance-m", type=float, default=oarm_cfg.risk_assoc_distance_m)
    p.add_argument("--risk-assoc-sigma-m", type=float, default=oarm_cfg.risk_assoc_sigma_m)
    p.add_argument("--risk-arrival-radius-m", type=float, default=oarm_cfg.risk_arrival_radius_m)
    p.add_argument("--unsafe-clearance-m", type=float, default=0.25)
    p.add_argument("--safe-clearance-m", type=float, default=0.35)
    p.add_argument("--min-valid-candidates", type=int, default=5)
    p.add_argument("--min-margin-spread", type=float, default=0.20)
    p.add_argument("--oracle-min-progress", type=float, default=oarm_cfg.yopo_preserve_oracle_min_progress)
    p.add_argument("--normal-weight", type=float, default=1.0)
    p.add_argument("--oracle-bonus", type=float, default=3.0)
    p.add_argument("--mixed-sign-bonus", type=float, default=6.0)
    p.add_argument("--high-spread-bonus", type=float, default=4.0)
    p.add_argument("--valid-rich-bonus", type=float, default=2.0)
    p.add_argument("--all-negative-bonus", type=float, default=2.0)
    p.add_argument("--max-weight", type=float, default=12.0)
    p.add_argument("--output-weights", default="")
    p.add_argument("--output-jsonl", default="")
    p.add_argument("--output-csv", default="")
    p.add_argument("--output-summary", default="")
    p.add_argument("--print-every", type=int, default=50)
    return p


if __name__ == "__main__":
    main()

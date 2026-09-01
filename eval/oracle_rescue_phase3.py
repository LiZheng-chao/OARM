import argparse
import json
import math
import os
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from OARM.dataset import OARMDataset
from OARM.eval.eval_dataset import (
    apply_checkpoint_eval_protocol,
    apply_eval_stage,
    build_world_states,
    flatten_labels,
    gt_sampler_options_from_args,
    maybe_generate_reaction_margin_labels,
    resolve_utility_delta_scale,
)
from OARM.loss import OARMLoss
from OARM.policy.oarm_candidate_generator import OARMCandidateGenerator
from OARM.policy.oarm_network import OARMNetwork
from OARM.utils.checkpoint import load_oarm_checkpoint, validate_checkpoint_metadata
from OARM.utils.yopo_dataset_context import yopo_dataset_cfg
from OARM.visibility.reaction_margin_labeler import ReactionMarginLabeler


def _safe_div(num, den):
    return float(num) / float(den) if den else None


def _append_value(acc, key, value):
    if torch.is_tensor(value):
        value = value.detach().reshape(-1).cpu()
        for item in value:
            item = float(item)
            if math.isfinite(item):
                acc[key].append(item)
        return
    value = float(value)
    if math.isfinite(value):
        acc[key].append(value)


def _quantiles(values):
    if not values:
        return {}
    tensor = torch.tensor(values, dtype=torch.float32)
    qs = torch.quantile(tensor, torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95]))
    return {
        "p05": float(qs[0]),
        "p25": float(qs[1]),
        "p50": float(qs[2]),
        "p75": float(qs[3]),
        "p95": float(qs[4]),
    }


def load_policy(args, device):
    state_dict = None
    metadata = {}
    if args.checkpoint:
        state_dict, metadata = load_oarm_checkpoint(args.checkpoint, map_location=device)
    args.yopo_preserve_utility_delta_scale = resolve_utility_delta_scale(args, metadata)
    apply_checkpoint_eval_protocol(args, metadata)

    policy = OARMNetwork(
        candidate_mode=args.candidate_mode,
        backbone_mode=args.backbone_mode,
        enable_yield_candidates=args.enable_yield_candidates,
        utility_delta_scale=args.yopo_preserve_utility_delta_scale,
        enable_rm_critic=args.eval_probabilistic_rm_critic,
        rm_critic_hazard_bins=args.rm_critic_hazard_bins,
        rm_critic_hazard_max_time_s=args.rm_critic_hazard_max_time_s,
    ).to(device)
    if state_dict is None:
        policy.eval()
        return policy, metadata

    if args.candidate_mode in {"yopo_preserve", "yopo_preserve_rerank", "a4_preserve_brake"}:
        is_oarm_preserve = any(key.startswith("preserve_network.") for key in state_dict)
        if is_oarm_preserve:
            validate_checkpoint_metadata(
                metadata,
                args.candidate_mode,
                args.backbone_mode,
                allow_mismatch=args.allow_checkpoint_mismatch,
                enable_yield_candidates=args.enable_yield_candidates,
                deployed_yaw_mode=args.deployed_yaw_mode,
                risk_label_source=args.risk_label_source,
                yopo_preserve_utility_delta_scale=args.yopo_preserve_utility_delta_scale,
            )
            policy.load_state_dict(state_dict)
        else:
            policy.preserve_network.load_yopo_state_dict(state_dict, strict=True)
    else:
        validate_checkpoint_metadata(
            metadata,
            args.candidate_mode,
            args.backbone_mode,
            allow_mismatch=args.allow_checkpoint_mismatch,
            enable_yield_candidates=args.enable_yield_candidates,
            deployed_yaw_mode=args.deployed_yaw_mode,
            risk_label_source=args.risk_label_source,
            yopo_preserve_utility_delta_scale=args.yopo_preserve_utility_delta_scale,
        )
        policy.load_state_dict(state_dict)
    policy.eval()
    return policy, metadata


def evaluate(args):
    apply_eval_stage(args)
    args.eval_reaction_margin = True
    args.eval_probabilistic_rm_critic = True
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    tau_s = [float(value) / 1000.0 for value in args.tau_ms]

    policy, metadata = load_policy(args, device)
    gt_sampler_options = gt_sampler_options_from_args(args)
    dataset = OARMDataset(
        mode=args.mode,
        dataset_root=args.dataset_root or None,
        use_privileged_risk_filter=args.use_privileged_risk_filter,
        risk_label_source=args.risk_label_source,
        gt_sampler_options=gt_sampler_options,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    with yopo_dataset_cfg(args.dataset_root or None):
        loss_fn = OARMLoss(
            eval_points=args.eval_points,
            use_esdf_collision=args.use_esdf_collision,
            use_occlusion_aware_visibility=args.use_occlusion_aware_visibility,
            enable_reaction_margin=True,
            enable_probabilistic_rm_critic=True,
            rm_critic_hazard_bins=args.rm_critic_hazard_bins,
            rm_critic_hazard_max_time_s=args.rm_critic_hazard_max_time_s,
            rm_critic_zero_bce_weight=args.rm_critic_zero_bce_weight,
            rm_critic_hazard_bce_weight=args.rm_critic_hazard_bce_weight,
            deployed_yaw_mode=args.deployed_yaw_mode,
            risk_assoc_distance_m=args.risk_assoc_distance_m,
            risk_assoc_sigma_m=args.risk_assoc_sigma_m,
            risk_arrival_radius_m=args.risk_arrival_radius_m,
        )
        line_of_sight = loss_fn.line_of_sight if args.use_occlusion_aware_visibility else None

    labeler = ReactionMarginLabeler(risk_arrival_radius_m=args.risk_arrival_radius_m)
    counters = {tau: defaultdict(int) for tau in tau_s}
    value_acc = defaultdict(list)
    seen_batches = 0
    seen_samples = 0
    hazard_bin_counts = [0 for _ in range(int(args.rm_critic_hazard_bins or 0))]

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
            traj_num = int(flat["traj_time"].numel() // max(depth.shape[0], 1))
            map_id_expanded = map_id.repeat_interleave(traj_num, dim=0)
            flat_labels = maybe_generate_reaction_margin_labels(
                flat_labels,
                flat,
                start_state_w,
                end_state_w,
                map_id_expanded,
                goal_w,
                args,
                labeler,
                line_of_sight,
                loss_fn,
            )
            loss_fn(start_state_w, end_state_w, flat, goal_w, flat_labels, map_id_expanded)

            batch_size = depth.shape[0]
            utility = flat["utility_score"].reshape(batch_size, traj_num)
            selected_id = utility.argmax(dim=1)
            batch_range = torch.arange(batch_size, device=device)
            reaction_window = flat_labels["reaction_window"].reshape(batch_size, traj_num)
            interaction_valid = flat_labels.get("rm_interaction_valid", flat_labels.get("reaction_margin_valid"))
            if interaction_valid is None:
                interaction_valid = torch.ones_like(reaction_window, dtype=torch.bool)
            else:
                interaction_valid = interaction_valid.reshape(batch_size, traj_num).bool()
            no_entry = flat_labels.get("rm_no_entry")
            if no_entry is None:
                no_entry = torch.zeros_like(interaction_valid, dtype=torch.bool)
            else:
                no_entry = no_entry.reshape(batch_size, traj_num).bool()
            timely_visible = flat_labels.get("rm_timely_visible")
            if timely_visible is None:
                timely_visible = torch.ones_like(interaction_valid, dtype=torch.bool)
            else:
                timely_visible = timely_visible.reshape(batch_size, traj_num).bool()
            blind_at_entry = flat_labels.get("rm_blind_at_entry", flat_labels.get("rm_right_censored"))
            if blind_at_entry is None:
                blind_at_entry = ~timely_visible
            else:
                blind_at_entry = blind_at_entry.reshape(batch_size, traj_num).bool()
            progress_mask = torch.ones_like(interaction_valid, dtype=torch.bool)
            candidate_type = flat.get("candidate_type")
            if candidate_type is not None:
                progress_mask = candidate_type.reshape(batch_size, traj_num) == OARMCandidateGenerator.PROGRESS
            finite_window = torch.isfinite(reaction_window)
            usable = interaction_valid & (~no_entry) & progress_mask & finite_window
            zero_window = usable & (blind_at_entry | (timely_visible & (reaction_window <= 1e-6)))
            progress = -OARMLoss.goal_progress_cost(start_state_w, end_state_w, goal_w, flat["traj_time"]).detach()
            progress = progress.reshape(batch_size, traj_num)

            selected_window = reaction_window[batch_range, selected_id]
            selected_usable = usable[batch_range, selected_id]
            selected_progress = progress[batch_range, selected_id]
            _append_value(value_acc, "selected_reaction_window_s", selected_window[selected_usable])
            _append_value(value_acc, "selected_progress", selected_progress[torch.isfinite(selected_progress)])
            _append_value(value_acc, "candidate_reaction_window_s", reaction_window[usable])
            _append_value(value_acc, "candidate_progress", progress[torch.isfinite(progress)])

            positive_event = usable & timely_visible & (~zero_window) & (reaction_window > 1e-6)
            if hazard_bin_counts and bool(positive_event.any()):
                horizon = max(float(args.rm_critic_hazard_max_time_s), 1e-3)
                k = len(hazard_bin_counts)
                idx = torch.floor(reaction_window[positive_event].clamp(min=0.0, max=horizon - 1e-6) / horizon * k).long()
                counts = torch.bincount(idx.clamp(min=0, max=k - 1), minlength=k).detach().cpu().tolist()
                hazard_bin_counts = [old + int(new) for old, new in zip(hazard_bin_counts, counts)]

            seen_batches += 1
            seen_samples += batch_size
            for tau in tau_s:
                ctr = counters[tau]
                ctr["samples"] += batch_size
                ctr["candidates"] += batch_size * traj_num
                ctr["usable_candidates"] += int(usable.sum().item())
                ctr["positive_window_candidates"] += int((usable & (reaction_window > 0.0)).sum().item())
                ctr["no_entry_candidates"] += int(no_entry.sum().item())
                ctr["zero_window_candidates"] += int((zero_window & interaction_valid).sum().item())

                selected_violation = selected_usable & (selected_window < tau)
                ctr["selected_usable"] += int(selected_usable.sum().item())
                ctr["selected_violation"] += int(selected_violation.sum().item())

                safe_alt = usable & (reaction_window >= tau)
                selected_mask = torch.zeros_like(safe_alt)
                selected_mask[batch_range, selected_id] = True
                safe_alt = safe_alt & (~selected_mask)
                if args.progress_rho > 0.0:
                    progress_threshold = args.progress_rho * selected_progress[:, None]
                    safe_alt = safe_alt & torch.isfinite(progress) & torch.isfinite(progress_threshold) & (progress >= progress_threshold)
                safe_alt_available = safe_alt.any(dim=1)
                rescuable = selected_violation & safe_alt_available
                ctr["safe_alt_available"] += int(safe_alt_available.sum().item())
                ctr["rescuable_violation"] += int(rescuable.sum().item())

                oracle_window = reaction_window.masked_fill(~safe_alt, -torch.inf).max(dim=1).values
                oracle_valid = torch.isfinite(oracle_window)
                _append_value(value_acc, f"tau_{int(round(tau * 1000))}_oracle_window_s", oracle_window[oracle_valid])

    per_tau = []
    for tau in tau_s:
        ctr = counters[tau]
        samples = ctr["samples"]
        candidates = ctr["candidates"]
        selected_violation = ctr["selected_violation"]
        per_tau.append(
            {
                "tau_ms": int(round(tau * 1000)),
                "samples": samples,
                "candidates": candidates,
                "interaction_valid_candidate_rate": _safe_div(ctr["usable_candidates"], candidates),
                "positive_window_candidate_rate": _safe_div(ctr["positive_window_candidates"], candidates),
                "no_entry_candidate_rate": _safe_div(ctr["no_entry_candidates"], candidates),
                "zero_window_candidate_rate": _safe_div(ctr["zero_window_candidates"], candidates),
                "selected_interaction_valid_rate": _safe_div(ctr["selected_usable"], samples),
                "R_sel_selected_violation_rate": _safe_div(selected_violation, samples),
                "R_sel_given_selected_valid": _safe_div(selected_violation, ctr["selected_usable"]),
                "safe_alt_available_rate": _safe_div(ctr["safe_alt_available"], samples),
                "R_rescue_rescuable_violation_rate": _safe_div(ctr["rescuable_violation"], samples),
                "R_rescue_given_violation": _safe_div(ctr["rescuable_violation"], selected_violation),
            }
        )

    output = {
        "checkpoint": args.checkpoint,
        "checkpoint_training_options": (metadata or {}).get("training_options", {}),
        "stage": args.stage,
        "candidate_mode": args.candidate_mode,
        "backbone_mode": args.backbone_mode,
        "mode": args.mode,
        "dataset_root": args.dataset_root,
        "risk_label_source": args.risk_label_source,
        "use_occlusion_aware_visibility": bool(args.use_occlusion_aware_visibility),
        "use_privileged_risk_filter": bool(args.use_privileged_risk_filter),
        "rm_critic_hazard_bins": args.rm_critic_hazard_bins,
        "rm_critic_hazard_max_time_s": args.rm_critic_hazard_max_time_s,
        "progress_rho": args.progress_rho,
        "eval_points": args.eval_points,
        "batches": seen_batches,
        "samples": seen_samples,
        "per_tau": per_tau,
        "hazard_bin_counts": hazard_bin_counts,
        "hazard_bin_edges_s": [
            float(i) * float(args.rm_critic_hazard_max_time_s) / max(int(args.rm_critic_hazard_bins or 0), 1)
            for i in range(int(args.rm_critic_hazard_bins or 0) + 1)
        ],
        "quantiles": {key: _quantiles(values) for key, values in value_acc.items()},
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, sort_keys=True)
        print(f"Wrote Phase 3 oracle rescue stats: {args.output}")


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["v0", "v1_occ", "v2_margin", "v3_yield", "a3h", "oarm3_s2_prob_rm", "a4a", "full"], default="oarm3_s2_prob_rm")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--allow-checkpoint-mismatch", action="store_true")
    p.add_argument("--candidate-mode", choices=["yopo", "typed_frontier", "yopo_preserve", "yopo_preserve_rerank", "a4_preserve_brake"], default="")
    p.add_argument("--backbone-mode", choices=["oarm_light", "yopo_original"], default="")
    p.add_argument("--enable-yield-candidates", action="store_true")
    p.add_argument("--deployed-yaw-mode", choices=["goal", "hold", "predicted"], default="")
    p.add_argument("--risk-label-source", choices=["proxy", "proxy_esdf", "gt_pointcloud"], default="")
    p.add_argument("--yopo-preserve-utility-delta-scale", type=float, default=None)
    p.add_argument("--yopo-preserve-safety-cost-threshold", type=float, default=None)
    p.add_argument("--yopo-preserve-safe-cost-threshold", type=float, default=None)
    p.add_argument("--yopo-preserve-geometry-oracle-source", choices=["", "esdf_cost", "gt_clearance"], default="")
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
    p.add_argument("--rm-critic-hazard-bins", type=int, default=None)
    p.add_argument("--rm-critic-hazard-max-time-s", type=float, default=None)
    p.add_argument("--rm-critic-zero-bce-weight", type=float, default=None)
    p.add_argument("--rm-critic-hazard-bce-weight", type=float, default=None)
    p.add_argument("--eval-points", type=int, default=60)
    p.add_argument("--mode", choices=["train", "valid"], default="valid")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--dataset-root", type=str, default="")
    p.add_argument("--eval-occlusion-risk", action="store_true")
    p.add_argument("--eval-reaction-margin", action="store_true")
    p.add_argument("--eval-margin-ranking", action="store_true")
    p.add_argument("--eval-probabilistic-rm-critic", action="store_true")
    p.add_argument("--eval-risk-point-guidance", action="store_true")
    p.add_argument("--eval-yaw-visibility", action="store_true")
    p.add_argument("--use-weak-margin-label", action="store_true")
    p.add_argument("--eval-backup-feasibility", action="store_true")
    p.add_argument("--eval-yield-feasibility", action="store_true")
    p.add_argument("--use-esdf-collision", action="store_true")
    p.add_argument("--use-occlusion-aware-visibility", action="store_true")
    p.add_argument("--use-privileged-risk-filter", action="store_true")
    p.add_argument("--tau-ms", type=float, nargs="+", default=[300.0, 500.0, 700.0, 900.0, 1100.0])
    p.add_argument("--progress-rho", type=float, default=0.7)
    p.add_argument("--output", type=str, default="")
    return p


if __name__ == "__main__":
    evaluate(parser().parse_args())




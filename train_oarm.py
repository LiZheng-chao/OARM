import argparse
import os
import random
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
YOPO_DIR = os.path.join(REPO_ROOT, "YOPO")
if YOPO_DIR not in sys.path:
    sys.path.insert(0, YOPO_DIR)

from OARM.policy.oarm_trainer import OARMTrainer
from OARM.config import get_oarm_training_preset


TRAINING_OPTION_KEYS = (
    "candidate_mode",
    "backbone_mode",
    "enable_yield_candidates",
    "train_occlusion_risk",
    "train_risk_point_guidance",
    "train_reaction_margin",
    "train_margin_ranking",
    "train_yaw_visibility",
    "deployed_yaw_mode",
    "risk_label_source",
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
    "use_weak_margin_label",
    "train_backup_feasibility",
    "train_yield_feasibility",
    "use_esdf_collision",
    "use_occlusion_aware_visibility",
    "use_privileged_risk_filter",
)


def configure_random_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage",
        choices=["v0", "v1_occ", "v2_margin", "v3_yield", "full"],
        default="v0",
        help="named ablation preset; CLI flags and --config override it",
    )
    p.add_argument(
        "--config",
        type=str,
        default="",
        help="optional simple key: value config file, e.g. OARM/configs/oarm_v2_margin.yaml",
    )
    p.add_argument("--epoch", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--yopo-checkpoint", type=str, default="", help="official YOPO checkpoint used to initialize candidate_mode=yopo_preserve")
    p.add_argument("--allow-checkpoint-mismatch", action="store_true")
    p.add_argument("--candidate-mode", choices=["yopo", "typed_frontier", "yopo_preserve", "yopo_preserve_rerank"], default="")
    p.add_argument("--backbone-mode", choices=["oarm_light", "yopo_original"], default="")
    p.add_argument("--enable-yield-candidates", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    p.add_argument("--train-risk-point-guidance", action="store_true")
    p.add_argument("--train-occlusion-risk", action="store_true")
    p.add_argument("--train-reaction-margin", action="store_true")
    p.add_argument("--train-margin-ranking", action="store_true")
    p.add_argument("--train-yaw-visibility", action="store_true")
    p.add_argument("--deployed-yaw-mode", choices=["goal", "hold", "predicted"], default="")
    p.add_argument("--risk-label-source", choices=["proxy", "proxy_esdf", "gt_pointcloud"], default="")
    p.add_argument("--gt-risk-point-count", type=int, default=None)
    p.add_argument("--gt-hidden-depth-margin-m", type=float, default=None)
    p.add_argument("--gt-min-forward-m", type=float, default=None)
    p.add_argument("--gt-max-forward-m", type=float, default=None)
    p.add_argument("--gt-horizon-fov-expand-deg", type=float, default=None)
    p.add_argument("--gt-vertical-fov-expand-deg", type=float, default=None)
    p.add_argument("--gt-depth-metric", choices=["forward", "ray"], default="")
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
    p.add_argument("--use-weak-margin-label", action="store_true")
    p.add_argument("--train-backup-feasibility", action="store_true")
    p.add_argument("--train-yield-feasibility", action="store_true")
    p.add_argument("--use-esdf-collision", action="store_true")
    p.add_argument("--use-occlusion-aware-visibility", action="store_true")
    p.add_argument("--use-privileged-risk-filter", action="store_true")
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--log-dir", type=str, default="")
    p.add_argument("--dataset-root", type=str, default="")
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--use-fused-adamw", action="store_true")
    p.add_argument("--train-yield-head-only", action="store_true")
    return p


def load_simple_config(path):
    if not path:
        return {}
    values = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ":" not in stripped:
                raise ValueError(f"Invalid config line {line_no} in {path}: {line.rstrip()}")
            key, value = stripped.split(":", 1)
            key = key.strip().replace("-", "_")
            value = value.split("#", 1)[0].strip()
            lowered = value.lower()
            if lowered in {"true", "yes", "1"}:
                values[key] = True
            elif lowered in {"false", "no", "0"}:
                values[key] = False
            else:
                try:
                    values[key] = int(value)
                except ValueError:
                    try:
                        values[key] = float(value)
                    except ValueError:
                        values[key] = value.strip("'\"")
    unknown = sorted(set(values) - set(TRAINING_OPTION_KEYS))
    if unknown:
        raise ValueError(f"Unknown OARM training config keys in {path}: {', '.join(unknown)}")
    return values


def resolve_training_options(args):
    preset = get_oarm_training_preset(args.stage)
    options = {key: getattr(preset, key) for key in TRAINING_OPTION_KEYS}
    options.update(load_simple_config(args.config))

    flag_map = {
        "enable_yield_candidates": args.enable_yield_candidates,
        "train_occlusion_risk": args.train_occlusion_risk,
        "train_risk_point_guidance": args.train_risk_point_guidance,
        "train_reaction_margin": args.train_reaction_margin,
        "train_margin_ranking": args.train_margin_ranking,
        "train_yaw_visibility": args.train_yaw_visibility,
        "use_weak_margin_label": args.use_weak_margin_label,
        "train_backup_feasibility": args.train_backup_feasibility,
        "train_yield_feasibility": args.train_yield_feasibility,
        "use_esdf_collision": args.use_esdf_collision,
        "use_occlusion_aware_visibility": args.use_occlusion_aware_visibility,
        "use_privileged_risk_filter": args.use_privileged_risk_filter,
    }
    for key, enabled in flag_map.items():
        if enabled:
            options[key] = True
    if args.candidate_mode:
        options["candidate_mode"] = args.candidate_mode
    if args.backbone_mode:
        options["backbone_mode"] = args.backbone_mode
    if args.deployed_yaw_mode:
        options["deployed_yaw_mode"] = args.deployed_yaw_mode
    if args.risk_label_source:
        options["risk_label_source"] = args.risk_label_source
    for key in (
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
    ):
        value = getattr(args, key)
        if value is not None and value != "":
            options[key] = value
    if options["train_backup_feasibility"]:
        options["train_yield_feasibility"] = True
        options["enable_yield_candidates"] = True
    if options["train_yield_feasibility"]:
        options["train_backup_feasibility"] = True
        options["enable_yield_candidates"] = True
    if options["train_margin_ranking"] and not options["train_reaction_margin"]:
        raise ValueError("train_margin_ranking requires train_reaction_margin=True")
    if options["use_occlusion_aware_visibility"] and not options["train_risk_point_guidance"]:
        raise ValueError("use_occlusion_aware_visibility requires train_risk_point_guidance=True")
    if (
        options["train_reaction_margin"]
        and options["risk_label_source"] == "gt_pointcloud"
        and not options["use_occlusion_aware_visibility"]
    ):
        raise ValueError("GT reaction-margin training requires occlusion-aware visibility")
    return options


if __name__ == "__main__":
    args = parser().parse_args()
    configure_random_seed(args.seed)
    training_options = resolve_training_options(args)
    print(f"OARM training stage: {args.stage}")
    for key in TRAINING_OPTION_KEYS:
        print(f"  {key}: {training_options[key]}")
    trainer = OARMTrainer(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        tensorboard_path=args.log_dir or None,
        checkpoint_path=args.checkpoint,
        yopo_checkpoint_path=args.yopo_checkpoint,
        save_on_exit=True,
        num_workers=args.num_workers,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        dataset_root=args.dataset_root or None,
        **training_options,
        experiment_options={
            "stage": args.stage,
            "config": args.config,
            "source_yopo_checkpoint": args.yopo_checkpoint,
            "seed": args.seed,
            "argv": sys.argv,
            "epoch": args.epoch,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "checkpoint": args.checkpoint,
            "candidate_mode": training_options["candidate_mode"],
            "backbone_mode": training_options["backbone_mode"],
            "num_workers": args.num_workers,
            "max_train_batches": args.max_train_batches,
            "max_val_batches": args.max_val_batches,
            "dataset_root": args.dataset_root,
            "grad_clip_norm": args.grad_clip_norm,
            "use_fused_adamw": args.use_fused_adamw,
            "train_yield_head_only": args.train_yield_head_only,
            **training_options,
        },
        config_path=args.config,
        log_interval=args.log_interval,
        allow_checkpoint_mismatch=args.allow_checkpoint_mismatch,
        grad_clip_norm=args.grad_clip_norm,
        use_fused_adamw=args.use_fused_adamw,
        train_yield_head_only=args.train_yield_head_only,
    )
    trainer.train(epoch=args.epoch)

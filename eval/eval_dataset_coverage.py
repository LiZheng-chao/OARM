import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np
import torch

from OARM.utils.occlusion import DepthFrontierExtractor


FRONTIER_EXTRACTOR = DepthFrontierExtractor()


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(finite) / len(finite) if finite else None


def parse_float(value) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def image_paths(dataset_root: Path, max_images_per_map: int) -> List[Path]:
    paths: List[Path] = []
    map_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name))
    for map_dir in map_dirs:
        imgs = sorted(map_dir.glob("*.png"), key=lambda p: int(p.stem.split("_")[-1]))
        paths.extend(imgs[:max_images_per_map])
    return paths


def depth_stats(path: Path, max_depth_m: float) -> Dict[str, float]:
    image = cv2.imread(str(path), -1)
    if image is None:
        raise FileNotFoundError(path)
    depth_norm = image.astype(np.float32) / 65535.0
    depth = depth_norm * max_depth_m
    valid = depth > 1e-3
    if not np.any(valid):
        return {
            "valid_depth_rate": 0.0,
            "near_depth_rate": 0.0,
            "far_or_saturated_rate": 1.0,
            "frontier_pixel_rate": 0.0,
            "depth_mean": 0.0,
        }
    depth_tensor = torch.from_numpy(depth_norm).float().view(1, 1, *depth_norm.shape)
    frontier = float(FRONTIER_EXTRACTOR(depth_tensor).mean().item())
    return {
        "valid_depth_rate": float(valid.mean()),
        "near_depth_rate": float((depth[valid] < 3.0).mean()),
        "far_or_saturated_rate": float((depth >= 0.98 * max_depth_m).mean()),
        "frontier_pixel_rate": frontier,
        "depth_mean": float(depth[valid].mean()),
    }


def summarize_raw_dataset(dataset_root: Path, max_images_per_map: int, max_depth_m: float) -> Dict:
    imgs = image_paths(dataset_root, max_images_per_map)
    pose_files = sorted(dataset_root.glob("pose-*.csv"), key=lambda p: int(p.stem.split("-")[-1]))
    pointclouds = sorted(dataset_root.glob("pointcloud-*.ply"), key=lambda p: int(p.stem.split("-")[-1]))
    stats = [depth_stats(path, max_depth_m) for path in imgs]
    return {
        "dataset_root": str(dataset_root),
        "map_count": len(pose_files),
        "pointcloud_count": len(pointclouds),
        "sampled_image_count": len(imgs),
        "valid_depth_rate": mean(item["valid_depth_rate"] for item in stats),
        "near_depth_rate": mean(item["near_depth_rate"] for item in stats),
        "far_or_saturated_rate": mean(item["far_or_saturated_rate"] for item in stats),
        "frontier_pixel_rate": mean(item["frontier_pixel_rate"] for item in stats),
        "depth_mean": mean(item["depth_mean"] for item in stats),
    }


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_annotated_jsonl(path: Path) -> Dict:
    rows = read_jsonl(path)
    gt_rows = [row for row in rows if parse_bool(row.get("valid_reaction_margin_gt"))]
    hidden_rows = [row for row in gt_rows if parse_bool(row.get("hidden_risk_gt"))]
    margins = [parse_float(row.get("reaction_margin_gt")) for row in gt_rows]
    hidden_margins = [parse_float(row.get("reaction_margin_gt")) for row in hidden_rows]
    margins = [m for m in margins if m is not None]
    hidden_margins = [m for m in hidden_margins if m is not None]
    return {
        "annotated_jsonl": str(path),
        "annotated_sample_count": len(rows),
        "gt_rmvr_valid_count": len(gt_rows),
        "gt_rmvr_coverage": len(gt_rows) / max(len(rows), 1),
        "hidden_risk_gt_count": len(hidden_rows),
        "hidden_risk_gt_coverage": len(hidden_rows) / max(len(rows), 1),
        "negative_margin_rate": mean(float(m < 0.0) for m in margins),
        "hidden_negative_margin_rate": mean(float(m < 0.0) for m in hidden_margins),
        "mean_reaction_margin_gt": mean(margins),
        "mean_reaction_margin_gt_hidden": mean(hidden_margins),
        "minimum_reaction_margin_gt": min(margins) if margins else None,
        "minimum_reaction_margin_gt_hidden": min(hidden_margins) if hidden_margins else None,
    }


def write_csv(path: Path, summary: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Summarize OARM/YOPO dataset coverage and optional GT margin annotations.")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--max-images-per-map", type=int, default=200)
    p.add_argument("--max-depth-m", type=float, default=20.0)
    p.add_argument("--annotated-jsonl", default=None)
    p.add_argument("--output-json", default=None)
    p.add_argument("--output-csv", default=None)
    return p


def main() -> None:
    args = parser().parse_args()
    summary = summarize_raw_dataset(Path(args.dataset_root), args.max_images_per_map, args.max_depth_m)
    if args.annotated_jsonl:
        summary.update(summarize_annotated_jsonl(Path(args.annotated_jsonl)))
    text = json.dumps(summary, indent=2)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    if args.output_csv:
        write_csv(Path(args.output_csv), summary)


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from OARM.utils.gt_risk_point_sampler import GTRiskPointSampler
from OARM.utils.occlusion import DepthFrontierExtractor


@dataclass
class SampleRecord:
    source_root: str
    source_map_id: int
    source_image_id: int
    image_path: str
    pose: Dict[str, str]
    frontier_pixel_rate: float
    far_or_saturated_rate: float
    valid_depth_rate: float
    raw_gt_risk_point_valid_rate: float
    raw_gt_risk_point_weight_sum: float
    raw_gt_risk_point_weight_mean: float
    bucket: str


def read_pose_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def write_pose_rows(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def image_id_from_path(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def sorted_images(map_dir: Path) -> List[Path]:
    return sorted(map_dir.glob("*.png"), key=image_id_from_path)


def read_depth(path: Path) -> torch.Tensor:
    image = cv2.imread(str(path), -1)
    if image is None:
        raise FileNotFoundError(path)
    depth = image.astype(np.float32) / 65535.0
    return torch.from_numpy(depth).float().unsqueeze(0)


def pose_to_rot_wb(row: Dict[str, str]) -> torch.Tensor:
    quat_wxyz = np.array(
        [
            float(row["qw"]),
            float(row["qx"]),
            float(row["qy"]),
            float(row["qz"]),
        ],
        dtype=np.float32,
    )
    rot = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_matrix()
    return torch.from_numpy(rot.astype(np.float32))


def pose_to_pos(row: Dict[str, str]) -> torch.Tensor:
    return torch.tensor([float(row["px"]), float(row["py"]), float(row["pz"])], dtype=torch.float32)


def classify_bucket(metrics: Dict[str, float], args: argparse.Namespace) -> str:
    gt_ok = metrics["raw_gt_risk_point_valid_rate"] >= args.min_gt_valid_rate
    frontier_ok = metrics["frontier_pixel_rate"] >= args.min_frontier_rate
    far_ok = metrics["far_or_saturated_rate"] >= args.min_far_rate
    weight_ok = metrics["raw_gt_risk_point_weight_sum"] >= args.min_gt_weight_sum
    if gt_ok and weight_ok and frontier_ok:
        return "occlusion_rich"
    if gt_ok and weight_ok and far_ok:
        return "reachable_hidden_risk"
    if frontier_ok or far_ok:
        return "frontier_only"
    return "low_risk"


def compute_sample_metrics(
    image_path: Path,
    pose_row: Dict[str, str],
    map_id: int,
    frontier_extractor: DepthFrontierExtractor,
    gt_sampler: GTRiskPointSampler,
) -> Dict[str, float]:
    depth = read_depth(image_path)
    depth_np = depth.squeeze(0).numpy()
    valid_depth = depth_np > 1e-6
    frontier = frontier_extractor(depth.unsqueeze(0)).squeeze(0)
    pos_w = pose_to_pos(pose_row)
    rot_wb = pose_to_rot_wb(pose_row)
    _risk_points_w, risk_weight = gt_sampler(depth, pos_w, rot_wb, map_id)
    return {
        "frontier_pixel_rate": float(frontier.mean().item()),
        "far_or_saturated_rate": float((depth_np >= 0.98).mean()),
        "valid_depth_rate": float(valid_depth.mean()),
        "raw_gt_risk_point_valid_rate": float((risk_weight > 1e-6).float().mean().item()),
        "raw_gt_risk_point_weight_sum": float(risk_weight.float().sum().item()),
        "raw_gt_risk_point_weight_mean": float(risk_weight.float().mean().item()),
    }


def iter_source_samples(root: Path, args: argparse.Namespace) -> Iterable[Tuple[int, int, Path, Dict[str, str], List[str]]]:
    pose_files = sorted(root.glob("pose-*.csv"), key=lambda p: int(p.stem.split("-")[-1]))
    for pose_path in pose_files:
        map_id = int(pose_path.stem.split("-")[-1])
        image_dir = root / str(map_id)
        pointcloud = root / f"pointcloud-{map_id}.ply"
        if not image_dir.is_dir() or not pointcloud.is_file():
            continue
        fieldnames, rows = read_pose_rows(pose_path)
        images = sorted_images(image_dir)
        if args.max_images_per_map is not None:
            images = images[: args.max_images_per_map]
        for image_path in images:
            image_id = image_id_from_path(image_path)
            if image_id >= len(rows):
                continue
            yield map_id, image_id, image_path, rows[image_id], fieldnames


def bucket_limits(total: int, args: argparse.Namespace) -> Dict[str, int]:
    return {
        "occlusion_rich": max(0, int(round(total * args.occlusion_rich_ratio))),
        "reachable_hidden_risk": max(0, int(round(total * args.reachable_risk_ratio))),
        "frontier_only": max(0, int(round(total * args.frontier_only_ratio))),
        "low_risk": max(0, int(round(total * args.low_risk_ratio))),
    }


def accept_record(record: SampleRecord, selected: Dict[str, List[SampleRecord]], limits: Dict[str, int]) -> bool:
    bucket = record.bucket
    if len(selected[bucket]) < limits[bucket]:
        selected[bucket].append(record)
        return True
    return False


def copy_selected_records(records: List[SampleRecord], output_root: Path, min_images_per_map: int) -> Dict:
    output_root.mkdir(parents=True, exist_ok=True)
    grouped: Dict[Tuple[str, int], List[SampleRecord]] = {}
    for record in records:
        key = (record.source_root, record.source_map_id)
        grouped.setdefault(key, []).append(record)

    manifest = {
        "format": "YOPO-compatible",
        "output_root": str(output_root),
        "map_count": 0,
        "image_count": 0,
        "maps": [],
        "bucket_counts": {},
    }
    filtered_groups = [item for item in sorted(grouped.items()) if len(item[1]) >= min_images_per_map]
    for new_map_id, ((source_root, source_map_id), map_records) in enumerate(filtered_groups):
        source_root_path = Path(source_root)
        source_pointcloud = source_root_path / f"pointcloud-{source_map_id}.ply"
        shutil.copy2(source_pointcloud, output_root / f"pointcloud-{new_map_id}.ply")
        dst_image_dir = output_root / str(new_map_id)
        dst_image_dir.mkdir(parents=True, exist_ok=True)

        pose_rows: List[Dict[str, str]] = []
        fieldnames = list(map_records[0].pose.keys())
        for new_image_id, record in enumerate(sorted(map_records, key=lambda r: r.source_image_id)):
            dst_image = dst_image_dir / f"img_{new_image_id}.png"
            shutil.copy2(record.image_path, dst_image)
            pose_rows.append(dict(record.pose))
            manifest["bucket_counts"][record.bucket] = manifest["bucket_counts"].get(record.bucket, 0) + 1

        write_pose_rows(output_root / f"pose-{new_map_id}.csv", fieldnames, pose_rows)
        manifest["maps"].append(
            {
                "source_root": source_root,
                "source_map_id": source_map_id,
                "merged_map_id": new_map_id,
                "image_count": len(map_records),
            }
        )
    manifest["map_count"] = len(manifest["maps"])
    manifest["image_count"] = int(sum(item["image_count"] for item in manifest["maps"]))
    return manifest


def summarize(records: List[SampleRecord]) -> Dict:
    def mean(key: str) -> float:
        if not records:
            return 0.0
        return float(sum(getattr(record, key) for record in records) / len(records))

    bucket_counts: Dict[str, int] = {}
    for record in records:
        bucket_counts[record.bucket] = bucket_counts.get(record.bucket, 0) + 1
    return {
        "sample_count": len(records),
        "bucket_counts": bucket_counts,
        "frontier_pixel_rate": mean("frontier_pixel_rate"),
        "far_or_saturated_rate": mean("far_or_saturated_rate"),
        "raw_gt_risk_point_valid_rate": mean("raw_gt_risk_point_valid_rate"),
        "raw_gt_risk_point_weight_sum": mean("raw_gt_risk_point_weight_sum"),
        "raw_gt_risk_point_weight_mean": mean("raw_gt_risk_point_weight_mean"),
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Curate YOPO-format raw simulator data into an OARM occlusion-rich dataset.")
    p.add_argument("--input-root", action="append", required=True, help="Raw YOPO-format dataset root generated by Simulator.")
    p.add_argument("--output-root", required=True, help="Curated YOPO-format dataset root.")
    p.add_argument("--target-images", type=int, default=60000)
    p.add_argument("--max-images-per-map", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--min-images-per-output-map", type=int, default=2)
    p.add_argument("--manifest-name", default="oarm_curation_manifest.json")
    p.add_argument("--metrics-jsonl", default="")
    p.add_argument("--min-frontier-rate", type=float, default=0.025)
    p.add_argument("--min-far-rate", type=float, default=0.08)
    p.add_argument("--min-gt-valid-rate", type=float, default=0.25)
    p.add_argument("--min-gt-weight-sum", type=float, default=2.0)
    p.add_argument("--occlusion-rich-ratio", type=float, default=0.50)
    p.add_argument("--reachable-risk-ratio", type=float, default=0.25)
    p.add_argument("--frontier-only-ratio", type=float, default=0.15)
    p.add_argument("--low-risk-ratio", type=float, default=0.10)
    p.add_argument("--gt-risk-point-count", type=int, default=64)
    p.add_argument("--gt-max-forward-m", type=float, default=10.0)
    p.add_argument("--gt-horizon-fov-expand-deg", type=float, default=90.0)
    p.add_argument("--gt-vertical-fov-expand-deg", type=float, default=20.0)
    p.add_argument("--gt-reachable-score-weight", type=float, default=0.65)
    p.add_argument("--print-every", type=int, default=1000)
    return p


def main() -> None:
    args = parser().parse_args()
    output_root = Path(args.output_root)
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{output_root} is not empty; pass --overwrite to replace it.")
        shutil.rmtree(output_root)

    selected: Dict[str, List[SampleRecord]] = {
        "occlusion_rich": [],
        "reachable_hidden_risk": [],
        "frontier_only": [],
        "low_risk": [],
    }
    limits = bucket_limits(args.target_images, args)
    frontier_extractor = DepthFrontierExtractor()
    metrics_path = Path(args.metrics_jsonl) if args.metrics_jsonl else None
    metrics_file = None
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_path.open("w", encoding="utf-8")

    try:
        seen = 0
        for input_root_str in args.input_root:
            input_root = Path(input_root_str).resolve()
            gt_sampler = GTRiskPointSampler(
                str(input_root),
                point_count=args.gt_risk_point_count,
                max_forward_m=args.gt_max_forward_m,
                horizon_fov_expand_deg=args.gt_horizon_fov_expand_deg,
                vertical_fov_expand_deg=args.gt_vertical_fov_expand_deg,
                reachable_score_weight=args.gt_reachable_score_weight,
            )
            for map_id, image_id, image_path, pose_row, _fieldnames in iter_source_samples(input_root, args):
                seen += 1
                metrics = compute_sample_metrics(image_path, pose_row, map_id, frontier_extractor, gt_sampler)
                bucket = classify_bucket(metrics, args)
                record = SampleRecord(
                    source_root=str(input_root),
                    source_map_id=map_id,
                    source_image_id=image_id,
                    image_path=str(image_path),
                    pose=pose_row,
                    bucket=bucket,
                    **metrics,
                )
                if metrics_file is not None:
                    metrics_file.write(json.dumps(asdict(record), sort_keys=True) + "\n")
                accept_record(record, selected, limits)
                selected_count = sum(len(items) for items in selected.values())
                if args.print_every and seen % args.print_every == 0:
                    print(f"seen={seen} selected={selected_count} buckets={ {k: len(v) for k, v in selected.items()} }")
                if selected_count >= args.target_images:
                    break
            if sum(len(items) for items in selected.values()) >= args.target_images:
                break
    finally:
        if metrics_file is not None:
            metrics_file.close()

    records: List[SampleRecord] = []
    for bucket in ("occlusion_rich", "reachable_hidden_risk", "frontier_only", "low_risk"):
        records.extend(selected[bucket])
    manifest = copy_selected_records(records, output_root, args.min_images_per_output_map)
    summary = summarize(records)
    manifest.update(
        {
            "curation_args": vars(args),
            "summary": summary,
        }
    )
    (output_root / args.manifest_name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote curated dataset: {output_root}")


if __name__ == "__main__":
    main()

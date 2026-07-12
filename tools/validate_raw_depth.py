import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def summarize_image(path: str) -> dict:
    image = cv2.imread(path, -1)
    if image is None:
        return {"path": path, "readable": False}
    return {
        "path": path,
        "readable": True,
        "dtype": str(image.dtype),
        "shape": list(image.shape),
        "min": int(image.min()),
        "max": int(image.max()),
        "mean": float(image.mean()),
        "nonzero": int(np.count_nonzero(image)),
        "pixels": int(image.size),
    }


def validate(args: argparse.Namespace) -> int:
    root = Path(args.dataset_root)
    image_paths = sorted(glob.glob(str(root / "*" / "img_*.png")))
    pointcloud_paths = sorted(root.glob("pointcloud-*.ply"))
    pose_paths = sorted(root.glob("pose-*.csv"))

    print(f"[OARM raw-depth check] root={root}")
    print(f"[OARM raw-depth check] png={len(image_paths)} pointcloud={len(pointcloud_paths)} pose_csv={len(pose_paths)}")

    if not image_paths:
        print("[OARM raw-depth check] ERROR: no depth PNG files found.", file=sys.stderr)
        return 2
    if not pointcloud_paths:
        print("[OARM raw-depth check] ERROR: no pointcloud-*.ply files found.", file=sys.stderr)
        return 2

    stride = max(1, len(image_paths) // max(args.sample_limit, 1))
    samples = image_paths[::stride][: args.sample_limit]
    stats = [summarize_image(path) for path in samples]
    unreadable = [item for item in stats if not item.get("readable")]
    if unreadable:
        print(f"[OARM raw-depth check] ERROR: unreadable PNG: {unreadable[0]['path']}", file=sys.stderr)
        return 2

    nonzero_images = [item for item in stats if item["nonzero"] > 0 and item["max"] > 0]
    nonzero_ratio = len(nonzero_images) / max(len(stats), 1)
    max_value = max(item["max"] for item in stats)
    mean_value = float(np.mean([item["mean"] for item in stats]))
    print(
        "[OARM raw-depth check] "
        f"samples={len(stats)} nonzero_ratio={nonzero_ratio:.3f} max={max_value} mean={mean_value:.2f}"
    )
    for item in stats[: min(args.print_samples, len(stats))]:
        print(
            "[OARM raw-depth check] sample "
            f"{item['path']} dtype={item['dtype']} shape={item['shape']} "
            f"min={item['min']} max={item['max']} mean={item['mean']:.2f} nonzero={item['nonzero']}"
        )

    if nonzero_ratio < args.min_nonzero_image_ratio or max_value <= 0:
        print(
            "[OARM raw-depth check] ERROR: depth images look all-black or mostly invalid.\n"
            "Most likely causes are: CUDA kernel did not run, CUDA architecture mismatch, "
            "container was started without GPU access, or the generated map/pointcloud is empty.\n"
            "Check `nvidia-smi`, rebuild Simulator for this GPU, and inspect the dataset_generator log.",
            file=sys.stderr,
        )
        return 3
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate YOPO-format raw depth images before OARM curation.")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--sample-limit", type=int, default=24)
    p.add_argument("--print-samples", type=int, default=3)
    p.add_argument("--min-nonzero-image-ratio", type=float, default=0.8)
    return p


def main() -> None:
    raise SystemExit(validate(parser().parse_args()))


if __name__ == "__main__":
    main()

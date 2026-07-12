import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def read_pose_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def discover_roots(paths: Iterable[str]) -> List[Path]:
    roots = [Path(path).resolve() for path in paths]
    missing = [str(path) for path in roots if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing dataset roots: {missing}")
    return roots


def merge_dataset_roots(roots: List[Path], output_root: Path) -> Dict:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "YOPO-compatible",
        "output_root": str(output_root),
        "sources": [],
        "maps": [],
    }
    next_map_id = 0
    for source_idx, root in enumerate(roots):
        source_meta = {"source_idx": source_idx, "root": str(root), "maps": []}
        pose_files = sorted(root.glob("pose-*.csv"), key=lambda p: int(p.stem.split("-")[-1]))
        for pose_file in pose_files:
            old_map_id = int(pose_file.stem.split("-")[-1])
            image_dir = root / str(old_map_id)
            pointcloud = root / f"pointcloud-{old_map_id}.ply"
            if not image_dir.is_dir() or not pointcloud.is_file():
                continue

            new_map_id = next_map_id
            next_map_id += 1
            copy_file(pointcloud, output_root / f"pointcloud-{new_map_id}.ply")
            dst_img_dir = output_root / str(new_map_id)
            dst_img_dir.mkdir(parents=True, exist_ok=True)
            image_count = 0
            for image_path in sorted(image_dir.glob("*.png"), key=lambda p: int(p.stem.split("_")[-1])):
                copy_file(image_path, dst_img_dir / image_path.name)
                image_count += 1

            fieldnames, rows = read_pose_rows(pose_file)
            for row in rows:
                for key in ("map_id", "map_idx", "env_id"):
                    if key in row:
                        row[key] = str(new_map_id)
            dst_pose = output_root / f"pose-{new_map_id}.csv"
            with dst_pose.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            map_meta = {
                "source_root": str(root),
                "source_map_id": old_map_id,
                "merged_map_id": new_map_id,
                "image_count": image_count,
                "pose_count": len(rows),
            }
            source_meta["maps"].append(map_meta)
            manifest["maps"].append(map_meta)
        manifest["sources"].append(source_meta)
    manifest["map_count"] = len(manifest["maps"])
    manifest["image_count"] = int(sum(item["image_count"] for item in manifest["maps"]))
    return manifest


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merge multiple YOPO-format dataset roots into one root.")
    p.add_argument("--input-root", action="append", required=True, help="YOPO-format dataset root to merge.")
    p.add_argument("--output-root", required=True, help="Merged YOPO-format dataset root.")
    p.add_argument("--manifest-name", default="merge_manifest.json")
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    output_root = Path(args.output_root)
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{output_root} is not empty; pass --overwrite to replace it.")
        shutil.rmtree(output_root)
    roots = discover_roots(args.input_root)
    manifest = merge_dataset_roots(roots, output_root)
    manifest_path = output_root / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Merged {manifest['map_count']} maps and {manifest['image_count']} images into {output_root}")


if __name__ == "__main__":
    main()

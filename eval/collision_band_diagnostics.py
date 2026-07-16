import argparse
import csv
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


VECTOR_FIELDS = (
    "first_collision_position_w",
    "min_clearance_exec_raw_position_w",
    "min_clearance_exec_position_w",
)


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_vector(value):
    if value in (None, ""):
        return None
    if isinstance(value, str):
        value = json.loads(value)
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        return None
    return arr


def parse_float(value, default=None):
    if value in (None, ""):
        return default
    try:
        parsed = float(value)
        return parsed if np.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def first_present(rows: Iterable[Dict], key: str):
    for row in rows:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def method_label(rows: List[Dict], path: str) -> Tuple[str, str]:
    method = first_present(rows, "method") or os.path.splitext(os.path.basename(path))[0]
    scenario = first_present(rows, "scenario") or os.path.basename(os.path.dirname(path))
    return str(method), str(scenario)


def first_collision_position(rows: List[Dict]) -> Tuple[Optional[np.ndarray], str]:
    for field in VECTOR_FIELDS:
        value = first_present(rows, field)
        pos = parse_vector(value)
        if pos is not None:
            return pos, field
    return None, "missing"


def load_pointcloud(dataset_dir: str, map_id: int):
    path = os.path.join(dataset_dir, f"pointcloud-{int(map_id)}.ply")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"GT pointcloud not found: {path}")
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points, dtype=np.float64)
    if points.size == 0:
        raise ValueError(f"GT pointcloud has no points: {path}")
    return path, points


def crop_points(points: np.ndarray, bounds: Dict[str, Tuple[float, float]]) -> np.ndarray:
    mask = (
        (points[:, 0] >= bounds["x"][0])
        & (points[:, 0] <= bounds["x"][1])
        & (points[:, 1] >= bounds["y"][0])
        & (points[:, 1] <= bounds["y"][1])
        & (points[:, 2] >= bounds["z"][0])
        & (points[:, 2] <= bounds["z"][1])
    )
    return points[mask]


def write_pointcloud(path: str, points: np.ndarray, color=None):
    import open3d as o3d

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if color is not None and len(points):
        colors = np.tile(np.asarray(color, dtype=np.float64)[None, :], (len(points), 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pcd)


def write_csv(path: str, rows: List[Dict]):
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys = sorted(set().union(*(row.keys() for row in rows)))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def clearance_at(tree: cKDTree, pos: np.ndarray) -> float:
    distance, _ = tree.query(np.asarray(pos, dtype=np.float64), k=1)
    return float(distance)


def diagnostics(args):
    logs = []
    for path in args.logs:
        if os.path.isdir(path):
            for root, _dirs, files in os.walk(path):
                for name in files:
                    if name.endswith(".jsonl"):
                        logs.append(os.path.join(root, name))
        else:
            logs.append(path)
    logs = sorted(logs)

    markers = []
    for path in logs:
        rows = read_jsonl(path)
        if not rows:
            continue
        position, source = first_collision_position(rows)
        if position is None:
            continue
        method, scenario = method_label(rows, path)
        markers.append(
            {
                "path": path,
                "method": method,
                "scenario": scenario,
                "position_source": source,
                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2]),
                "first_collision_clearance": parse_float(first_present(rows, "first_collision_clearance")),
                "first_collision_goal_distance": parse_float(first_present(rows, "first_collision_goal_distance")),
                "time_to_collision_from_motion_exec": parse_float(
                    first_present(rows, "time_to_collision_from_motion_exec")
                ),
                "distance_travelled_before_collision_exec": parse_float(
                    first_present(rows, "distance_travelled_before_collision_exec")
                ),
                "progress_at_collision_exec": parse_float(first_present(rows, "progress_at_collision_exec")),
            }
        )

    bounds = {
        "x": (args.x_min, args.x_max),
        "y": (args.y_min, args.y_max),
        "z": (args.z_min, args.z_max),
    }
    output = {
        "logs": logs,
        "marker_count": len(markers),
        "crop_bounds": bounds,
        "markers": markers,
    }

    if args.dataset_dir:
        map_path, points = load_pointcloud(args.dataset_dir, args.map_id)
        crop = crop_points(points, bounds)
        tree = cKDTree(points)
        output.update(
            {
                "map_path": map_path,
                "point_count": int(len(points)),
                "crop_point_count": int(len(crop)),
                "crop_min": crop.min(axis=0).astype(float).tolist() if len(crop) else None,
                "crop_max": crop.max(axis=0).astype(float).tolist() if len(crop) else None,
                "crop_mean": crop.mean(axis=0).astype(float).tolist() if len(crop) else None,
            }
        )
        probe_points = {
            "start_0_0_2": np.asarray([0.0, 0.0, 2.0]),
            "collision_band_center_2p77_0_2": np.asarray([2.77, 0.0, 2.0]),
            "forward_5_0_2": np.asarray([5.0, 0.0, 2.0]),
        }
        output["probe_clearances"] = {
            name: clearance_at(tree, point) for name, point in probe_points.items()
        }
        for marker in markers:
            pos = np.asarray([marker["x"], marker["y"], marker["z"]], dtype=np.float64)
            marker["gt_pointcloud_clearance"] = clearance_at(tree, pos)
        if args.crop_output:
            write_pointcloud(args.crop_output, crop, color=(0.7, 0.7, 0.7))
            output["crop_output"] = args.crop_output
        if args.marker_output and markers:
            marker_points = np.asarray([[m["x"], m["y"], m["z"]] for m in markers], dtype=np.float64)
            write_pointcloud(args.marker_output, marker_points, color=(1.0, 0.0, 0.0))
            output["marker_output"] = args.marker_output

    if args.csv_output:
        write_csv(args.csv_output, markers)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, sort_keys=True)
    print(json.dumps(output, indent=2, sort_keys=True))


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+", help="execution-monitor JSONL logs or directories")
    p.add_argument("--dataset-dir", default="", help="directory containing pointcloud-*.ply")
    p.add_argument("--map-id", type=int, default=0)
    p.add_argument("--x-min", type=float, default=2.4)
    p.add_argument("--x-max", type=float, default=3.1)
    p.add_argument("--y-min", type=float, default=-5.0)
    p.add_argument("--y-max", type=float, default=5.0)
    p.add_argument("--z-min", type=float, default=0.5)
    p.add_argument("--z-max", type=float, default=3.5)
    p.add_argument("--output", default="")
    p.add_argument("--csv-output", default="")
    p.add_argument("--crop-output", default="")
    p.add_argument("--marker-output", default="")
    return p


if __name__ == "__main__":
    diagnostics(parser().parse_args())

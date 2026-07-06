import argparse
import json
import os
from functools import lru_cache

import numpy as np
from scipy.spatial import cKDTree


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


@lru_cache(maxsize=16)
def load_pointcloud(map_id, dataset_dir):
    path = os.path.join(dataset_dir, f"pointcloud-{int(map_id)}.ply")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"GT pointcloud not found: {path}")
    import open3d as o3d

    pointcloud = o3d.io.read_point_cloud(path)
    points = np.asarray(pointcloud.points, dtype=np.float32)
    if points.size == 0:
        raise ValueError(f"GT pointcloud has no points: {path}")
    return points, cKDTree(points)


def parse_vector(row, keys):
    for key, source in keys:
        value = row.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, str):
            value = json.loads(value)
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape == (3,):
            return arr, source
    return None, "missing_executed_positions"


def parse_float(value, default=None):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def active_rows(rows):
    if not any("run_active" in row for row in rows):
        return rows
    return [row for row in rows if parse_bool(row.get("run_active"))]


def filter_run(rows, run_id):
    if not rows:
        return rows
    if run_id:
        return [row for row in rows if str(row.get("run_id", "")) == str(run_id)]
    run_ids = [row.get("run_id") for row in rows if row.get("run_id") not in (None, "")]
    if not run_ids:
        return rows
    latest = run_ids[-1]
    return [row for row in rows if row.get("run_id") == latest]


def filter_goal_segment(rows, goal_segment_id):
    if not rows:
        return rows
    if goal_segment_id != "":
        return [row for row in rows if str(row.get("goal_segment_id", "")) == str(goal_segment_id)]
    segment_ids = [row.get("goal_segment_id") for row in rows if row.get("goal_segment_id") not in (None, "")]
    if not segment_ids:
        return rows
    latest = segment_ids[-1]
    return [row for row in rows if str(row.get("goal_segment_id", "")) == str(latest)]


def densify_positions(positions, step):
    if positions.shape[0] < 2 or step <= 0.0:
        return positions
    samples = [positions[0]]
    for start, end in zip(positions[:-1], positions[1:]):
        delta = end - start
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-6:
            continue
        segment_count = max(1, int(np.ceil(distance / step)))
        for i in range(1, segment_count + 1):
            alpha = i / segment_count
            samples.append(start + alpha * delta)
    return np.stack(samples, axis=0).astype(np.float32)


def executed_positions(rows):
    positions = []
    sources = []
    for row in rows:
        pos, source = parse_vector(
            row,
            (
                ("odom_pos_w", "executed_odom_to_gt_pointcloud"),
                ("position_w", "position_field_to_gt_pointcloud"),
                ("pos_w", "position_field_to_gt_pointcloud"),
                ("start_pos_w", "reference_or_planner_start_to_gt_pointcloud"),
            ),
        )
        if pos is not None:
            positions.append(pos)
            sources.append(source)
    if not positions:
        return np.zeros((0, 3), dtype=np.float32), "missing_executed_positions"
    if "executed_odom_to_gt_pointcloud" in sources:
        source = "executed_odom_to_gt_pointcloud"
    elif "position_field_to_gt_pointcloud" in sources:
        source = "position_field_to_gt_pointcloud"
    else:
        source = "reference_or_planner_start_to_gt_pointcloud"
    return np.stack(positions, axis=0), source


def trim_at_first_arrival(rows, success_distance):
    if success_distance <= 0.0:
        return rows, False
    for idx, row in enumerate(rows):
        goal_distance = parse_float(row.get("goal_distance"))
        if goal_distance is not None and goal_distance <= success_distance:
            return rows[: idx + 1], True
    return rows, False


def execution_summary(rows, args):
    rows = active_rows(filter_goal_segment(filter_run(rows, args.run_id), args.goal_segment_id))
    if not args.keep_after_arrival:
        rows, reached_goal_during_run = trim_at_first_arrival(rows, args.success_distance)
    else:
        reached_goal_during_run = False
    times = [parse_float(row.get("time", row.get("timestamp"))) for row in rows]
    times = [time for time in times if time is not None]
    positions, clearance_source = executed_positions(rows)
    raw_positions = positions
    raw_position_count = int(positions.shape[0])
    positions = densify_positions(positions, args.clearance_sample_step)
    map_id = int(rows[0].get("map_id", args.map_id)) if rows else int(args.map_id)

    min_clearance = None
    mean_clearance = None
    min_clearance_position = None
    min_clearance_raw_position = None
    min_clearance_raw_time = None
    min_clearance_raw_goal_distance = None
    collision_exec = False
    if positions.shape[0] > 0:
        _points, tree = load_pointcloud(map_id, args.dataset_dir)
        distances, _ = tree.query(positions, k=1)
        dense_min_index = int(np.argmin(distances))
        min_clearance = float(distances[dense_min_index])
        mean_clearance = float(np.mean(distances))
        min_clearance_position = positions[dense_min_index].astype(float).tolist()
        if raw_positions.shape[0] > 0:
            raw_distances, _ = tree.query(raw_positions, k=1)
            raw_min_index = int(np.argmin(raw_distances))
            min_clearance_raw_position = raw_positions[raw_min_index].astype(float).tolist()
            if raw_position_count == len(rows):
                raw_row = rows[raw_min_index]
                min_clearance_raw_time = parse_float(raw_row.get("time", raw_row.get("timestamp")))
                min_clearance_raw_goal_distance = parse_float(raw_row.get("goal_distance"))
        collision_exec = bool(min_clearance < args.collision_clearance)

    goal_distance_values = [parse_float(row.get("goal_distance")) for row in rows]
    goal_distance_values = [value for value in goal_distance_values if value is not None]
    goal_distance_final = goal_distance_values[-1] if goal_distance_values else None
    goal_distance_min = min(goal_distance_values) if goal_distance_values else None

    path_time = (max(times) - min(times)) if times else 0.0
    timeout_exec = bool(args.max_time > 0.0 and path_time >= args.max_time)
    reached_goal = bool(
        reached_goal_during_run
        or (goal_distance_final is not None and goal_distance_final <= args.success_distance)
        or (goal_distance_min is not None and goal_distance_min <= args.success_distance)
    )
    success_exec = bool(reached_goal and not collision_exec and not timeout_exec)
    speeds = [parse_float(row.get("speed")) for row in rows]
    speeds = [speed for speed in speeds if speed is not None]

    return {
        "collision_exec": collision_exec,
        "success_exec": success_exec,
        "timeout_exec": timeout_exec,
        "min_clearance_exec": min_clearance,
        "mean_clearance_exec": mean_clearance,
        "min_clearance_exec_position_w": min_clearance_position,
        "min_clearance_exec_raw_position_w": min_clearance_raw_position,
        "min_clearance_exec_raw_time": min_clearance_raw_time,
        "min_clearance_exec_raw_goal_distance": min_clearance_raw_goal_distance,
        "path_time_exec": path_time,
        "mean_speed_exec": float(np.mean(speeds)) if speeds else None,
        "goal_distance_final": goal_distance_final,
        "goal_distance_min": goal_distance_min,
        "reached_goal_exec": reached_goal,
        "collision_exec_source": clearance_source,
        "success_exec_source": "final_goal_distance_and_monitor",
        "exec_rows_active": int(len(rows)),
        "exec_position_count": raw_position_count,
        "exec_clearance_sample_count": int(positions.shape[0]),
        "exec_clearance_sample_step": float(args.clearance_sample_step),
        "goal_segment_id": rows[0].get("goal_segment_id") if rows else None,
        "monitor_collision_clearance": float(args.collision_clearance),
        "monitor_success_distance": float(args.success_distance),
        "monitor_max_time": float(args.max_time),
        "monitor_keep_after_arrival": bool(args.keep_after_arrival),
    }


def monitor(args):
    rows = filter_goal_segment(filter_run(list(read_jsonl(args.input)), args.run_id), args.goal_segment_id)
    exec_rows = list(read_jsonl(args.exec_input)) if args.exec_input else rows
    summary = execution_summary(exec_rows, args)
    summary["execution_monitor_input"] = args.exec_input or args.input
    annotated = [{**row, **summary} for row in rows]
    write_jsonl(args.output, annotated)
    print(
        json.dumps(
            {"rows": len(rows), "exec_rows": len(exec_rows), "output": args.output, **summary},
            indent=2,
            sort_keys=True,
        )
    )


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="ROS benchmark JSONL from test_oarm_ros.py --log-jsonl")
    p.add_argument("--output", required=True, help="JSONL with execution monitor fields added to every row")
    p.add_argument("--exec-input", default="", help="optional high-rate odometry JSONL from --exec-log-jsonl")
    p.add_argument("--run-id", default="", help="optional run_id filter; defaults to the latest run_id in the log")
    p.add_argument("--goal-segment-id", default="", help="optional goal_segment_id filter; defaults to the latest segment")
    p.add_argument("--dataset-dir", default="dataset", help="directory containing pointcloud-*.ply")
    p.add_argument("--map-id", type=int, default=0, help="fallback map id when rows do not contain map_id")
    p.add_argument("--collision-clearance", type=float, default=0.25)
    p.add_argument("--success-distance", type=float, default=1.0)
    p.add_argument("--max-time", type=float, default=0.0, help="seconds; <=0 disables timeout")
    p.add_argument("--clearance-sample-step", type=float, default=0.05, help="meters between interpolated clearance samples")
    p.add_argument("--keep-after-arrival", action="store_true", help="include rows after first entering success distance")
    return p


if __name__ == "__main__":
    monitor(parser().parse_args())

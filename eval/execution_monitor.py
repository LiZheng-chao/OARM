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


def row_time(row):
    return parse_float(row.get("time", row.get("timestamp")))


def row_goal_distance(row):
    return parse_float(row.get("goal_distance"))


def interpolate_optional(start, end, alpha):
    if start is None or end is None:
        return np.nan
    return float(start + alpha * (end - start))


def densify_trajectory(positions, times, goal_distances, step):
    if positions.shape[0] < 2 or step <= 0.0:
        dense_times = None if times is None else np.asarray(times, dtype=np.float64)
        dense_goals = None if goal_distances is None else np.asarray(goal_distances, dtype=np.float64)
        return positions, dense_times, dense_goals
    dense_times = [] if times is not None else None
    dense_goals = [] if goal_distances is not None else None
    samples = [positions[0]]
    if dense_times is not None:
        dense_times.append(times[0])
    if dense_goals is not None:
        dense_goals.append(goal_distances[0])
    for segment_idx, (start, end) in enumerate(zip(positions[:-1], positions[1:])):
        delta = end - start
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-6:
            continue
        segment_count = max(1, int(np.ceil(distance / step)))
        for i in range(1, segment_count + 1):
            alpha = i / segment_count
            samples.append(start + alpha * delta)
            if dense_times is not None:
                dense_times.append(interpolate_optional(times[segment_idx], times[segment_idx + 1], alpha))
            if dense_goals is not None:
                dense_goals.append(interpolate_optional(goal_distances[segment_idx], goal_distances[segment_idx + 1], alpha))
    dense_positions = np.stack(samples, axis=0).astype(np.float32)
    dense_times = None if dense_times is None else np.asarray(dense_times, dtype=np.float64)
    dense_goals = None if dense_goals is None else np.asarray(dense_goals, dtype=np.float64)
    return dense_positions, dense_times, dense_goals


def densify_positions(positions, step):
    return densify_trajectory(positions, None, None, step)[0]


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


def first_arrival_event(rows, success_distance):
    if success_distance <= 0.0:
        return None
    for row in rows:
        goal_distance = row_goal_distance(row)
        if goal_distance is not None and goal_distance <= success_distance:
            position, _source = parse_vector(
                row,
                (
                    ("odom_pos_w", "executed_odom_to_gt_pointcloud"),
                    ("position_w", "position_field_to_gt_pointcloud"),
                    ("pos_w", "position_field_to_gt_pointcloud"),
                    ("start_pos_w", "reference_or_planner_start_to_gt_pointcloud"),
                ),
            )
            return {
                "event": "arrival",
                "time": row_time(row),
                "position": position,
                "goal_distance": goal_distance,
            }
    return None


def first_collision_event(rows, args, map_id):
    raw_positions, source = executed_positions(rows)
    if raw_positions.shape[0] == 0 or args.collision_clearance <= 0.0:
        return None, source
    times = [row_time(row) for row in rows] if len(rows) == raw_positions.shape[0] else None
    goals = [row_goal_distance(row) for row in rows] if len(rows) == raw_positions.shape[0] else None
    positions, dense_times, dense_goals = densify_trajectory(raw_positions, times, goals, args.clearance_sample_step)
    _points, tree = load_pointcloud(map_id, args.dataset_dir)
    distances, _ = tree.query(positions, k=1)
    hit_indices = np.flatnonzero(distances < args.collision_clearance)
    if hit_indices.size == 0:
        return None, source
    idx = int(hit_indices[0])
    event_time = None
    event_goal_distance = None
    if dense_times is not None and np.isfinite(dense_times[idx]):
        event_time = float(dense_times[idx])
    if dense_goals is not None and np.isfinite(dense_goals[idx]):
        event_goal_distance = float(dense_goals[idx])
    return {
        "event": "collision",
        "time": event_time,
        "position": positions[idx].astype(float).tolist(),
        "goal_distance": event_goal_distance,
        "clearance": float(distances[idx]),
        "source": source,
    }, source


def event_position_list(event):
    if not event:
        return None
    position = event.get("position")
    if position is None:
        return None
    if isinstance(position, np.ndarray):
        return position.astype(float).tolist()
    return position


def choose_terminal_event(arrival_event, collision_event, args):
    candidates = []
    if arrival_event is not None and not args.keep_after_arrival:
        candidates.append(arrival_event)
    if collision_event is not None and not args.keep_after_collision:
        candidates.append(collision_event)
    if not candidates:
        return None
    timed = [event for event in candidates if event.get("time") is not None]
    if timed:
        return min(timed, key=lambda event: event["time"])
    return candidates[0]


def trim_at_time(rows, cutoff_time):
    if cutoff_time is None:
        return rows
    trimmed = []
    for row in rows:
        time = row_time(row)
        if time is None or time <= cutoff_time:
            trimmed.append(row)
    if trimmed:
        return trimmed
    return rows[:1]


def event_happened_before_or_at(event, terminal_event):
    if event is None:
        return False
    if terminal_event is None:
        return True
    event_time = event.get("time")
    terminal_time = terminal_event.get("time")
    if event_time is None or terminal_time is None:
        return event.get("event") == terminal_event.get("event")
    return event_time <= terminal_time + 1e-6


def execution_summary(rows, args):
    rows = active_rows(filter_goal_segment(filter_run(rows, args.run_id), args.goal_segment_id))
    untrimmed_count = len(rows)
    map_id = int(rows[0].get("map_id", args.map_id)) if rows else int(args.map_id)
    arrival_event = first_arrival_event(rows, args.success_distance)
    collision_event, clearance_source = first_collision_event(rows, args, map_id)
    terminal_event = choose_terminal_event(arrival_event, collision_event, args)
    if terminal_event is not None:
        rows = trim_at_time(rows, terminal_event.get("time"))

    times = [row_time(row) for row in rows]
    times = [time for time in times if time is not None]
    positions, clearance_source = executed_positions(rows)
    raw_positions = positions
    raw_position_count = int(positions.shape[0])
    positions = densify_positions(positions, args.clearance_sample_step)

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
                min_clearance_raw_time = row_time(raw_row)
                min_clearance_raw_goal_distance = row_goal_distance(raw_row)
        collision_exec = bool(min_clearance < args.collision_clearance)

    if collision_event is not None and event_happened_before_or_at(collision_event, terminal_event):
        collision_exec = True
        event_clearance = collision_event.get("clearance")
        if event_clearance is not None and (min_clearance is None or event_clearance < min_clearance):
            min_clearance = event_clearance
            min_clearance_position = event_position_list(collision_event)
            min_clearance_raw_position = event_position_list(collision_event)
            min_clearance_raw_time = collision_event.get("time")
            min_clearance_raw_goal_distance = collision_event.get("goal_distance")

    goal_distance_values = [row_goal_distance(row) for row in rows]
    goal_distance_values = [value for value in goal_distance_values if value is not None]
    goal_distance_final = goal_distance_values[-1] if goal_distance_values else None
    goal_distance_min = min(goal_distance_values) if goal_distance_values else None

    terminal_time = terminal_event.get("time") if terminal_event else None
    path_time = (max(times) - min(times)) if times else 0.0
    if terminal_time is not None and times:
        path_time = max(0.0, float(terminal_time) - min(times))
    timeout_exec = bool(args.max_time > 0.0 and path_time >= args.max_time)
    reached_goal = bool(event_happened_before_or_at(arrival_event, terminal_event))
    success_exec = bool(reached_goal and not collision_exec and not timeout_exec)
    speeds = [parse_float(row.get("speed")) for row in rows]
    speeds = [speed for speed in speeds if speed is not None]

    first_time = min(times) if times else None
    first_collision_time = collision_event.get("time") if collision_event else None
    time_to_collision = None
    if first_time is not None and first_collision_time is not None:
        time_to_collision = max(0.0, float(first_collision_time) - first_time)

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
        "success_exec_source": "first_terminal_event_monitor",
        "exec_rows_active": int(len(rows)),
        "exec_rows_untrimmed": int(untrimmed_count),
        "exec_position_count": raw_position_count,
        "exec_clearance_sample_count": int(positions.shape[0]),
        "exec_clearance_sample_step": float(args.clearance_sample_step),
        "goal_segment_id": rows[0].get("goal_segment_id") if rows else None,
        "first_arrival_time_exec": arrival_event.get("time") if arrival_event else None,
        "first_arrival_position_w": event_position_list(arrival_event),
        "first_collision_time_exec": first_collision_time,
        "first_collision_position_w": event_position_list(collision_event),
        "first_collision_goal_distance": collision_event.get("goal_distance") if collision_event else None,
        "first_collision_clearance": collision_event.get("clearance") if collision_event else None,
        "time_to_collision_exec": time_to_collision,
        "monitor_terminal_event": terminal_event.get("event") if terminal_event else None,
        "monitor_terminal_time": terminal_time,
        "monitor_trimmed_at_terminal": bool(terminal_event is not None),
        "monitor_collision_clearance": float(args.collision_clearance),
        "monitor_success_distance": float(args.success_distance),
        "monitor_max_time": float(args.max_time),
        "monitor_keep_after_arrival": bool(args.keep_after_arrival),
        "monitor_keep_after_collision": bool(args.keep_after_collision),
    }


def monitor(args):
    rows = filter_goal_segment(filter_run(list(read_jsonl(args.input)), args.run_id), args.goal_segment_id)
    exec_rows = list(read_jsonl(args.exec_input)) if args.exec_input else rows
    summary = execution_summary(exec_rows, args)
    summary["execution_monitor_input"] = args.exec_input or args.input
    output_rows = rows
    if not args.keep_output_after_terminal:
        output_rows = trim_at_time(output_rows, summary.get("monitor_terminal_time"))
    annotated = [{**row, **summary} for row in output_rows]
    write_jsonl(args.output, annotated)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "rows_written": len(annotated),
                "exec_rows": len(exec_rows),
                "output": args.output,
                **summary,
            },
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
    p.add_argument("--keep-after-collision", action="store_true", help="include rows after first entering collision clearance")
    p.add_argument(
        "--keep-output-after-terminal",
        action="store_true",
        help="write all planner rows instead of trimming benchmark rows at first arrival/collision",
    )
    return p


if __name__ == "__main__":
    monitor(parser().parse_args())

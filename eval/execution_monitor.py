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


def row_speed(row):
    speed = parse_float(row.get("speed"))
    if speed is not None:
        return speed
    vel, _source = parse_vector(row, (("odom_vel_w", "odom_velocity"), ("velocity_w", "velocity"), ("vel_w", "velocity")))
    if vel is None:
        return None
    return float(np.linalg.norm(vel))


def first_motion_event(rows, args):
    if not rows:
        return None
    threshold = float(args.motion_start_speed)
    duration = float(args.motion_start_duration)
    fallback = None
    chosen = None
    for idx, row in enumerate(rows):
        speed = row_speed(row)
        time = row_time(row)
        if speed is None or time is None or speed < threshold:
            continue
        if fallback is None:
            fallback = (idx, row, speed, time)
        if duration <= 0.0:
            chosen = (idx, row, speed, time)
            break
        valid = True
        covered = False
        for later in rows[idx:]:
            later_time = row_time(later)
            if later_time is None:
                continue
            if later_time - time > duration:
                covered = True
                break
            later_speed = row_speed(later)
            if later_speed is None or later_speed < threshold:
                valid = False
                break
        if valid and covered:
            chosen = (idx, row, speed, time)
            break
    if chosen is None:
        chosen = fallback
    if chosen is None:
        return None
    idx, row, speed, time = chosen
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
        "event": "motion_start",
        "index": int(idx),
        "time": time,
        "position": position,
        "goal_distance": row_goal_distance(row),
        "speed": speed,
        "threshold": threshold,
        "duration": duration,
    }


def nearest_row_at_time(rows, event_time):
    if event_time is None or not rows:
        return None
    best_row = None
    best_delta = None
    for row in rows:
        time = row_time(row)
        if time is None:
            continue
        delta = abs(time - event_time)
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_row = row
    return best_row


def path_distance_between_times(rows, start_time, end_time):
    if start_time is None or end_time is None or end_time < start_time:
        return None
    positions = []
    for row in rows:
        time = row_time(row)
        if time is None or time < start_time or time > end_time:
            continue
        pos, _source = parse_vector(
            row,
            (
                ("odom_pos_w", "executed_odom_to_gt_pointcloud"),
                ("position_w", "position_field_to_gt_pointcloud"),
                ("pos_w", "position_field_to_gt_pointcloud"),
                ("start_pos_w", "reference_or_planner_start_to_gt_pointcloud"),
            ),
        )
        if pos is not None:
            positions.append(pos.astype(np.float64))
    if len(positions) < 2:
        return 0.0 if positions else None
    distance = 0.0
    for start, end in zip(positions[:-1], positions[1:]):
        distance += float(np.linalg.norm(end - start))
    return distance


def execution_summary(rows, args):
    rows = active_rows(filter_goal_segment(filter_run(rows, args.run_id), args.goal_segment_id))
    untrimmed_count = len(rows)
    active_rows_untrimmed = list(rows)
    map_id = int(rows[0].get("map_id", args.map_id)) if rows else int(args.map_id)
    arrival_event = first_arrival_event(rows, args.success_distance)
    collision_event, clearance_source = first_collision_event(rows, args, map_id)
    motion_event = first_motion_event(rows, args)
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

    speeds = [row_speed(row) for row in rows]
    speeds = [speed for speed in speeds if speed is not None]
    motion_start_time = motion_event.get("time") if motion_event else None
    motion_start_goal_distance = motion_event.get("goal_distance") if motion_event else None
    motion_rows = [row for row in rows if motion_start_time is not None and (row_time(row) is None or row_time(row) >= motion_start_time)]
    motion_speeds = [row_speed(row) for row in motion_rows]
    motion_speeds = [speed for speed in motion_speeds if speed is not None]

    first_time = min(times) if times else None
    first_collision_time = collision_event.get("time") if collision_event else None
    first_arrival_time = arrival_event.get("time") if arrival_event else None
    time_to_collision = None
    if first_time is not None and first_collision_time is not None:
        time_to_collision = max(0.0, float(first_collision_time) - first_time)
    time_to_collision_from_motion = None
    if motion_start_time is not None and first_collision_time is not None:
        time_to_collision_from_motion = max(0.0, float(first_collision_time) - motion_start_time)
    time_to_arrival_from_motion = None
    if motion_start_time is not None and first_arrival_time is not None:
        time_to_arrival_from_motion = max(0.0, float(first_arrival_time) - motion_start_time)
    time_to_terminal_from_motion = None
    if motion_start_time is not None and terminal_time is not None:
        time_to_terminal_from_motion = max(0.0, float(terminal_time) - motion_start_time)

    collision_row = nearest_row_at_time(active_rows_untrimmed, first_collision_time)
    arrival_row = nearest_row_at_time(active_rows_untrimmed, first_arrival_time)
    first_collision_speed = row_speed(collision_row) if collision_row is not None else None
    first_arrival_speed = row_speed(arrival_row) if arrival_row is not None else None
    progress_at_collision = None
    if motion_start_goal_distance is not None and collision_event is not None and collision_event.get("goal_distance") is not None:
        progress_at_collision = float(motion_start_goal_distance - collision_event.get("goal_distance"))
    progress_at_terminal = None
    if motion_start_goal_distance is not None and goal_distance_final is not None:
        progress_at_terminal = float(motion_start_goal_distance - goal_distance_final)
    distance_before_collision = path_distance_between_times(active_rows_untrimmed, motion_start_time, first_collision_time)
    distance_before_terminal = path_distance_between_times(active_rows_untrimmed, motion_start_time, terminal_time)

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
        "mean_speed_motion_exec": float(np.mean(motion_speeds)) if motion_speeds else None,
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
        "first_arrival_time_exec": first_arrival_time,
        "first_arrival_position_w": event_position_list(arrival_event),
        "first_arrival_speed_exec": first_arrival_speed,
        "first_collision_time_exec": first_collision_time,
        "first_collision_position_w": event_position_list(collision_event),
        "first_collision_goal_distance": collision_event.get("goal_distance") if collision_event else None,
        "first_collision_clearance": collision_event.get("clearance") if collision_event else None,
        "first_collision_speed_exec": first_collision_speed,
        "time_to_collision_exec": time_to_collision,
        "motion_start_time_exec": motion_start_time,
        "motion_start_position_w": event_position_list(motion_event),
        "motion_start_goal_distance": motion_start_goal_distance,
        "motion_start_speed_exec": motion_event.get("speed") if motion_event else None,
        "motion_start_delay_exec": (max(0.0, float(motion_start_time) - min(times)) if motion_start_time is not None and times else None),
        "time_to_collision_from_motion_exec": time_to_collision_from_motion,
        "time_to_arrival_from_motion_exec": time_to_arrival_from_motion,
        "time_to_terminal_from_motion_exec": time_to_terminal_from_motion,
        "progress_at_collision_exec": progress_at_collision,
        "progress_at_terminal_exec": progress_at_terminal,
        "distance_travelled_before_collision_exec": distance_before_collision,
        "distance_travelled_before_terminal_exec": distance_before_terminal,
        "monitor_terminal_event": terminal_event.get("event") if terminal_event else None,
        "monitor_terminal_time": terminal_time,
        "monitor_trimmed_at_terminal": bool(terminal_event is not None),
        "monitor_collision_clearance": float(args.collision_clearance),
        "monitor_success_distance": float(args.success_distance),
        "monitor_max_time": float(args.max_time),
        "monitor_keep_after_arrival": bool(args.keep_after_arrival),
        "monitor_keep_after_collision": bool(args.keep_after_collision),
        "monitor_motion_start_speed": float(args.motion_start_speed),
        "monitor_motion_start_duration": float(args.motion_start_duration),
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
    p.add_argument("--motion-start-speed", type=float, default=0.5, help="m/s threshold for execution-start timing")
    p.add_argument("--motion-start-duration", type=float, default=0.2, help="seconds the speed threshold should be sustained")
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

import argparse
import json
import math
import os
from functools import lru_cache

import numpy as np
import torch
from scipy.spatial import cKDTree

from OARM.config import oarm_cfg
from OARM.policy.oarm_poly_solver import quintic_coefficients, sample_polynomial, sample_yaw_cubic, yaw_cubic_coefficients
from OARM.visibility.esdf_visibility import ESDFLineOfSight
from OARM.visibility.first_visible_time import reaction_margin_components
from OARM.utils.yopo_compat import ensure_yopo_path

ensure_yopo_path()
from config.config import cfg


def parse_vector(row, key, length=3):
    value = row.get(key)
    if value is None:
        raise ValueError(f"Missing required field: {key}")
    if isinstance(value, str):
        value = json.loads(value)
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (length,):
        raise ValueError(f"Field {key} must have shape ({length},), got {arr.shape}")
    return arr


def parse_float(row, key, default=None):
    value = row.get(key, default)
    if value is None or value == "":
        if default is None:
            raise ValueError(f"Missing required field: {key}")
        return float(default)
    return float(value)


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def goal_oriented_yaw(sampled_pos, sampled_vel, sampled_time, row, yaw0):
    if "goal_w" not in row:
        yaw = torch.full_like(sampled_time, float(yaw0))
        return yaw
    goal = parse_vector(row, "goal_w")
    pos_np = sampled_pos.detach().cpu().numpy()[0]
    vel_np = sampled_vel.detach().cpu().numpy()[0]
    time_np = sampled_time.detach().cpu().numpy()[0]
    yaw_values = []
    last_yaw = float(yaw0)
    last_time = float(time_np[0])
    for i, current_time in enumerate(time_np):
        dt = max(float(current_time - last_time), 1e-3)
        vel_dir = vel_np[i]
        goal_dir = goal - pos_np[i]
        vel_dir = vel_dir / (np.linalg.norm(vel_dir) + 1e-5)
        goal_dist = np.linalg.norm(goal_dir)
        goal_dir = goal_dir / (goal_dist + 1e-5)
        goal_yaw = np.arctan2(goal_dir[1], goal_dir[0])
        delta_yaw = wrap_to_pi(goal_yaw - last_yaw)
        weight = 6.0 * abs(delta_yaw) / np.pi
        dir_des = vel_dir + weight * goal_dir
        yaw_desired = np.arctan2(dir_des[1], dir_des[0]) if goal_dist > 0.5 else last_yaw
        yaw_diff = wrap_to_pi(yaw_desired - last_yaw)
        yaw_change = np.clip(yaw_diff, -0.5 * np.pi * dt, 0.5 * np.pi * dt)
        last_yaw = wrap_to_pi(last_yaw + yaw_change)
        last_time = float(current_time)
        yaw_values.append(last_yaw)
    return torch.tensor([yaw_values], dtype=sampled_time.dtype, device=sampled_time.device)


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


def build_selected_trajectory(row, eval_points, device, deployed_yaw_mode):
    start_pos = parse_vector(row, "start_pos_w")
    start_vel = parse_vector(row, "start_vel_w")
    start_acc = parse_vector(row, "start_acc_w")
    end_pos = parse_vector(row, "selected_end_pos_w")
    end_vel = parse_vector(row, "selected_end_vel_w")
    end_acc = parse_vector(row, "selected_end_acc_w")
    traj_time = parse_float(row, "selected_time")
    yaw0 = parse_float(row, "yaw0", 0.0)
    yaw_terminal = parse_float(row, "selected_yaw_terminal", yaw0)

    start_state = torch.from_numpy(np.stack([start_pos, start_vel, start_acc])[None]).to(device=device)
    end_state = torch.from_numpy(np.stack([end_pos, end_vel, end_acc])[None]).to(device=device)
    time_tensor = torch.tensor([traj_time], dtype=torch.float32, device=device)
    coeff = quintic_coefficients(start_state, end_state, time_tensor)
    sampled_pos, sampled_vel, _, _ = sample_polynomial(coeff, time_tensor, eval_points, include_zero=True)
    sampled_time = time_tensor[:, None] * torch.linspace(0.0, 1.0, eval_points, device=device)[None, :]

    if deployed_yaw_mode == "predicted":
        yaw0_t = torch.tensor([yaw0], dtype=torch.float32, device=device)
        yaw_rate0 = torch.zeros_like(yaw0_t)
        yaw_t = torch.tensor([yaw_terminal], dtype=torch.float32, device=device)
        yaw_coeff = yaw_cubic_coefficients(yaw0_t, yaw_rate0, yaw_t, time_tensor)
        yaw_ref, _ = sample_yaw_cubic(yaw_coeff, time_tensor, eval_points, include_zero=True)
    elif deployed_yaw_mode == "hold":
        yaw_ref = torch.full_like(sampled_time, float(yaw0))
    else:
        yaw_ref = goal_oriented_yaw(sampled_pos, sampled_vel, sampled_time, row, yaw0)
    return sampled_pos, sampled_time, yaw_ref


def build_candidate_trajectory(row, candidate, eval_points, device, deployed_yaw_mode):
    start_pos = parse_vector(row, "start_pos_w")
    start_vel = parse_vector(row, "start_vel_w")
    start_acc = parse_vector(row, "start_acc_w")
    end_pos = np.asarray(candidate.get("end_pos_w"), dtype=np.float32)
    end_vel = np.asarray(candidate.get("end_vel_w"), dtype=np.float32)
    end_acc = np.asarray(candidate.get("end_acc_w"), dtype=np.float32)
    if end_pos.shape != (3,) or end_vel.shape != (3,) or end_acc.shape != (3,):
        raise ValueError("Candidate geometry must contain end_pos_w/end_vel_w/end_acc_w with length 3")
    traj_time = float(candidate.get("time"))
    yaw0 = parse_float(row, "yaw0", 0.0)
    yaw_terminal = float(candidate.get("yaw_terminal", yaw0))

    start_state = torch.from_numpy(np.stack([start_pos, start_vel, start_acc])[None]).to(device=device)
    end_state = torch.from_numpy(np.stack([end_pos, end_vel, end_acc])[None]).to(device=device)
    time_tensor = torch.tensor([traj_time], dtype=torch.float32, device=device)
    coeff = quintic_coefficients(start_state, end_state, time_tensor)
    sampled_pos, sampled_vel, _, _ = sample_polynomial(coeff, time_tensor, eval_points, include_zero=True)
    sampled_time = time_tensor[:, None] * torch.linspace(0.0, 1.0, eval_points, device=device)[None, :]

    if deployed_yaw_mode == "predicted":
        yaw0_t = torch.tensor([yaw0], dtype=torch.float32, device=device)
        yaw_rate0 = torch.zeros_like(yaw0_t)
        yaw_t = torch.tensor([yaw_terminal], dtype=torch.float32, device=device)
        yaw_coeff = yaw_cubic_coefficients(yaw0_t, yaw_rate0, yaw_t, time_tensor)
        yaw_ref, _ = sample_yaw_cubic(yaw_coeff, time_tensor, eval_points, include_zero=True)
    elif deployed_yaw_mode == "hold":
        yaw_ref = torch.full_like(sampled_time, float(yaw0))
    else:
        yaw_ref = goal_oriented_yaw(sampled_pos, sampled_vel, sampled_time, row, yaw0)
    return sampled_pos, sampled_time, yaw_ref


def candidate_reaction_margin_gt(row, candidate, line_of_sight, args, device, map_id):
    sampled_pos, sampled_time, yaw_ref = build_candidate_trajectory(
        row, candidate, args.eval_points, device, args.deployed_yaw_mode
    )
    min_clearance = trajectory_min_clearance(sampled_pos, map_id, args.dataset_dir)
    result = {
        "min_clearance_gt": min_clearance,
        "collision_gt": bool(min_clearance < args.collision_clearance),
    }
    risk_points = select_gt_risk_points(
        sampled_pos,
        map_id,
        args.dataset_dir,
        args.risk_radius,
        args.max_risk_points,
    )
    if risk_points is None:
        result.update({
            "gt_annotation_status": "no_nearby_gt_risk_points",
            "gt_risk_point_count": 0,
            "reaction_margin_gt": None,
            "reaction_window_gt": None,
            "selected_rmvr_gt": None,
            "valid_reaction_margin_gt": False,
            "rm_event_valid_gt": False,
            "rm_right_censored_gt": False,
            "rm_no_entry_gt": True,
            "risk_visible_at_t0_gt": False,
            "critical_risk_point_id": None,
            "critical_risk_weight": None,
            "hidden_risk_gt": None,
        })
        return result

    risk_points_t = torch.tensor(risk_points, dtype=torch.float32, device=device).unsqueeze(0)
    map_tensor = torch.tensor([map_id], dtype=torch.long, device=device)
    visibility_mask = line_of_sight(sampled_pos, risk_points_t, map_tensor) if args.use_esdf_los else None
    components = reaction_margin_components(
        sampled_pos,
        sampled_time,
        yaw_ref,
        risk_points_t,
        horizon_fov_rad=math.radians(args.horizon_fov_deg),
        vertical_fov_rad=math.radians(args.vertical_fov_deg),
        reaction_time=args.reaction_time,
        visibility_mask=visibility_mask,
        max_arrival_distance_m=args.arrival_radius,
    )
    first_vis = components["first_visible_time"]
    arrival = components["first_entry_time"]
    margin = components["reaction_margin_points"]
    window = components["observation_lead_time"]
    visible_before_entry = components["visible_before_entry"].bool()
    visible_at_t0 = torch.isfinite(first_vis) & (first_vis <= args.hidden_risk_eps)
    valid_mask = components["arrival_valid"].bool() & torch.isfinite(margin)
    if bool(valid_mask.any()):
        masked_margin = torch.where(valid_mask, margin, torch.full_like(margin, torch.inf))
        margin_flat = masked_margin.reshape(-1)
        first_flat = first_vis.reshape(-1)
        arrival_flat = arrival.reshape(-1)
        window_flat = window.reshape(-1)
        event_flat = visible_before_entry.reshape(-1)
        visible_t0_flat = visible_at_t0.reshape(-1)
        crit_idx = int(torch.argmin(margin_flat).detach().cpu())
        crit_first = first_flat[crit_idx]
        crit_arrival = arrival_flat[crit_idx]
        reaction_margin_gt = float(margin_flat[crit_idx].detach().cpu())
        reaction_window_gt = float(window_flat[crit_idx].detach().cpu())
        event_valid = bool(event_flat[crit_idx].detach().cpu())
        result.update({
            "gt_annotation_status": "ok",
            "gt_risk_point_count": int(len(risk_points)),
            "critical_first_visible_time_gt": None if bool(torch.isinf(crit_first)) else float(crit_first.detach().cpu()),
            "critical_arrival_time_gt": None if bool(torch.isinf(crit_arrival)) else float(crit_arrival.detach().cpu()),
            "reaction_margin_gt": reaction_margin_gt,
            "reaction_window_gt": reaction_window_gt,
            "selected_rmvr_gt": float(reaction_margin_gt < 0.0),
            "valid_reaction_margin_gt": True,
            "rm_event_valid_gt": event_valid,
            "rm_right_censored_gt": bool(not event_valid),
            "rm_no_entry_gt": False,
            "risk_visible_at_t0_gt": bool(visible_t0_flat[crit_idx].detach().cpu()),
            "critical_risk_point_id": crit_idx,
            "critical_risk_weight": 1.0,
            "hidden_risk_gt": bool(torch.isinf(crit_first) or crit_first > args.hidden_risk_eps),
        })
    else:
        result.update({
            "gt_annotation_status": "censored",
            "gt_risk_point_count": int(len(risk_points)),
            "reaction_margin_gt": None,
            "reaction_window_gt": None,
            "selected_rmvr_gt": None,
            "valid_reaction_margin_gt": False,
            "rm_event_valid_gt": False,
            "rm_right_censored_gt": False,
            "rm_no_entry_gt": True,
            "risk_visible_at_t0_gt": False,
            "critical_risk_point_id": None,
            "critical_risk_weight": None,
            "hidden_risk_gt": None,
        })
    return result


def select_gt_risk_points(sampled_pos, map_id, dataset_dir, risk_radius, max_points):
    points, tree = load_pointcloud(map_id, dataset_dir)
    traj_np = sampled_pos.detach().cpu().numpy().reshape(-1, 3)
    index_set = set()
    for point in traj_np:
        index_set.update(tree.query_ball_point(point, risk_radius))
    if not index_set:
        return None
    indices = np.fromiter(index_set, dtype=np.int64)
    risk_points = points[indices]
    distances, _ = cKDTree(traj_np).query(risk_points, k=1)
    order = np.argsort(distances)[:max_points]
    return risk_points[order]


def trajectory_min_clearance(sampled_pos, map_id, dataset_dir):
    _points, tree = load_pointcloud(map_id, dataset_dir)
    traj_np = sampled_pos.detach().cpu().numpy().reshape(-1, 3)
    distances, _ = tree.query(traj_np, k=1)
    return float(np.min(distances))


def annotate_row(row, line_of_sight, args, device):
    if "selected_end_pos_w" not in row:
        row["gt_annotation_status"] = "missing_geometry_fields"
        return row
    map_id = int(row.get("map_id", args.map_id))
    sampled_pos, sampled_time, yaw_ref = build_selected_trajectory(row, args.eval_points, device, args.deployed_yaw_mode)
    min_clearance = trajectory_min_clearance(sampled_pos, map_id, args.dataset_dir)
    selected_collision = bool(min_clearance < args.collision_clearance)
    row["selected_traj_min_clearance_gt"] = min_clearance
    row["selected_traj_collision_gt"] = selected_collision
    row["min_clearance_gt"] = min_clearance
    row["collision_gt"] = selected_collision
    risk_points = select_gt_risk_points(
        sampled_pos,
        map_id,
        args.dataset_dir,
        args.risk_radius,
        args.max_risk_points,
    )
    if risk_points is None:
        row.update(
            {
                "gt_annotation_status": "no_nearby_gt_risk_points",
                "gt_risk_point_count": 0,
                "first_visible_time_gt": None,
                "arrival_time_gt": None,
                "arrival_time_to_risk_gt": None,
                "reaction_margin_gt": None,
                "selected_rmvr_gt": None,
"valid_reaction_margin_gt": False,
"hidden_risk_gt": None,
                "uses_privileged_gt_annotation": True,
                "uses_privileged_online": False,
                "deployed_yaw_mode": args.deployed_yaw_mode,
            }
        )
        return row

    risk_points_t = torch.tensor(risk_points, dtype=torch.float32, device=device).unsqueeze(0)
    map_tensor = torch.tensor([map_id], dtype=torch.long, device=device)
    visibility_mask = line_of_sight(sampled_pos, risk_points_t, map_tensor) if args.use_esdf_los else None
    components = reaction_margin_components(
        sampled_pos,
        sampled_time,
        yaw_ref,
        risk_points_t,
        horizon_fov_rad=math.radians(args.horizon_fov_deg),
        vertical_fov_rad=math.radians(args.vertical_fov_deg),
        reaction_time=args.reaction_time,
        visibility_mask=visibility_mask,
        max_arrival_distance_m=args.arrival_radius,
    )
    first_vis = components['first_visible_time']
    arrival = components['first_entry_time']
    margin = components['reaction_margin_points']
    valid_mask = components['arrival_valid'].bool() & torch.isfinite(margin)
    finite_first = torch.where(torch.isinf(first_vis), torch.full_like(first_vis, float('nan')), first_vis)
    first_visible_time_gt = tensor_nanmin(finite_first)

    if bool(valid_mask.any()):
        masked_margin = torch.where(valid_mask, margin, torch.full_like(margin, torch.inf))
        margin_flat = masked_margin.reshape(-1)
        first_flat = first_vis.reshape(-1)
        arrival_flat = arrival.reshape(-1)
        crit_idx = int(torch.argmin(margin_flat).detach().cpu())
        crit_first = first_flat[crit_idx]
        crit_arrival = arrival_flat[crit_idx]
        reaction_margin_gt = float(margin_flat[crit_idx].detach().cpu())
        critical_first_visible_time_gt = None if bool(torch.isinf(crit_first)) else float(crit_first.detach().cpu())
        critical_arrival_time_gt = None if bool(torch.isinf(crit_arrival)) else float(crit_arrival.detach().cpu())
        hidden_risk_gt = bool(torch.isinf(crit_first) or crit_first > args.hidden_risk_eps)
        selected_rmvr_gt = float(reaction_margin_gt < 0.0)
        valid_reaction_margin_gt = True
        reaction_margin_censored_gt = False
    else:
        reaction_margin_gt = None
        critical_first_visible_time_gt = None
        critical_arrival_time_gt = None
        hidden_risk_gt = None
        selected_rmvr_gt = None
        valid_reaction_margin_gt = False
        reaction_margin_censored_gt = True

    row.update(
        {
            "gt_annotation_status": "ok",
            "first_visible_time_gt": first_visible_time_gt,
            "critical_first_visible_time_gt": critical_first_visible_time_gt,
            "critical_arrival_time_gt": critical_arrival_time_gt,
            "arrival_time_gt": critical_arrival_time_gt,
            "arrival_time_to_risk_gt": critical_arrival_time_gt,
            "reaction_margin_gt": reaction_margin_gt,
            "selected_rmvr_gt": selected_rmvr_gt,
            "valid_reaction_margin_gt": valid_reaction_margin_gt,
            "reaction_margin_censored_gt": reaction_margin_censored_gt,
            "hidden_risk_gt": hidden_risk_gt,
            "uses_privileged_gt_annotation": True,
            "uses_privileged_online": False,
            "deployed_yaw_mode": args.deployed_yaw_mode,
        }
    )
    if args.annotate_candidates and row.get("candidates"):
        annotated = []
        selected_id = int(row.get("selected_id", -1))
        for candidate in row["candidates"]:
            cand = dict(candidate)
            try:
                cand.update(candidate_reaction_margin_gt(row, cand, line_of_sight, args, device, map_id))
            except Exception as exc:
                cand["gt_annotation_status"] = f"error:{type(exc).__name__}"
                cand["gt_annotation_error"] = str(exc)
            cand["is_selected"] = int(cand.get("id", -2)) == selected_id
            annotated.append(cand)
        row["candidates"] = annotated
        valid_candidates = [c for c in annotated if c.get("valid_reaction_margin_gt") and c.get("reaction_margin_gt") is not None]
        if valid_candidates:
            oracle = max(valid_candidates, key=lambda c: c["reaction_margin_gt"])
            row["candidate_oracle_id"] = int(oracle.get("id", -1))
            row["candidate_oracle_reaction_margin_gt"] = float(oracle["reaction_margin_gt"])
            row["candidate_oracle_selection_score"] = float(oracle.get("selection_score", 0.0))
            row["safe_candidate_available_gt"] = bool(oracle["reaction_margin_gt"] > 0.0)
            if reaction_margin_gt is not None:
                row["candidate_oracle_margin_gap_gt"] = float(oracle["reaction_margin_gt"] - reaction_margin_gt)
    return row


def tensor_nanmin(value):
    finite = value[torch.isfinite(value)]
    if finite.numel() == 0:
        return None
    return float(finite.amin().detach().cpu())


def annotate(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    line_of_sight = ESDFLineOfSight(device=device) if args.use_esdf_los else None
    rows = [annotate_row(row, line_of_sight, args, device) for row in read_jsonl(args.input)]
    write_jsonl(args.output, rows)
    ok = sum(1 for row in rows if row.get("gt_annotation_status") == "ok")
    selected_rmvr = [row.get("selected_rmvr_gt") for row in rows if row.get("selected_rmvr_gt") is not None]
    hidden_rows = [row for row in rows if row.get("hidden_risk_gt") is True and row.get("selected_rmvr_gt") is not None]
    summary = {
        "rows": len(rows),
        "annotated_rows": ok,
"gt_rmvr_valid_count": len(selected_rmvr),
"gt_rmvr_coverage": len(selected_rmvr) / max(len(rows), 1),
"hidden_risk_sample_count": len(hidden_rows),
"selected_rmvr_gt": (sum(selected_rmvr) / len(selected_rmvr)) if selected_rmvr else None,
"selected_rmvr_gt_hidden": (sum(row["selected_rmvr_gt"] for row in hidden_rows) / len(hidden_rows)) if hidden_rows else None,
        "output": args.output,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="ROS benchmark JSONL from test_oarm_ros.py --log-jsonl")
    p.add_argument("--output", required=True, help="annotated JSONL output")
    p.add_argument("--dataset-dir", default="dataset", help="directory containing pointcloud-*.ply")
    p.add_argument("--map-id", type=int, default=0, help="fallback map id when rows do not contain map_id")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--eval-points", type=int, default=40)
    p.add_argument("--risk-radius", type=float, default=2.0)
    p.add_argument("--max-risk-points", type=int, default=64)
    p.add_argument("--reaction-time", type=float, default=oarm_cfg.reaction_time)
    p.add_argument("--arrival-radius", type=float, default=oarm_cfg.risk_arrival_radius_m)
    p.add_argument("--collision-clearance", type=float, default=oarm_cfg.visibility_clearance_m)
    p.add_argument("--hidden-risk-eps", type=float, default=1e-3)
    p.add_argument("--horizon-fov-deg", type=float, default=cfg["horizon_camera_fov"])
    p.add_argument("--vertical-fov-deg", type=float, default=cfg["vertical_camera_fov"])
    p.add_argument("--use-esdf-los", action="store_true", help="require GT ESDF line-of-sight for first-visible time")
    p.add_argument("--annotate-candidates", action="store_true", help="also annotate per-candidate geometry stored by --log-candidate-table")
    p.add_argument(
        "--deployed-yaw-mode",
        choices=["goal", "hold", "predicted"],
        default="goal",
        help="yaw model used for GT visibility annotation; goal matches the current ROS controller best",
    )
    return p


if __name__ == "__main__":
    annotate(parser().parse_args())

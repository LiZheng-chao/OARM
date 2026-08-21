from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class BrakeTrajectoryDiagnostics:
    feasible: bool
    duration: float
    stop_distance: float
    peak_accel: float
    peak_jerk: float
    peak_thrust_accel: float
    peak_tilt_deg: float
    max_accel: float
    max_jerk: float
    max_thrust_accel: float
    max_tilt_deg: float
    iterations: int
    fallback_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class BrakeCommand:
    end_pos: np.ndarray
    end_vel: np.ndarray
    end_acc: np.ndarray
    duration: float
    diagnostics: BrakeTrajectoryDiagnostics


def _as_vec3(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(3)


def _quintic_coefficients(pos0, vel0, acc0, pos1, vel1, acc1, duration: float) -> np.ndarray:
    state = np.stack([pos0, vel0, acc0, pos1, vel1, acc1], axis=0).astype(np.float64)
    t = float(max(duration, 1e-3))
    coef_inv = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0.5, 0, 0, 0],
            [-10 / t ** 3, -6 / t ** 2, -3 / (2 * t), 10 / t ** 3, -4 / t ** 2, 1 / (2 * t)],
            [15 / t ** 4, 8 / t ** 3, 3 / (2 * t ** 2), -15 / t ** 4, 7 / t ** 3, -1 / t ** 2],
            [-6 / t ** 5, -3 / t ** 4, -1 / (2 * t ** 3), 6 / t ** 5, -3 / t ** 4, 1 / (2 * t ** 3)],
        ],
        dtype=np.float64,
    )
    return coef_inv @ state


def sample_quintic_trajectory(start_pos, start_vel, start_acc, end_pos, end_vel, end_acc, duration: float, sample_count: int = 81):
    coeff = _quintic_coefficients(
        _as_vec3(start_pos),
        _as_vec3(start_vel),
        _as_vec3(start_acc),
        _as_vec3(end_pos),
        _as_vec3(end_vel),
        _as_vec3(end_acc),
        duration,
    )
    t = np.linspace(0.0, float(duration), int(max(sample_count, 2)), dtype=np.float64)
    pos = coeff[0] + coeff[1] * t[:, None] + coeff[2] * t[:, None] ** 2 + coeff[3] * t[:, None] ** 3 + coeff[4] * t[:, None] ** 4 + coeff[5] * t[:, None] ** 5
    vel = coeff[1] + 2 * coeff[2] * t[:, None] + 3 * coeff[3] * t[:, None] ** 2 + 4 * coeff[4] * t[:, None] ** 3 + 5 * coeff[5] * t[:, None] ** 4
    acc = 2 * coeff[2] + 6 * coeff[3] * t[:, None] + 12 * coeff[4] * t[:, None] ** 2 + 20 * coeff[5] * t[:, None] ** 3
    jerk = 6 * coeff[3] + 24 * coeff[4] * t[:, None] + 60 * coeff[5] * t[:, None] ** 2
    return pos.astype(np.float32), vel.astype(np.float32), acc.astype(np.float32), jerk.astype(np.float32)


def evaluate_brake_trajectory(
    start_pos,
    start_vel,
    start_acc,
    end_pos,
    end_vel,
    end_acc,
    duration: float,
    sample_count: int = 81,
    max_accel: float = 6.0,
    max_jerk: float = 30.0,
    max_thrust_accel: float = 18.0,
    max_tilt_deg: float = 50.0,
    gravity: float = 9.81,
) -> Dict[str, float]:
    _, _, acc, jerk = sample_quintic_trajectory(start_pos, start_vel, start_acc, end_pos, end_vel, end_acc, duration, sample_count)
    acc_norm = np.linalg.norm(acc, axis=1)
    jerk_norm = np.linalg.norm(jerk, axis=1)
    thrust_vec = acc + np.array([0.0, 0.0, float(gravity)], dtype=np.float32)[None, :]
    thrust_norm = np.linalg.norm(thrust_vec, axis=1)
    lateral = np.linalg.norm(thrust_vec[:, :2], axis=1)
    vertical = np.maximum(thrust_vec[:, 2], 1e-6)
    tilt = np.degrees(np.arctan2(lateral, vertical))
    peak_accel = float(np.max(acc_norm))
    peak_jerk = float(np.max(jerk_norm))
    peak_thrust = float(np.max(thrust_norm))
    peak_tilt = float(np.max(tilt))
    feasible = (
        peak_accel <= float(max_accel) + 1e-6
        and peak_jerk <= float(max_jerk) + 1e-6
        and peak_thrust <= float(max_thrust_accel) + 1e-6
        and peak_tilt <= float(max_tilt_deg) + 1e-6
    )
    return {
        "feasible": bool(feasible),
        "peak_accel": peak_accel,
        "peak_jerk": peak_jerk,
        "peak_thrust_accel": peak_thrust,
        "peak_tilt_deg": peak_tilt,
    }


def constrained_brake_command(
    start_pos,
    start_vel,
    start_acc=None,
    goal=None,
    min_time: float = 0.45,
    brake_accel: float = 3.0,
    max_time: float = 5.0,
    max_accel: float = 6.0,
    max_jerk: float = 30.0,
    max_thrust_accel: float = 18.0,
    max_tilt_deg: float = 50.0,
    sample_count: int = 81,
    time_growth: float = 1.25,
    target_z=None,
    z_rate: float = 0.8,
    min_command_z: float = -np.inf,
    max_command_z: float = np.inf,
    gravity: float = 9.81,
) -> BrakeCommand:
    start_pos = _as_vec3(start_pos)
    start_vel = _as_vec3(start_vel)
    start_acc = np.zeros(3, dtype=np.float32) if start_acc is None else _as_vec3(start_acc)
    goal = start_pos if goal is None else _as_vec3(goal)
    speed = float(np.linalg.norm(start_vel))
    brake_accel = max(float(brake_accel), 1e-3)
    stop_distance = speed * speed / (2.0 * brake_accel) if speed > 1e-5 else 0.0
    direction = start_vel / speed if speed > 1e-5 else np.zeros(3, dtype=np.float32)
    end_pos = start_pos + direction * stop_distance

    z_goal = goal[2] if target_z is None else float(target_z)
    base_time = speed / brake_accel if speed > 1e-5 else 0.0
    duration = float(np.clip(max(float(min_time), base_time), 0.1, float(max_time)))
    max_z_step = max(0.0, float(z_rate)) * duration
    z_delta = float(np.clip(z_goal - start_pos[2], -max_z_step, max_z_step))
    end_pos[2] = float(np.clip(start_pos[2] + z_delta, float(min_command_z), float(max_command_z)))
    end_vel = np.zeros(3, dtype=np.float32)
    end_acc = np.zeros(3, dtype=np.float32)

    best_eval = None
    best_duration = duration
    iterations = 0
    while True:
        iterations += 1
        metrics = evaluate_brake_trajectory(
            start_pos,
            start_vel,
            start_acc,
            end_pos,
            end_vel,
            end_acc,
            best_duration,
            sample_count=sample_count,
            max_accel=max_accel,
            max_jerk=max_jerk,
            max_thrust_accel=max_thrust_accel,
            max_tilt_deg=max_tilt_deg,
            gravity=gravity,
        )
        best_eval = metrics
        if metrics["feasible"] or best_duration >= float(max_time) - 1e-9:
            break
        best_duration = min(float(max_time), best_duration * max(float(time_growth), 1.01))
        max_z_step = max(0.0, float(z_rate)) * best_duration
        z_delta = float(np.clip(z_goal - start_pos[2], -max_z_step, max_z_step))
        end_pos[2] = float(np.clip(start_pos[2] + z_delta, float(min_command_z), float(max_command_z)))

    fallback_reason = None if best_eval["feasible"] else "constraints_exceeded_at_max_time"
    diagnostics = BrakeTrajectoryDiagnostics(
        feasible=bool(best_eval["feasible"]),
        duration=float(best_duration),
        stop_distance=float(stop_distance),
        peak_accel=float(best_eval["peak_accel"]),
        peak_jerk=float(best_eval["peak_jerk"]),
        peak_thrust_accel=float(best_eval["peak_thrust_accel"]),
        peak_tilt_deg=float(best_eval["peak_tilt_deg"]),
        max_accel=float(max_accel),
        max_jerk=float(max_jerk),
        max_thrust_accel=float(max_thrust_accel),
        max_tilt_deg=float(max_tilt_deg),
        iterations=int(iterations),
        fallback_reason=fallback_reason,
    )
    return BrakeCommand(end_pos.astype(np.float32), end_vel, end_acc, float(best_duration), diagnostics)


def deterministic_brake_endpoint(
    start_pos,
    start_vel,
    goal,
    selected_time: float,
    distance_scale: float = 0.0,
    retreat_distance: float = 0.0,
    target_z=None,
    z_rate: float = 0.8,
    min_command_z: float = -np.inf,
    max_command_z: float = np.inf,
):
    """Backward-compatible deterministic stop-reference endpoint.

    New online BRAKE code should use constrained_brake_command(); this helper
    remains for callers that only need an endpoint tuple.
    """
    command = constrained_brake_command(
        start_pos,
        start_vel,
        start_acc=np.zeros(3, dtype=np.float32),
        goal=goal,
        min_time=selected_time,
        max_time=selected_time,
        brake_accel=max(float(np.linalg.norm(start_vel)) / max(float(selected_time), 1e-3), 1e-3),
        target_z=target_z,
        z_rate=z_rate,
        min_command_z=min_command_z,
        max_command_z=max_command_z,
    )
    if float(distance_scale) != 0.0 or float(retreat_distance) != 0.0:
        # Preserve legacy optional offsets for non-ROS callers.
        start_pos = _as_vec3(start_pos)
        start_vel = _as_vec3(start_vel)
        goal = _as_vec3(goal)
        end_pos = start_pos + start_vel * float(selected_time) * float(distance_scale)
        goal_vec = goal - start_pos
        goal_vec[2] = 0.0
        goal_norm = float(np.linalg.norm(goal_vec))
        if float(retreat_distance) > 0.0 and goal_norm > 1e-3:
            end_pos -= (goal_vec / goal_norm) * float(retreat_distance)
        end_pos[2] = command.end_pos[2]
        return end_pos.astype(np.float32), command.end_vel, command.end_acc
    return command.end_pos, command.end_vel, command.end_acc

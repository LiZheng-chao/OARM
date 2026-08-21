import numpy as np


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
    """Return a deterministic terminal state for an emergency brake command.

    The endpoint is close to the current reference state, optionally retreats
    away from the goal in XY, and always commands zero terminal velocity and
    acceleration so the published polynomial is an actual brake trajectory.
    """
    start_pos = np.asarray(start_pos, dtype=np.float32).reshape(3)
    start_vel = np.asarray(start_vel, dtype=np.float32).reshape(3)
    goal = np.asarray(goal, dtype=np.float32).reshape(3)
    selected_time = float(np.clip(selected_time, 0.1, 10.0))

    emergency_target = start_pos + start_vel * selected_time * float(distance_scale)
    goal_vec = goal - start_pos
    goal_vec[2] = 0.0
    goal_norm = float(np.linalg.norm(goal_vec))
    if float(retreat_distance) > 0.0 and goal_norm > 1e-3:
        emergency_target -= (goal_vec / goal_norm) * float(retreat_distance)

    z_goal = goal[2] if target_z is None else float(target_z)
    max_z_step = max(0.0, float(z_rate)) * selected_time
    z_delta = float(np.clip(z_goal - start_pos[2], -max_z_step, max_z_step))
    emergency_target[2] = float(np.clip(start_pos[2] + z_delta, float(min_command_z), float(max_command_z)))

    end_vel = np.zeros(3, dtype=np.float32)
    end_acc = np.zeros(3, dtype=np.float32)
    return emergency_target.astype(np.float32), end_vel, end_acc

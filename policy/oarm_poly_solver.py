import torch


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return (angle + torch.pi) % (2.0 * torch.pi) - torch.pi


def quintic_coefficients(start_state: torch.Tensor, end_state: torch.Tensor, traj_time: torch.Tensor) -> torch.Tensor:
    """Return quintic coefficients for batched 3D PVA boundary conditions.

    Args:
        start_state: [N, 3, 3] with rows [pos, vel, acc].
        end_state: [N, 3, 3] with rows [pos, vel, acc].
        traj_time: [N].
    Returns:
        coeff: [N, 3, 6], axis-major polynomial coefficients.
    """

    t = traj_time.clamp(min=1e-3).view(-1, 1)
    p0, v0, a0 = start_state[:, 0], start_state[:, 1], start_state[:, 2]
    p1, v1, a1 = end_state[:, 0], end_state[:, 1], end_state[:, 2]

    c0 = p0
    c1 = v0
    c2 = 0.5 * a0
    c3 = (
        10 * (p1 - p0) / t ** 3
        - (6 * v0 + 4 * v1) / t ** 2
        - (1.5 * a0 - 0.5 * a1) / t
    )
    c4 = (
        -15 * (p1 - p0) / t ** 4
        + (8 * v0 + 7 * v1) / t ** 3
        + (1.5 * a0 - a1) / t ** 2
    )
    c5 = (
        6 * (p1 - p0) / t ** 5
        - (3 * v0 + 3 * v1) / t ** 4
        - (0.5 * a0 - 0.5 * a1) / t ** 3
    )
    return torch.stack([c0, c1, c2, c3, c4, c5], dim=-1)


def sample_polynomial(coeff: torch.Tensor, traj_time: torch.Tensor, eval_points: int = 30, include_zero: bool = False):
    """Sample position, velocity, acceleration and jerk at normalized time bins."""

    n = coeff.shape[0]
    if include_zero:
        tau = torch.linspace(0.0, 1.0, eval_points, device=coeff.device, dtype=coeff.dtype)
    else:
        tau = torch.linspace(1.0 / eval_points, 1.0, eval_points, device=coeff.device, dtype=coeff.dtype)
    t = traj_time.view(n, 1) * tau.view(1, eval_points)

    powers_p = torch.stack([torch.ones_like(t), t, t ** 2, t ** 3, t ** 4, t ** 5], dim=-1)
    powers_v = torch.stack([torch.ones_like(t), 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4], dim=-1)
    powers_a = torch.stack([2 * torch.ones_like(t), 6 * t, 12 * t ** 2, 20 * t ** 3], dim=-1)
    powers_j = torch.stack([6 * torch.ones_like(t), 24 * t, 60 * t ** 2], dim=-1)

    pos = torch.sum(coeff[:, None, :, :] * powers_p[:, :, None, :], dim=-1)
    vel = torch.sum(coeff[:, None, :, 1:] * powers_v[:, :, None, :], dim=-1)
    acc = torch.sum(coeff[:, None, :, 2:] * powers_a[:, :, None, :], dim=-1)
    jerk = torch.sum(coeff[:, None, :, 3:] * powers_j[:, :, None, :], dim=-1)
    return pos, vel, acc, jerk


def yaw_cubic_coefficients(yaw0: torch.Tensor, yaw_rate0: torch.Tensor, yaw1: torch.Tensor, traj_time: torch.Tensor):
    """Cubic yaw polynomial with terminal yaw rate fixed to zero."""

    t = traj_time.clamp(min=1e-3)
    yaw1 = yaw0 + wrap_to_pi(yaw1 - yaw0)
    c0 = yaw0
    c1 = yaw_rate0
    c2 = (3 * (yaw1 - yaw0) - (2 * yaw_rate0) * t) / (t ** 2)
    c3 = (2 * (yaw0 - yaw1) + yaw_rate0 * t) / (t ** 3)
    return torch.stack([c0, c1, c2, c3], dim=-1)


def sample_yaw_cubic(coeff: torch.Tensor, traj_time: torch.Tensor, eval_points: int = 30, include_zero: bool = False):
    n = coeff.shape[0]
    if include_zero:
        tau = torch.linspace(0.0, 1.0, eval_points, device=coeff.device, dtype=coeff.dtype)
    else:
        tau = torch.linspace(1.0 / eval_points, 1.0, eval_points, device=coeff.device, dtype=coeff.dtype)
    t = traj_time.view(n, 1) * tau.view(1, eval_points)
    yaw = coeff[:, 0:1] + coeff[:, 1:2] * t + coeff[:, 2:3] * t ** 2 + coeff[:, 3:4] * t ** 3
    yaw_rate = coeff[:, 1:2] + 2.0 * coeff[:, 2:3] * t + 3.0 * coeff[:, 3:4] * t ** 2
    return yaw, yaw_rate

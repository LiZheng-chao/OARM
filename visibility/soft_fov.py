import torch


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return (angle + torch.pi) % (2.0 * torch.pi) - torch.pi


def bearing_to_point(observer_pos: torch.Tensor, points: torch.Tensor):
    """Bearing from sampled observer positions to risk points.

    Args:
        observer_pos: [N, T, 3].
        points: [N, Q, 3].
    Returns:
        yaw, pitch, distance with shape [N, T, Q].
    """

    rel = points[:, None, :, :] - observer_pos[:, :, None, :]
    yaw = torch.atan2(rel[..., 1], rel[..., 0])
    dist_xy = rel[..., 0:2].norm(dim=-1).clamp(min=1e-6)
    pitch = torch.atan2(rel[..., 2], dist_xy)
    return yaw, pitch, rel.norm(dim=-1)


def soft_fov_score(
    observer_pos: torch.Tensor,
    yaw_ref: torch.Tensor,
    points: torch.Tensor,
    horizon_fov_rad: float,
    vertical_fov_rad: float,
    pitch_ref: torch.Tensor = None,
    softness: float = 0.08,
) -> torch.Tensor:
    """Differentiable visibility proxy for point-in-FOV tests.

    Args:
        observer_pos: [N, T, 3].
        yaw_ref: [N, T] camera/yaw reference.
        points: [N, Q, 3] risk points in the same frame.
    Returns:
        score: [N, T, Q], near 1 inside FOV and near 0 outside.
    """

    if pitch_ref is None:
        pitch_ref = torch.zeros_like(yaw_ref)
    yaw_to_point, pitch_to_point, _ = bearing_to_point(observer_pos, points)
    yaw_err = wrap_to_pi(yaw_to_point - yaw_ref[..., None]).abs()
    pitch_err = (pitch_to_point - pitch_ref[..., None]).abs()
    yaw_score = torch.sigmoid((0.5 * horizon_fov_rad - yaw_err) / softness)
    pitch_score = torch.sigmoid((0.5 * vertical_fov_rad - pitch_err) / softness)
    return yaw_score * pitch_score


def hard_fov_mask(
    observer_pos: torch.Tensor,
    yaw_ref: torch.Tensor,
    points: torch.Tensor,
    horizon_fov_rad: float,
    vertical_fov_rad: float,
    pitch_ref: torch.Tensor = None,
) -> torch.Tensor:
    if pitch_ref is None:
        pitch_ref = torch.zeros_like(yaw_ref)
    yaw_to_point, pitch_to_point, _ = bearing_to_point(observer_pos, points)
    yaw_err = wrap_to_pi(yaw_to_point - yaw_ref[..., None]).abs()
    pitch_err = (pitch_to_point - pitch_ref[..., None]).abs()
    return (yaw_err <= 0.5 * horizon_fov_rad) & (pitch_err <= 0.5 * vertical_fov_rad)

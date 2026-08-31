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


def bearing_to_point_camera_frame(observer_pos: torch.Tensor, points: torch.Tensor, camera_rot_w: torch.Tensor):
    rel_w = points[:, None, :, :] - observer_pos[:, :, None, :]
    rel_c = torch.matmul(camera_rot_w.transpose(-1, -2)[:, :, None, :, :], rel_w[..., None]).squeeze(-1)
    x = rel_c[..., 0]
    y = rel_c[..., 1]
    z = rel_c[..., 2]
    dist_xy = torch.sqrt(x.square() + y.square()).clamp(min=1e-6)
    yaw = torch.atan2(y, x.clamp(min=1e-6))
    pitch = torch.atan2(z, dist_xy)
    return yaw, pitch, rel_c.norm(dim=-1), x > 0.0

def hard_fov_mask(observer_pos, yaw_ref, points, horizon_fov_rad, vertical_fov_rad, pitch_ref=None, camera_rot_w=None, max_range_m=None):
    if camera_rot_w is not None:
        yaw_to_point, pitch_to_point, distance, in_front = bearing_to_point_camera_frame(observer_pos, points, camera_rot_w)
        yaw_err = yaw_to_point.abs()
        pitch_err = pitch_to_point.abs()
        mask = in_front & (yaw_err <= 0.5 * horizon_fov_rad) & (pitch_err <= 0.5 * vertical_fov_rad)
        if max_range_m is not None and float(max_range_m) > 0.0:
            mask = mask & (distance <= float(max_range_m))
        return mask
    if pitch_ref is None:
        pitch_ref = torch.zeros_like(yaw_ref)
    yaw_to_point, pitch_to_point, distance = bearing_to_point(observer_pos, points)
    yaw_err = wrap_to_pi(yaw_to_point - yaw_ref[..., None]).abs()
    pitch_err = (pitch_to_point - pitch_ref[..., None]).abs()
    mask = (yaw_err <= 0.5 * horizon_fov_rad) & (pitch_err <= 0.5 * vertical_fov_rad)
    if max_range_m is not None and float(max_range_m) > 0.0:
        mask = mask & (distance <= float(max_range_m))
    return mask

def soft_fov_score(observer_pos, yaw_ref, points, horizon_fov_rad, vertical_fov_rad, pitch_ref=None, softness=0.08, camera_rot_w=None):
    if camera_rot_w is not None:
        yaw_to_point, pitch_to_point, _, in_front = bearing_to_point_camera_frame(observer_pos, points, camera_rot_w)
        yaw_err = yaw_to_point.abs()
        pitch_err = pitch_to_point.abs()
        yaw_score = torch.sigmoid((0.5 * horizon_fov_rad - yaw_err) / softness)
        pitch_score = torch.sigmoid((0.5 * vertical_fov_rad - pitch_err) / softness)
        return yaw_score * pitch_score * in_front.to(dtype=yaw_score.dtype)
    if pitch_ref is None:
        pitch_ref = torch.zeros_like(yaw_ref)
    yaw_to_point, pitch_to_point, _ = bearing_to_point(observer_pos, points)
    yaw_err = wrap_to_pi(yaw_to_point - yaw_ref[..., None]).abs()
    pitch_err = (pitch_to_point - pitch_ref[..., None]).abs()
    yaw_score = torch.sigmoid((0.5 * horizon_fov_rad - yaw_err) / softness)
    pitch_score = torch.sigmoid((0.5 * vertical_fov_rad - pitch_err) / softness)
    return yaw_score * pitch_score

import torch

from OARM.config import oarm_cfg
from OARM.visibility.soft_fov import hard_fov_mask


def first_visible_time(
    sampled_pos: torch.Tensor,
    sampled_time: torch.Tensor,
    yaw_ref: torch.Tensor,
    risk_points: torch.Tensor,
    horizon_fov_rad: float,
    vertical_fov_rad: float,
    visibility_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Return first sampled time each risk point enters FOV.

    Args:
        sampled_pos: [N, T, 3].
        sampled_time: [N, T].
        yaw_ref: [N, T].
        risk_points: [N, Q, 3].
    Returns:
        first_time: [N, Q], inf if never visible.
    """

    visible = hard_fov_mask(sampled_pos, yaw_ref, risk_points, horizon_fov_rad, vertical_fov_rad)
    if visibility_mask is not None:
        visible = visible & visibility_mask
    inf_time = torch.full_like(sampled_time[:, :1], torch.inf)
    time_grid = sampled_time[:, :, None].expand_as(visible).float()
    first = torch.where(visible, time_grid, torch.inf).amin(dim=1)
    never_visible = torch.isinf(first)
    return torch.where(never_visible, inf_time.expand_as(first), first)


def arrival_time_to_points(
    sampled_pos: torch.Tensor,
    sampled_time: torch.Tensor,
    risk_points: torch.Tensor,
    max_arrival_distance_m: float = None,
):
    """Approximate arrival time by the closest sampled trajectory point."""

    dist = (sampled_pos[:, :, None, :] - risk_points[:, None, :, :]).norm(dim=-1)
    min_distance, closest_id = dist.min(dim=1)
    arrival_time = torch.gather(sampled_time, 1, closest_id)
    if max_arrival_distance_m is None:
        return arrival_time
    return torch.where(
        min_distance <= max_arrival_distance_m,
        arrival_time,
        torch.full_like(arrival_time, torch.inf),
    )


def reaction_margin(
    sampled_pos: torch.Tensor,
    sampled_time: torch.Tensor,
    yaw_ref: torch.Tensor,
    risk_points: torch.Tensor,
    horizon_fov_rad: float,
    vertical_fov_rad: float,
    reaction_time: float = oarm_cfg.reaction_time,
    visibility_mask: torch.Tensor = None,
    max_arrival_distance_m: float = oarm_cfg.risk_arrival_radius_m,
    no_arrival_margin: float = oarm_cfg.no_arrival_margin_m,
) -> torch.Tensor:
    first_time = first_visible_time(
        sampled_pos,
        sampled_time,
        yaw_ref,
        risk_points,
        horizon_fov_rad,
        vertical_fov_rad,
        visibility_mask=visibility_mask,
    )
    arrival_time = arrival_time_to_points(
        sampled_pos,
        sampled_time,
        risk_points,
        max_arrival_distance_m=max_arrival_distance_m,
    )
    margin = arrival_time - first_time - reaction_time
    margin = torch.where(torch.isinf(arrival_time), no_arrival_margin * torch.ones_like(margin), margin)
    return torch.where(torch.isinf(first_time) & torch.isfinite(arrival_time), -reaction_time * torch.ones_like(margin), margin)

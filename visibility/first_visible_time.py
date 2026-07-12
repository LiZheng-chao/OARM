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
    camera_rot_w: torch.Tensor = None,
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

    visible = hard_fov_mask(sampled_pos, yaw_ref, risk_points, horizon_fov_rad, vertical_fov_rad, camera_rot_w=camera_rot_w)
    if visibility_mask is not None:
        visible = visible & visibility_mask
    inf_time = torch.full_like(sampled_time[:, :1], torch.inf)
    time_grid = sampled_time[:, :, None].expand_as(visible).float()
    first = torch.where(visible, time_grid, torch.inf).amin(dim=1)
    never_visible = torch.isinf(first)
    return torch.where(never_visible, inf_time.expand_as(first), first)


def first_entry_time_to_points(sampled_pos, sampled_time, risk_points, radius_m):
    dist = (sampled_pos[:, :, None, :] - risk_points[:, None, :, :]).norm(dim=-1)
    inside = dist <= radius_m
    entry_valid = inside.any(dim=1)
    first_idx = inside.float().argmax(dim=1)
    idx1 = first_idx.clamp(min=1)
    idx0 = idx1 - 1
    time_grid = sampled_time[:, :, None].expand_as(dist)
    d0 = torch.gather(dist, 1, idx0[:, None, :]).squeeze(1)
    d1 = torch.gather(dist, 1, idx1[:, None, :]).squeeze(1)
    t0 = torch.gather(time_grid, 1, idx0[:, None, :]).squeeze(1)
    t1 = torch.gather(time_grid, 1, idx1[:, None, :]).squeeze(1)
    alpha = ((d0 - radius_m) / (d0 - d1).clamp(min=1e-6)).clamp(0.0, 1.0)
    entry_time = t0 + alpha * (t1 - t0)
    entry_time = torch.where(first_idx == 0, sampled_time[:, :1].expand_as(entry_time), entry_time)
    entry_time = torch.where(entry_valid, entry_time, torch.full_like(entry_time, torch.inf))
    return entry_time, entry_valid

def arrival_time_to_points(sampled_pos, sampled_time, risk_points, max_arrival_distance_m=None):
    if max_arrival_distance_m is None:
        max_arrival_distance_m = oarm_cfg.risk_arrival_radius_m
    entry_time, _ = first_entry_time_to_points(sampled_pos, sampled_time, risk_points, max_arrival_distance_m)
    return entry_time

def reaction_margin_components(sampled_pos, sampled_time, yaw_ref, risk_points, horizon_fov_rad, vertical_fov_rad, reaction_time=oarm_cfg.reaction_time, visibility_mask=None, max_arrival_distance_m=oarm_cfg.risk_arrival_radius_m, camera_rot_w=None):
    first_time = first_visible_time(sampled_pos, sampled_time, yaw_ref, risk_points, horizon_fov_rad, vertical_fov_rad, visibility_mask=visibility_mask, camera_rot_w=camera_rot_w)
    entry_time, entry_valid = first_entry_time_to_points(sampled_pos, sampled_time, risk_points, max_arrival_distance_m)
    visible_before_entry = torch.isfinite(first_time) & entry_valid & (first_time <= entry_time)
    effective_first_time = torch.where(visible_before_entry, first_time, entry_time)
    observation_lead_time = entry_time - effective_first_time
    required_reaction_time = torch.full_like(entry_time, float(reaction_time))
    margin = observation_lead_time - required_reaction_time
    margin = torch.where(entry_valid, margin, torch.full_like(margin, torch.inf))
    return dict(first_visible_time=first_time, first_entry_time=entry_time, arrival_time=entry_time, arrival_valid=entry_valid, visible_before_entry=visible_before_entry, never_visible_before_entry=entry_valid & ~visible_before_entry, observation_lead_time=torch.where(entry_valid, observation_lead_time, torch.full_like(observation_lead_time, torch.inf)), required_reaction_time=required_reaction_time, reaction_margin_points=margin, margin_exact_valid=entry_valid, margin_censored=~entry_valid)

def reaction_margin(sampled_pos, sampled_time, yaw_ref, risk_points, horizon_fov_rad, vertical_fov_rad, reaction_time=oarm_cfg.reaction_time, visibility_mask=None, max_arrival_distance_m=oarm_cfg.risk_arrival_radius_m, no_arrival_margin=None, camera_rot_w=None):
    components = reaction_margin_components(sampled_pos, sampled_time, yaw_ref, risk_points, horizon_fov_rad, vertical_fov_rad, reaction_time=reaction_time, visibility_mask=visibility_mask, max_arrival_distance_m=max_arrival_distance_m, camera_rot_w=camera_rot_w)
    return components['reaction_margin_points']

import math
from typing import Dict

import torch

from OARM.config import oarm_cfg
from OARM.visibility.first_visible_time import reaction_margin_components
from OARM.utils.yopo_compat import ensure_yopo_path

ensure_yopo_path()
from config.config import cfg


class ReactionMarginLabeler:
    def __init__(
        self,
        horizon_fov_rad: float = math.radians(cfg['horizon_camera_fov']),
        vertical_fov_rad: float = math.radians(cfg['vertical_camera_fov']),
        reaction_time: float = oarm_cfg.reaction_time,
        risk_arrival_radius_m: float = oarm_cfg.risk_arrival_radius_m,
        no_arrival_margin_s: float = oarm_cfg.no_arrival_margin_s,
        softmin_tau: float = 0.15,
    ):
        self.horizon_fov_rad = horizon_fov_rad
        self.vertical_fov_rad = vertical_fov_rad
        self.reaction_time = reaction_time
        self.risk_arrival_radius_m = risk_arrival_radius_m
        self.no_arrival_margin_s = no_arrival_margin_s
        self.softmin_tau = softmin_tau

    def __call__(
        self,
        sampled_pos_w: torch.Tensor,
        sampled_time: torch.Tensor,
        yaw_ref: torch.Tensor,
        risk_points_w: torch.Tensor,
        risk_weight: torch.Tensor = None,
        visibility_mask: torch.Tensor = None,
        camera_rot_w: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        components = reaction_margin_components(
            sampled_pos_w,
            sampled_time,
            yaw_ref,
            risk_points_w,
            horizon_fov_rad=self.horizon_fov_rad,
            vertical_fov_rad=self.vertical_fov_rad,
            reaction_time=self.reaction_time,
            visibility_mask=visibility_mask,
            max_arrival_distance_m=self.risk_arrival_radius_m,
            camera_rot_w=camera_rot_w,
        )
        point_margin = components['reaction_margin_points']
        point_window = components['observation_lead_time']
        entry_valid = components['arrival_valid'].bool()
        visible_before_entry = components.get('visible_before_entry', torch.zeros_like(entry_valid)).bool()
        visible_at_t0 = torch.isfinite(components['first_visible_time']) & (components['first_visible_time'] <= 1e-6)
        if risk_weight is None:
            risk_weight = torch.ones_like(point_margin)
        risk_weight = risk_weight.to(device=point_margin.device, dtype=point_margin.dtype)
        weighted_valid = (risk_weight > 1e-6) & entry_valid & torch.isfinite(point_margin)
        weighted_event_valid = weighted_valid & visible_before_entry
        valid_weight = torch.where(weighted_valid, risk_weight, torch.zeros_like(risk_weight))
        candidate_valid = valid_weight.sum(dim=-1) > 1e-6
        event_valid_weight = torch.where(weighted_event_valid, risk_weight, torch.zeros_like(risk_weight))
        candidate_event_valid = event_valid_weight.sum(dim=-1) > 1e-6

        inf = torch.full_like(point_margin, torch.inf)
        weighted_margin = torch.where(weighted_valid, point_margin, inf)
        margin_min = weighted_margin.amin(dim=-1)
        margin_min = torch.where(candidate_valid, margin_min, torch.full_like(margin_min, torch.inf))
        critical_idx = weighted_margin.argmin(dim=-1)
        critical_idx = torch.where(candidate_valid, critical_idx, torch.full_like(critical_idx, -1))
        gathered_idx = critical_idx.clamp(min=0).unsqueeze(-1)
        critical_weight = risk_weight.gather(dim=-1, index=gathered_idx).squeeze(-1)
        critical_weight = torch.where(candidate_valid, critical_weight, torch.zeros_like(critical_weight))

        weighted_window = torch.where(weighted_valid, point_window, inf)
        window_min = weighted_window.amin(dim=-1)
        window_min = torch.where(candidate_valid, window_min, torch.full_like(window_min, torch.inf))

        tau = max(self.softmin_tau, 1e-4)
        invalid_log_weight = torch.full_like(risk_weight, -torch.inf)
        log_weight = torch.where(weighted_valid, valid_weight.clamp(min=1e-6).log(), invalid_log_weight)
        log_norm = torch.logsumexp(log_weight, dim=-1, keepdim=True)
        log_norm = torch.where(candidate_valid[:, None], log_norm, torch.zeros_like(log_norm))
        normalized_log_weight = log_weight - log_norm
        margin_softmin = -tau * torch.logsumexp(normalized_log_weight - point_margin / tau, dim=-1)
        margin_softmin = torch.where(candidate_valid, margin_softmin, torch.full_like(margin_softmin, torch.inf))
        window_softmin = -tau * torch.logsumexp(normalized_log_weight - point_window / tau, dim=-1)
        window_softmin = torch.where(candidate_valid, window_softmin, torch.full_like(window_softmin, torch.inf))

        masked_entry_time = torch.where(weighted_valid, components['first_entry_time'], inf)
        arrival_time_min = masked_entry_time.amin(dim=-1)
        arrival_time_min = torch.where(candidate_valid, arrival_time_min, torch.full_like(arrival_time_min, torch.inf))

        point_no_entry = (risk_weight > 1e-6) & ~entry_valid
        point_right_censored = weighted_valid & ~visible_before_entry
        point_censored = point_no_entry | point_right_censored
        candidate_right_censored = point_right_censored.any(dim=-1)
        candidate_no_entry = point_no_entry.any(dim=-1) & ~candidate_valid
        candidate_censored = candidate_right_censored | candidate_no_entry
        candidate_visible_at_t0 = (visible_at_t0 & (risk_weight > 1e-6)).any(dim=-1)
        return {
            'reaction_margin_points': point_margin,
            'reaction_margin_min': margin_min,
            'reaction_margin_softmin': margin_softmin,
            'reaction_margin_valid': candidate_valid,
            'reaction_margin_censored': candidate_censored,
            'reaction_margin_point_valid': weighted_valid,
            'reaction_margin_point_censored': point_censored,
            'reaction_window_points': point_window,
            'reaction_window_min': window_min,
            'reaction_window_softmin': window_softmin,
            'reaction_window_gt': window_softmin,
            'reaction_margin_gt': margin_softmin,
            'rm_event_valid_gt': candidate_event_valid,
            'rm_right_censored_gt': candidate_right_censored,
            'rm_no_entry_gt': candidate_no_entry,
            'risk_visible_at_t0_gt': candidate_visible_at_t0,
            'critical_risk_point_id': critical_idx,
            'critical_risk_weight': critical_weight,
            'reaction_margin_point_event_valid': weighted_event_valid,
            'reaction_margin_point_right_censored': point_right_censored,
            'reaction_margin_point_no_entry': point_no_entry,
            'first_visible_time': components['first_visible_time'],
            'first_entry_time': components['first_entry_time'],
            'arrival_time_min': arrival_time_min,
        }

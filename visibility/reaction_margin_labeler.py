import math
from typing import Dict

import torch

from OARM.config import oarm_cfg
from OARM.visibility.first_visible_time import arrival_time_to_points, reaction_margin
from OARM.utils.yopo_compat import ensure_yopo_path

ensure_yopo_path()
from config.config import cfg


class ReactionMarginLabeler:
    """Build per-candidate reaction-margin labels from sampled trajectories."""

    def __init__(
        self,
        horizon_fov_rad: float = math.radians(cfg["horizon_camera_fov"]),
        vertical_fov_rad: float = math.radians(cfg["vertical_camera_fov"]),
        reaction_time: float = oarm_cfg.reaction_time,
        softmin_tau: float = 0.15,
    ):
        self.horizon_fov_rad = horizon_fov_rad
        self.vertical_fov_rad = vertical_fov_rad
        self.reaction_time = reaction_time
        self.softmin_tau = softmin_tau

    def __call__(
        self,
        sampled_pos_w: torch.Tensor,
        sampled_time: torch.Tensor,
        yaw_ref: torch.Tensor,
        risk_points_w: torch.Tensor,
        risk_weight: torch.Tensor = None,
        visibility_mask: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        point_margin = reaction_margin(
            sampled_pos_w,
            sampled_time,
            yaw_ref,
            risk_points_w,
            horizon_fov_rad=self.horizon_fov_rad,
            vertical_fov_rad=self.vertical_fov_rad,
            reaction_time=self.reaction_time,
            visibility_mask=visibility_mask,
            max_arrival_distance_m=oarm_cfg.risk_arrival_radius_m,
            no_arrival_margin=oarm_cfg.no_arrival_margin_m,
        )
        arrival_time = arrival_time_to_points(
            sampled_pos_w,
            sampled_time,
            risk_points_w,
            max_arrival_distance_m=oarm_cfg.risk_arrival_radius_m,
        )
        if risk_weight is None:
            risk_weight = torch.ones_like(point_margin)
        risk_weight = risk_weight.float()
        weighted_margin = torch.where(risk_weight > 0.0, point_margin, torch.inf)
        margin_min = weighted_margin.amin(dim=-1)
        margin_min = torch.where(
            torch.isinf(margin_min),
            torch.full_like(margin_min, oarm_cfg.no_arrival_margin_m),
            margin_min,
        )

        no_weight = risk_weight.sum(dim=-1) <= 1e-6
        margin_valid = ~no_weight
        tau = max(self.softmin_tau, 1e-4)
        invalid_log_weight = torch.full_like(risk_weight, -torch.inf)
        log_weight = torch.where(risk_weight > 1e-6, risk_weight.clamp(min=1e-6).log(), invalid_log_weight)
        log_norm = torch.logsumexp(log_weight, dim=-1, keepdim=True)
        log_weight = torch.where(no_weight[:, None], torch.zeros_like(log_weight), log_weight - log_norm)
        margin_softmin = -tau * torch.logsumexp(log_weight - point_margin / tau, dim=-1)
        margin_softmin = torch.where(
            no_weight,
            torch.full_like(margin_softmin, oarm_cfg.no_arrival_margin_m),
            margin_softmin,
        )
        invalid_arrival = torch.full_like(arrival_time, torch.inf)
        masked_arrival = torch.where(risk_weight > 1e-6, arrival_time, invalid_arrival)
        arrival_time_min = masked_arrival.amin(dim=-1)
        arrival_time_min = torch.where(torch.isinf(arrival_time_min), torch.zeros_like(arrival_time_min), arrival_time_min)
        return {
            "reaction_margin_points": point_margin,
            "reaction_margin_min": margin_min,
            "reaction_margin_softmin": margin_softmin,
            "reaction_margin_valid": margin_valid,
            "arrival_time_min": arrival_time_min,
        }

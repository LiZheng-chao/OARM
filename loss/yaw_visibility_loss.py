import math

import torch
import torch.nn as nn

from OARM.config import oarm_cfg
from OARM.visibility.soft_fov import soft_fov_score
from OARM.utils.yopo_compat import ensure_yopo_path

ensure_yopo_path()
from config.config import cfg


class YawVisibilityLoss(nn.Module):
    """Differentiable risk-point visibility cost for the OARM yaw head."""

    def __init__(
        self,
        horizon_fov_rad: float = math.radians(cfg["horizon_camera_fov"]),
        vertical_fov_rad: float = math.radians(cfg["vertical_camera_fov"]),
        yaw_rate_weight: float = 0.02,
    ):
        super().__init__()
        self.horizon_fov_rad = horizon_fov_rad
        self.vertical_fov_rad = vertical_fov_rad
        self.yaw_rate_weight = yaw_rate_weight

    def forward(
        self,
        sampled_pos_w: torch.Tensor,
        yaw_ref: torch.Tensor,
        yaw_rate: torch.Tensor,
        risk_points_w: torch.Tensor,
        risk_weight: torch.Tensor = None,
        sampled_time: torch.Tensor = None,
        arrival_time: torch.Tensor = None,
        visibility_mask: torch.Tensor = None,
    ):
        if risk_weight is None:
            risk_weight = torch.ones(
                risk_points_w.shape[:-1],
                device=risk_points_w.device,
                dtype=risk_points_w.dtype,
            )
        risk_weight = risk_weight.float()
        vis = soft_fov_score(
            sampled_pos_w,
            yaw_ref,
            risk_points_w,
            horizon_fov_rad=self.horizon_fov_rad,
            vertical_fov_rad=self.vertical_fov_rad,
        )
        if visibility_mask is not None:
            vis = vis * visibility_mask.to(device=vis.device, dtype=vis.dtype)
        if sampled_time is None:
            best_vis = vis.max(dim=1).values
        else:
            time_weight = torch.exp(-sampled_time / max(oarm_cfg.yaw_early_time_tau, 1e-3))
            if arrival_time is not None:
                latest_useful_time = arrival_time - oarm_cfg.reaction_time
                before_arrival = torch.sigmoid((latest_useful_time[:, None, :] - sampled_time[:, :, None]) / 0.1)
                time_weight = time_weight[:, :, None] * before_arrival
                best_vis = (time_weight * vis).sum(dim=1) / time_weight.sum(dim=1).clamp(min=1e-6)
                denom = risk_weight.sum(dim=-1).clamp(min=1e-3)
                visibility_cost = (risk_weight * (1.0 - best_vis)).sum(dim=-1) / denom
                yaw_smooth_cost = yaw_rate.square().mean(dim=-1)
                return visibility_cost + self.yaw_rate_weight * yaw_smooth_cost
            time_weight = time_weight / time_weight.sum(dim=1, keepdim=True).clamp(min=1e-6)
            best_vis = (time_weight[:, :, None] * vis).sum(dim=1)
        denom = risk_weight.sum(dim=-1).clamp(min=1e-3)
        visibility_cost = (risk_weight * (1.0 - best_vis)).sum(dim=-1) / denom
        yaw_smooth_cost = yaw_rate.square().mean(dim=-1)
        return visibility_cost + self.yaw_rate_weight * yaw_smooth_cost

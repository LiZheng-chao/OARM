import math
from typing import Tuple

import torch

from OARM.config import oarm_cfg
from OARM.utils.yopo_compat import ensure_yopo_path

ensure_yopo_path()
from config.config import cfg


class RiskPointSampler:
    """Sample fixed-size occlusion risk point proposals behind depth frontiers.

    This is the training-interface version of privileged risk points: it uses
    depth frontiers as proposals today, while leaving a single place to add
    GT ESDF filtering later.
    """

    def __init__(
        self,
        point_count: int = oarm_cfg.risk_point_count,
        depth_max_m: float = oarm_cfg.risk_depth_max_m,
        offsets_m: Tuple[float, ...] = oarm_cfg.risk_point_offsets_m,
    ):
        self.point_count = point_count
        self.depth_max_m = depth_max_m
        self.offsets_m = torch.tensor(offsets_m, dtype=torch.float32)
        self.horizon_fov = math.radians(cfg["horizon_camera_fov"])
        self.vertical_fov = math.radians(cfg["vertical_camera_fov"])

    def __call__(self, depth: torch.Tensor, frontier_map: torch.Tensor):
        """Return body-frame risk points and weights.

        Args:
            depth: [1, H, W] normalized 0..1 depth.
            frontier_map: [1, H, W] frontier probability/mask.
        Returns:
            risk_points_b: [Q, 3], risk_weight: [Q].
        """

        if depth.dim() != 3 or depth.shape[0] != 1:
            raise ValueError(f"depth must be [1,H,W], got {tuple(depth.shape)}")
        if frontier_map.dim() != 3 or frontier_map.shape[0] != 1:
            raise ValueError(f"frontier_map must be [1,H,W], got {tuple(frontier_map.shape)}")

        device = depth.device
        dtype = depth.dtype
        height, width = depth.shape[-2:]
        score = frontier_map.reshape(-1).float()
        q = self.point_count
        if score.numel() == 0:
            return self.empty_points(device, dtype)

        k = min(q, score.numel())
        top_score, top_id = torch.topk(score, k=k, largest=True)
        if k < q:
            pad_n = q - k
            top_id = torch.cat([top_id, top_id.new_zeros(pad_n)], dim=0)
            top_score = torch.cat([top_score, top_score.new_zeros(pad_n)], dim=0)

        pixel_v = torch.div(top_id, width, rounding_mode="floor").to(dtype)
        pixel_u = (top_id % width).to(dtype)
        depth_flat = depth.reshape(-1)
        depth_m = depth_flat[top_id].to(dtype) * self.depth_max_m
        offsets = self.offsets_m.to(device=device, dtype=dtype)
        offset = offsets[torch.arange(q, device=device) % offsets.numel()]
        ray_depth = (depth_m + offset).clamp(min=0.2, max=self.depth_max_m + offsets.max())

        yaw = ((pixel_u + 0.5) / width - 0.5) * self.horizon_fov
        pitch = (0.5 - (pixel_v + 0.5) / height) * self.vertical_fov
        cos_pitch = torch.cos(pitch)
        direction = torch.stack(
            [
                cos_pitch * torch.cos(yaw),
                cos_pitch * torch.sin(yaw),
                torch.sin(pitch),
            ],
            dim=-1,
        )
        risk_points_b = direction * ray_depth[:, None]
        risk_weight = top_score.to(dtype).clamp(0.0, 1.0)
        return risk_points_b, risk_weight

    def empty_points(self, device, dtype):
        points = torch.zeros((self.point_count, 3), device=device, dtype=dtype)
        points[:, 0] = self.depth_max_m
        weights = torch.zeros((self.point_count,), device=device, dtype=dtype)
        return points, weights


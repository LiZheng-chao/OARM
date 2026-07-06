import math

import torch
import torch.nn.functional as F

from OARM.config import oarm_cfg
from OARM.utils.yopo_compat import ensure_yopo_path

ensure_yopo_path()
from config.config import cfg


def visible_free_distance_from_depth(
    depth: torch.Tensor,
    candidate_pos_b: torch.Tensor,
    depth_max_m: float = oarm_cfg.risk_depth_max_m,
    sector_px: int = oarm_cfg.visible_free_sector_px,
) -> torch.Tensor:
    """Estimate visible free distance along each candidate endpoint direction.

    Args:
        depth: [B, 1, H, W] normalized depth image.
        candidate_pos_b: [B*K, 3] endpoint direction in body/camera frame.
    Returns:
        visible distance [B*K] in meters from the current depth image.
    """

    if depth.dim() != 4 or depth.shape[1] != 1:
        raise ValueError(f"depth must be [B,1,H,W], got {tuple(depth.shape)}")
    batch, _, height, width = depth.shape
    candidate_count = candidate_pos_b.shape[0]
    if candidate_count % batch != 0:
        raise ValueError("candidate_pos_b first dimension must be a multiple of depth batch size")
    repeat = candidate_count // batch

    depth_expanded = depth.repeat_interleave(repeat, dim=0)
    direction = candidate_pos_b / candidate_pos_b.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    yaw = torch.atan2(direction[:, 1], direction[:, 0])
    pitch = torch.atan2(direction[:, 2], direction[:, 0:2].norm(dim=-1).clamp(min=1e-6))

    horizon_fov = math.radians(cfg["horizon_camera_fov"])
    vertical_fov = math.radians(cfg["vertical_camera_fov"])
    grid_x = (yaw / (0.5 * horizon_fov)).clamp(-1.0, 1.0)
    grid_y = (-pitch / (0.5 * vertical_fov)).clamp(-1.0, 1.0)

    offsets = [(0.0, 0.0)]
    if sector_px > 0:
        px_x = 2.0 / max(width - 1, 1)
        px_y = 2.0 / max(height - 1, 1)
        for du in range(-sector_px, sector_px + 1):
            for dv in range(-sector_px, sector_px + 1):
                if du == 0 and dv == 0:
                    continue
                offsets.append((du * px_x, dv * px_y))

    samples = []
    for dx, dy in offsets:
        grid = torch.stack(
            [
                (grid_x + dx).clamp(-1.0, 1.0),
                (grid_y + dy).clamp(-1.0, 1.0),
            ],
            dim=-1,
        ).view(candidate_count, 1, 1, 2)
        sampled = F.grid_sample(depth_expanded, grid, mode="bilinear", align_corners=True)
        samples.append(sampled.view(candidate_count))
    depth_samples = torch.stack(samples, dim=-1)
    return depth_samples.amin(dim=-1) * depth_max_m

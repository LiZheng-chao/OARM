import torch
import torch.nn.functional as F

from OARM.config import oarm_cfg


class DepthFrontierExtractor:
    """Extract a simple occlusion-frontier mask from normalized depth images."""

    def __init__(
        self,
        edge_threshold: float = oarm_cfg.risk_depth_edge_threshold,
        far_threshold: float = oarm_cfg.risk_depth_far_threshold,
        border_px: int = oarm_cfg.risk_frontier_border_px,
    ):
        self.edge_threshold = edge_threshold
        self.far_threshold = far_threshold
        self.border_px = border_px

    def __call__(self, depth: torch.Tensor) -> torch.Tensor:
        if depth.dim() != 4 or depth.shape[1] != 1:
            raise ValueError("depth must have shape [B, 1, H, W]")

        finite = torch.isfinite(depth)
        safe_depth = torch.where(finite, depth, torch.zeros_like(depth))
        dx = F.pad(safe_depth[:, :, :, 1:] - safe_depth[:, :, :, :-1], (0, 1, 0, 0))
        dy = F.pad(safe_depth[:, :, 1:, :] - safe_depth[:, :, :-1, :], (0, 0, 0, 1))
        edge = torch.sqrt(dx.square() + dy.square()) > self.edge_threshold
        far = finite & (depth > self.far_threshold)
        invalid = (~finite) | (depth <= 0.0)

        border = torch.zeros_like(depth, dtype=torch.bool)
        p = self.border_px
        if p > 0:
            border[:, :, :p, :] = True
            border[:, :, -p:, :] = True
            border[:, :, :, :p] = True
            border[:, :, :, -p:] = True

        far_boundary = self.mask_boundary(far)
        invalid_boundary = self.mask_boundary(invalid)
        frontier = edge | far_boundary | invalid_boundary | (far & border)
        return frontier.float()

    @staticmethod
    def mask_boundary(mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.float()
        dilated = F.max_pool2d(mask_f, kernel_size=3, stride=1, padding=1)
        eroded = 1.0 - F.max_pool2d(1.0 - mask_f, kernel_size=3, stride=1, padding=1)
        return (dilated - eroded) > 0.0


def candidate_frontier_overlap(frontier_map: torch.Tensor, vertical_num: int, horizon_num: int) -> torch.Tensor:
    """Downsample frontier evidence to the primitive lattice."""

    return F.adaptive_avg_pool2d(frontier_map, (vertical_num, horizon_num))


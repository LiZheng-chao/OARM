import hashlib
import math
import os
from functools import lru_cache

import numpy as np
import torch

from OARM.config import oarm_cfg
from OARM.utils.yopo_compat import ensure_yopo_path

ensure_yopo_path()
from config.config import cfg


@lru_cache(maxsize=16)
def load_pointcloud_points(dataset_dir: str, map_id: int):
    path = os.path.join(dataset_dir, f"pointcloud-{int(map_id)}.ply")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"GT pointcloud not found: {path}")
    import open3d as o3d

    pointcloud = o3d.io.read_point_cloud(path)
    points = np.asarray(pointcloud.points, dtype=np.float32)
    if points.size == 0:
        raise ValueError(f"GT pointcloud has no points: {path}")
    return points


class GTRiskPointSampler:
    """Sample hidden risk points from the full offline pointcloud.

    This sampler gives training/evaluation an explicit privileged GT source
    separate from depth-frontier proxy labels. It projects full-map points into
    the current depth image and keeps points that are in the camera frustum but
    lie behind the observed depth, i.e. hidden by current visible geometry.
    """

    def __init__(
        self,
        dataset_dir: str,
        point_count: int = oarm_cfg.risk_point_count,
        depth_max_m: float = oarm_cfg.risk_depth_max_m,
        hidden_depth_margin_m: float = oarm_cfg.gt_hidden_depth_margin_m,
        min_forward_m: float = oarm_cfg.gt_min_forward_m,
        max_forward_m: float = oarm_cfg.gt_max_forward_m,
        horizon_fov_expand_deg: float = oarm_cfg.gt_horizon_fov_expand_deg,
        vertical_fov_expand_deg: float = oarm_cfg.gt_vertical_fov_expand_deg,
        depth_metric: str = oarm_cfg.gt_depth_metric,
    ):
        self.dataset_dir = dataset_dir
        self.point_count = int(point_count)
        self.depth_max_m = float(depth_max_m)
        self.hidden_depth_margin_m = float(hidden_depth_margin_m)
        self.min_forward_m = float(min_forward_m)
        self.max_forward_m = float(max_forward_m)
        if depth_metric not in {"forward", "ray"}:
            raise ValueError(f"Unknown gt depth metric: {depth_metric}")
        self.depth_metric = depth_metric
        self.horizon_fov = math.radians(cfg["horizon_camera_fov"])
        self.vertical_fov = math.radians(cfg["vertical_camera_fov"])
        self.expanded_horizon_fov = self.horizon_fov + math.radians(float(horizon_fov_expand_deg))
        self.expanded_vertical_fov = self.vertical_fov + math.radians(float(vertical_fov_expand_deg))

    def __call__(self, depth: torch.Tensor, pos_w: torch.Tensor, rot_wb: torch.Tensor, map_id: int):
        if depth.dim() != 3 or depth.shape[0] != 1:
            raise ValueError(f"depth must be [1,H,W], got {tuple(depth.shape)}")
        device = depth.device
        dtype = depth.dtype
        empty_points, empty_weight = self.empty_points(device, dtype, pos_w)

        try:
            points_w_np = load_pointcloud_points(self.dataset_dir, int(map_id))
        except Exception as exc:
            print(f"GTRiskPointSampler disabled for map {int(map_id)}: {exc}")
            return empty_points, empty_weight

        points_w = torch.as_tensor(points_w_np, device=device, dtype=dtype)
        pos_w = pos_w.to(device=device, dtype=dtype)
        rot_wb = rot_wb.to(device=device, dtype=dtype)
        points_b = torch.matmul(rot_wb.transpose(0, 1), (points_w - pos_w).unsqueeze(-1)).squeeze(-1)

        x = points_b[:, 0]
        y = points_b[:, 1]
        z = points_b[:, 2]
        yaw = torch.atan2(y, x.clamp(min=1e-4))
        planar = torch.sqrt(x.square() + y.square()).clamp(min=1e-4)
        pitch = torch.atan2(z, planar)
        point_range = points_b.norm(dim=-1)
        depth_axis = point_range if self.depth_metric == "ray" else x
        forward = (x > self.min_forward_m) & (depth_axis < min(self.max_forward_m, self.depth_max_m))
        in_current_fov = (yaw.abs() < 0.5 * self.horizon_fov) & (pitch.abs() < 0.5 * self.vertical_fov)
        in_expanded_fov = (yaw.abs() < 0.5 * self.expanded_horizon_fov) & (pitch.abs() < 0.5 * self.expanded_vertical_fov)
        candidate_mask = forward & in_expanded_fov
        if not bool(candidate_mask.any()):
            return empty_points, empty_weight

        height, width = depth.shape[-2:]
        local_yaw = yaw[candidate_mask]
        local_pitch = pitch[candidate_mask]
        local_current_fov = in_current_fov[candidate_mask]
        local_depth = depth_axis[candidate_mask]
        hidden = ~local_current_fov
        hidden_gap = torch.ones_like(local_depth)
        if bool(local_current_fov.any()):
            u = ((local_yaw[local_current_fov] / self.horizon_fov) + 0.5) * width - 0.5
            v = (0.5 - local_pitch[local_current_fov] / self.vertical_fov) * height - 0.5
            ui = u.round().long().clamp(0, width - 1)
            vi = v.round().long().clamp(0, height - 1)
            observed_depth = depth[0, vi, ui].to(dtype) * self.depth_max_m
            current_depth = local_depth[local_current_fov]
            current_hidden = current_depth > (observed_depth + self.hidden_depth_margin_m)
            hidden[local_current_fov] = current_hidden
            hidden_gap[local_current_fov] = (current_depth - observed_depth).clamp(min=0.0)
        if not bool(hidden.any()):
            return empty_points, empty_weight

        hidden_points_w = points_w[candidate_mask][hidden]
        hidden_depth = local_depth[hidden]
        hidden_yaw = local_yaw[hidden]
        hidden_pitch = local_pitch[hidden]
        hidden_gap = hidden_gap[hidden].clamp(min=0.0)
        hidden_score = hidden_gap / hidden_gap.max().clamp(min=1e-3)
        side_score = (hidden_yaw.abs() / max(0.5 * self.expanded_horizon_fov, 1e-3)).clamp(0.0, 1.0)
        pitch_score = (1.0 - hidden_pitch.abs() / max(0.5 * self.expanded_vertical_fov, 1e-3)).clamp(0.0, 1.0)
        score = torch.maximum(hidden_score, 0.35 + 0.65 * side_score) * pitch_score
        score = score * torch.exp(-hidden_depth / max(self.max_forward_m, 1e-3))

        k = min(self.point_count, hidden_points_w.shape[0])
        top_score, top_id = torch.topk(score, k=k, largest=True)
        risk_points_w = hidden_points_w[top_id]
        risk_weight = top_score.clamp(0.0, 1.0)
        if k < self.point_count:
            pad_points = empty_points[: self.point_count - k]
            pad_weight = empty_weight[: self.point_count - k]
            risk_points_w = torch.cat([risk_points_w, pad_points], dim=0)
            risk_weight = torch.cat([risk_weight, pad_weight], dim=0)
        return risk_points_w.to(dtype), risk_weight.to(dtype)

    def cache_metadata(self):
        return {
            "point_count": self.point_count,
            "depth_max_m": self.depth_max_m,
            "hidden_depth_margin_m": self.hidden_depth_margin_m,
            "min_forward_m": self.min_forward_m,
            "max_forward_m": self.max_forward_m,
            "horizon_fov": self.horizon_fov,
            "vertical_fov": self.vertical_fov,
            "expanded_horizon_fov": self.expanded_horizon_fov,
            "expanded_vertical_fov": self.expanded_vertical_fov,
            "depth_metric": self.depth_metric,
        }

    def cache_tag(self):
        items = sorted(self.cache_metadata().items())
        payload = repr(items).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()[:10]
    def empty_points(self, device, dtype, pos_w):
        points = torch.zeros((self.point_count, 3), device=device, dtype=dtype)
        points[:, 0] = self.depth_max_m
        if pos_w is not None:
            points = points + pos_w.to(device=device, dtype=dtype)[None, :]
        weights = torch.zeros((self.point_count,), device=device, dtype=dtype)
        return points, weights

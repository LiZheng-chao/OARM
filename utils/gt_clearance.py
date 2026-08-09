import os
from functools import lru_cache

import numpy as np
import torch
from scipy.spatial import cKDTree


@lru_cache(maxsize=32)
def load_pointcloud_tree(dataset_dir, map_id):
    path = os.path.join(dataset_dir, f"pointcloud-{int(map_id)}.ply")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"GT pointcloud not found for clearance oracle: {path}")
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("GT clearance oracle requires open3d to read pointcloud .ply files") from exc
    pointcloud = o3d.io.read_point_cloud(path)
    points = np.asarray(pointcloud.points, dtype=np.float32)
    if points.size == 0:
        raise ValueError(f"GT pointcloud has no points: {path}")
    return cKDTree(points)


def candidate_min_clearance_gt(sampled_pos_w, map_id, dataset_dir):
    """Return per-candidate minimum pointcloud distance along sampled trajectories.

    sampled_pos_w: [candidate_count, sample_count, 3] world positions.
    map_id: [candidate_count] map ids aligned with sampled_pos_w.
    """
    if not dataset_dir:
        raise ValueError("dataset_dir is required for GT clearance oracle")
    if sampled_pos_w.dim() != 3 or sampled_pos_w.shape[-1] != 3:
        raise ValueError(f"sampled_pos_w must have shape [N,T,3], got {tuple(sampled_pos_w.shape)}")
    device = sampled_pos_w.device
    dtype = sampled_pos_w.dtype
    sampled_np = sampled_pos_w.detach().cpu().numpy().astype(np.float32, copy=False)
    map_np = map_id.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
    if map_np.shape[0] != sampled_np.shape[0]:
        raise ValueError("map_id must be expanded to match sampled_pos_w candidates")
    out = np.full((sampled_np.shape[0],), np.nan, dtype=np.float32)
    for mid in np.unique(map_np):
        idx = np.nonzero(map_np == mid)[0]
        tree = load_pointcloud_tree(os.path.abspath(dataset_dir), int(mid))
        flat = sampled_np[idx].reshape(-1, 3)
        dist, _ = tree.query(flat, k=1, workers=-1)
        out[idx] = dist.reshape(len(idx), sampled_np.shape[1]).min(axis=1).astype(np.float32)
    return torch.as_tensor(out, device=device, dtype=dtype)

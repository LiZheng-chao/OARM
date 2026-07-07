import hashlib
import os

import torch
from torch.utils.data import Dataset

from OARM.config import oarm_cfg
from OARM.utils.occlusion import DepthFrontierExtractor, candidate_frontier_overlap
from OARM.utils.privileged_risk_filter import PrivilegedRiskPointFilter
from OARM.utils.gt_risk_point_sampler import GTRiskPointSampler
from OARM.utils.risk_point_sampler import RiskPointSampler
from OARM.utils.yopo_compat import ensure_yopo_path
from OARM.utils.yopo_dataset_context import resolve_dataset_dir, yopo_dataset_cfg

ensure_yopo_path()
from config.config import cfg
from policy.yopo_dataset import YOPODataset


class OARMDataset(Dataset):
    """YOPO dataset wrapper with OARM frontier fields.

    It returns the original YOPO tuple plus a label dictionary. The first labels
    are weak proxies from the current depth frame; privileged ESDF-derived risk
    points can be added later without changing the trainer interface.
    """

    def __init__(
        self,
        mode="train",
        val_ratio=0.1,
        dataset_root=None,
        use_privileged_risk_filter=oarm_cfg.use_privileged_risk_filter,
        risk_label_source=oarm_cfg.risk_label_source,
    ):
        super().__init__()
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        dataset_dir = resolve_dataset_dir(dataset_root)
        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(
                f"OARM/YOPODataset expects training data at {dataset_dir}. "
                "Generate or place the YOPO dataset there before running dataset or one-batch-loss checks."
            )
        self.dataset_dir = dataset_dir
        with yopo_dataset_cfg(dataset_dir):
            self.base = YOPODataset(mode=mode, val_ratio=val_ratio)
        dataset_tag = self.dataset_cache_tag(dataset_dir)
        self.cache_dir = os.path.join(
            repo_root,
            "OARM",
            "cache",
            f"{oarm_cfg.privileged_risk_cache_dir}_{dataset_tag}",
        )
        self.frontier = DepthFrontierExtractor()
        self.risk_sampler = RiskPointSampler()
        if risk_label_source not in {"proxy", "proxy_esdf", "gt_pointcloud"}:
            raise ValueError(f"Unknown risk_label_source: {risk_label_source}")
        self.risk_label_source = risk_label_source
        self.gt_risk_sampler = GTRiskPointSampler(dataset_dir) if risk_label_source == "gt_pointcloud" else None
        self.use_privileged_risk_filter = bool(use_privileged_risk_filter or risk_label_source == "proxy_esdf")
        self.risk_filter = None
        self.vertical_num = cfg["vertical_num"]
        self.horizon_num = cfg["horizon_num"]

    @staticmethod
    def dataset_cache_tag(dataset_dir):
        name = os.path.basename(os.path.normpath(dataset_dir)) or "dataset"
        digest = hashlib.sha1(os.path.abspath(dataset_dir).encode("utf-8")).hexdigest()[:8]
        safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)
        return f"{safe_name}_{digest}"

    def __len__(self):
        return len(self.base)

    def __getitem__(self, item):
        image, pos, rot_wb, obs_b, map_id = self.base[item]
        depth = torch.as_tensor(image).float()
        if depth.dim() == 2:
            depth = depth.unsqueeze(0)
        elif depth.dim() != 3 or depth.shape[0] != 1:
            raise ValueError(f"Unexpected depth shape: {tuple(depth.shape)}")

        frontier_map = self.frontier(depth.unsqueeze(0)).squeeze(0)
        risk_points_b, risk_weight = self.risk_sampler(depth, frontier_map)
        pos_t = torch.as_tensor(pos, dtype=torch.float32)
        rot_t = torch.as_tensor(rot_wb, dtype=torch.float32)
        risk_points_w = torch.matmul(rot_t, risk_points_b.unsqueeze(-1)).squeeze(-1) + pos_t
        risk_esdf = torch.full_like(risk_weight, torch.nan)
        if self.risk_label_source == "gt_pointcloud":
            cached = self.load_cached_privileged_labels(item)
            if cached is not None and "risk_points_w" in cached:
                risk_points_w = cached["risk_points_w"]
                risk_weight = cached["risk_weight"]
                risk_esdf = cached.get("risk_esdf", torch.full_like(risk_weight, torch.nan))
            else:
                risk_points_w, risk_weight = self.gt_risk_sampler(depth, pos_t, rot_t, map_id)
                risk_esdf = torch.full_like(risk_weight, torch.nan)
                self.save_cached_privileged_labels(item, risk_weight, risk_esdf, risk_points_w=risk_points_w)
        elif self.use_privileged_risk_filter:
            cached = self.load_cached_privileged_labels(item)
            if cached is not None:
                risk_weight = cached["risk_weight"]
                risk_esdf = cached["risk_esdf"]
            elif self.ensure_risk_filter():
                risk_weight, risk_esdf = self.risk_filter(risk_points_w, risk_weight, map_id)
                self.save_cached_privileged_labels(item, risk_weight, risk_esdf)
        yaw0 = torch.atan2(rot_t[1, 0], rot_t[0, 0])
        risk_lattice = candidate_frontier_overlap(
            frontier_map.unsqueeze(0), self.vertical_num, self.horizon_num
        ).squeeze(0)
        labels = {
            "frontier_map": frontier_map.float(),
            "occlusion_risk": risk_lattice.float(),
            "frontier_backup_feasible": (risk_lattice < oarm_cfg.yield_risk_threshold).float(),
            "yield_feasible": (risk_lattice < oarm_cfg.yield_risk_threshold).float(),
            "backup_feasible": (risk_lattice < oarm_cfg.yield_risk_threshold).float(),
            "frontier_yield_feasible": (risk_lattice < oarm_cfg.yield_risk_threshold).float(),
            "risk_points_w": risk_points_w.float(),
            "risk_weight": risk_weight.float(),
            "risk_esdf": risk_esdf.float(),
            "hidden_risk_gt": (risk_weight > 1e-6).float(),
            "uses_gt_reaction_margin": torch.tensor(self.risk_label_source == "gt_pointcloud", dtype=torch.float32),
            "uses_proxy_reaction_margin": torch.tensor(self.risk_label_source != "gt_pointcloud", dtype=torch.float32),
            "reaction_margin_label_source_id": torch.tensor({"proxy": 0, "proxy_esdf": 1, "gt_pointcloud": 2}[self.risk_label_source], dtype=torch.long),
            "yaw0": yaw0.float(),
            "yaw_rate0": torch.zeros((), dtype=torch.float32),
        }
        return depth, pos, rot_wb, obs_b, map_id, labels

    def ensure_risk_filter(self):
        if self.risk_filter is not None:
            return True
        try:
            with yopo_dataset_cfg(self.dataset_dir):
                self.risk_filter = PrivilegedRiskPointFilter(device=torch.device("cpu"))
        except Exception as exc:
            print(f"PrivilegedRiskPointFilter disabled: {exc}")
            self.use_privileged_risk_filter = False
            return False
        return True

    def cache_path(self, item):
        if not oarm_cfg.cache_privileged_risk_labels:
            return None
        image_path = getattr(self.base, "img_list", [None])[item]
        if image_path is None:
            return None
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        map_id = int(self.base.map_idx[item])
        return os.path.join(self.cache_dir, f"map{map_id}_{image_name}_{self.risk_label_source}.pt")

    def load_cached_privileged_labels(self, item):
        path = self.cache_path(item)
        if path is None or not os.path.isfile(path):
            return None
        try:
            try:
                data = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                data = torch.load(path, map_location="cpu")
            if "risk_weight" not in data:
                return None
            if self.risk_label_source == "gt_pointcloud":
                if "risk_points_w" in data:
                    return data
            elif "risk_esdf" in data and torch.isfinite(data["risk_esdf"]).all():
                return data
        except Exception:
            return None
        return None

    def save_cached_privileged_labels(self, item, risk_weight, risk_esdf, risk_points_w=None):
        path = self.cache_path(item)
        if path is None:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp"
        data = {
            "risk_label_source": self.risk_label_source,
            "risk_weight": risk_weight.detach().cpu(),
            "risk_esdf": risk_esdf.detach().cpu(),
        }
        if risk_points_w is not None:
            data["risk_points_w"] = risk_points_w.detach().cpu()
        torch.save(data, tmp_path)
        os.replace(tmp_path, path)

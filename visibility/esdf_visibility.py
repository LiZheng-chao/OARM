import torch

from OARM.config import oarm_cfg
from OARM.utils.yopo_compat import ensure_yopo_path

ensure_yopo_path()


class ESDFLineOfSight:
    def __init__(
        self,
        device=None,
        ray_samples: int = oarm_cfg.visibility_ray_samples,
        ray_step_m: float = oarm_cfg.visibility_ray_step_m,
        clearance_m: float = oarm_cfg.visibility_clearance_m,
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ray_samples = ray_samples
        self.ray_step_m = ray_step_m
        self.clearance_m = clearance_m
        from loss.safety_loss import SafetyLoss

        self.query_backend = SafetyLoss(torch.eye(6, device=self.device))

    @torch.no_grad()
    def __call__(
        self,
        observer_pos_w: torch.Tensor,
        risk_points_w: torch.Tensor,
        map_id: torch.Tensor,
    ) -> torch.Tensor:
        n, t, _ = observer_pos_w.shape
        q = risk_points_w.shape[1]
        device = observer_pos_w.device
        dtype = observer_pos_w.dtype

        start = observer_pos_w[:, :, None, None, :]
        end = risk_points_w[:, None, :, None, :]
        ray_vec = end - start
        ray_length = ray_vec.squeeze(3).norm(dim=-1)
        if self.ray_samples <= 0 and self.ray_step_m <= 0.0:
            return torch.ones((n, t, q), device=device, dtype=torch.bool)

        if self.ray_step_m > 0.0:
            per_ray_samples = torch.ceil(ray_length / max(float(self.ray_step_m), 1e-3)).to(torch.long) - 1
            per_ray_samples = per_ray_samples.clamp(min=0)
            if self.ray_samples > 0:
                per_ray_samples = torch.maximum(per_ray_samples, torch.full_like(per_ray_samples, int(self.ray_samples)))
            adaptive_samples = int(per_ray_samples.max().detach().cpu().item()) if per_ray_samples.numel() > 0 else 0
        else:
            per_ray_samples = torch.full((n, t, q), int(self.ray_samples), device=device, dtype=torch.long)
            adaptive_samples = int(self.ray_samples)
        sample_count = max(int(self.ray_samples), adaptive_samples)
        if sample_count <= 0:
            return torch.ones((n, t, q), device=device, dtype=torch.bool)

        alpha = torch.linspace(
            1.0 / (sample_count + 1),
            sample_count / (sample_count + 1),
            sample_count,
            device=device,
            dtype=dtype,
        )
        ray_points = start + alpha[None, None, None, :, None] * ray_vec
        query_points = ray_points.reshape(n, t * q * sample_count, 3).to(self.device)
        query_map_id = map_id.to(self.device).long()
        _, dist = self.query_backend.get_distance_cost(query_points, query_map_id)
        dist = dist.to(device=device, dtype=dtype).reshape(n, t, q, sample_count)

        sample_id = torch.arange(sample_count, device=device)
        valid_sample = sample_id[None, None, None, :] < per_ray_samples[..., None].to(device)
        clear_or_unused = (dist > self.clearance_m) | ~valid_sample
        return clear_or_unused.all(dim=-1)

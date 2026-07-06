import torch

from OARM.config import oarm_cfg
from OARM.utils.yopo_compat import ensure_yopo_path

ensure_yopo_path()


class ESDFLineOfSight:
    """Occlusion-aware visibility mask from GT ESDF ray consistency."""

    def __init__(
        self,
        device=None,
        ray_samples: int = oarm_cfg.visibility_ray_samples,
        clearance_m: float = oarm_cfg.visibility_clearance_m,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ray_samples = ray_samples
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
        """Return [N, T, Q] True where the ray is free by ESDF clearance.

        Args:
            observer_pos_w: [N, T, 3].
            risk_points_w: [N, Q, 3].
            map_id: [N] candidate-expanded map id.
        """

        n, t, _ = observer_pos_w.shape
        q = risk_points_w.shape[1]
        device = observer_pos_w.device
        dtype = observer_pos_w.dtype
        if self.ray_samples <= 0:
            return torch.ones((n, t, q), device=device, dtype=torch.bool)

        alpha = torch.linspace(
            1.0 / (self.ray_samples + 1),
            self.ray_samples / (self.ray_samples + 1),
            self.ray_samples,
            device=device,
            dtype=dtype,
        )
        start = observer_pos_w[:, :, None, None, :]
        end = risk_points_w[:, None, :, None, :]
        ray_points = start + alpha[None, None, None, :, None] * (end - start)
        query_points = ray_points.reshape(n, t * q * self.ray_samples, 3).to(self.device)
        query_map_id = map_id.to(self.device).long()
        _, dist = self.query_backend.get_distance_cost(query_points, query_map_id)
        dist = dist.to(device=device, dtype=dtype).reshape(n, t, q, self.ray_samples)
        return (dist > self.clearance_m).all(dim=-1)


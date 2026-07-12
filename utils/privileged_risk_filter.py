import torch

from OARM.config import oarm_cfg
from OARM.utils.yopo_compat import ensure_yopo_path

ensure_yopo_path()


class PrivilegedRiskPointFilter:
    """Filter risk point proposals with YOPO's GT ESDF backend.

    The filter keeps the OARM dataset interface stable while upgrading labels
    from depth-frontier proposals to training-time privileged risk weights.
    """

    def __init__(
        self,
        device=None,
        risk_distance_m: float = oarm_cfg.privileged_risk_distance_m,
        sigma_m: float = oarm_cfg.privileged_risk_sigma_m,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.risk_distance_m = risk_distance_m
        self.sigma_m = sigma_m
        from loss.safety_loss import SafetyLoss

        self.query_backend = SafetyLoss(torch.eye(6, device=self.device))

    @torch.no_grad()
    def __call__(self, risk_points_w: torch.Tensor, risk_weight: torch.Tensor, map_id: int):
        """Return ESDF-filtered risk weights and queried distances.

        Args:
            risk_points_w: [Q, 3] world-frame risk point proposals.
            risk_weight: [Q] proposal weights from depth/frontier evidence.
            map_id: integer dataset map id.
        """

        original_device = risk_points_w.device
        points = risk_points_w.to(self.device).unsqueeze(0)
        map_tensor = torch.tensor([int(map_id)], device=self.device, dtype=torch.long)
        _, dist = self.query_backend.get_distance_cost(points, map_tensor)
        dist = dist.squeeze(0).to(device=original_device, dtype=risk_points_w.dtype)
        privileged_weight = torch.sigmoid((self.risk_distance_m - dist) / max(self.sigma_m, 1e-3))
        filtered_weight = risk_weight.to(device=original_device, dtype=risk_points_w.dtype) * privileged_weight
        return filtered_weight.clamp(0.0, 1.0), dist

import torch
import torch.nn as nn

from OARM.utils.yopo_compat import ensure_yopo_path

ensure_yopo_path()
from loss.safety_loss import SafetyLoss


class ESDFCollisionLoss(nn.Module):
    """Variable-time collision cost using YOPO's ESDF query backend.

    This wrapper reuses YOPO's map loading and grid-sampling utility without
    editing YOPO files. It expects sampled trajectory points, so it is compatible
    with OARM's per-candidate trajectory time.
    """

    def __init__(self, device=None):
        super().__init__()
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.query_backend = SafetyLoss(torch.eye(6, device=device))

    def forward(self, sampled_pos_w: torch.Tensor, map_id: torch.Tensor) -> torch.Tensor:
        n, eval_points, _ = sampled_pos_w.shape
        if map_id.shape[0] != n:
            raise ValueError("map_id must be expanded to match sampled_pos_w candidates")
        cost, _ = self.query_backend.get_distance_cost(sampled_pos_w, map_id)
        return cost.reshape(n, eval_points).mean(dim=-1)

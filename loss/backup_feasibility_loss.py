import torch
import torch.nn as nn
import torch.nn.functional as F

from OARM.config import oarm_cfg


class StoppingFeasibilityLoss(nn.Module):
    """Stopping/yield feasibility supervision for OARM candidates.

    This is a learned visible-free-distance prior, not a certified backup
    trajectory guarantee. The legacy tensor name is kept for checkpoint
    compatibility with earlier OARM prototypes.
    """

    def __init__(self, no_backup_penalty: float = oarm_cfg.no_backup_penalty):
        super().__init__()
        self.no_backup_penalty = no_backup_penalty

    def forward(self, backup_logit: torch.Tensor, backup_feasible: torch.Tensor):
        backup_feasible = backup_feasible.reshape_as(backup_logit).float()
        loss = F.binary_cross_entropy_with_logits(backup_logit, backup_feasible)
        penalty = (1.0 - backup_feasible).detach() * self.no_backup_penalty
        return loss, penalty


def weak_backup_label_from_risk(occlusion_risk: torch.Tensor, threshold: float = oarm_cfg.backup_risk_threshold):
    """Early proxy: a candidate has yield margin if its lattice cell is not near an occlusion frontier."""

    return (occlusion_risk < threshold).float()


weak_yield_label_from_risk = weak_backup_label_from_risk


def stopping_distance(
    velocity: torch.Tensor,
    acc_max: float = oarm_cfg.yield_acc_max_mps2,
    safe_distance: float = oarm_cfg.yield_safe_distance_m,
    latency_s: float = oarm_cfg.yield_latency_s,
) -> torch.Tensor:
    speed = velocity.norm(dim=-1)
    return speed * latency_s + speed.square() / (2.0 * max(acc_max, 1e-3)) + safe_distance


def stopping_backup_label(
    end_velocity: torch.Tensor,
    visible_free_distance: torch.Tensor,
    acc_max: float = oarm_cfg.yield_acc_max_mps2,
    safe_distance: float = oarm_cfg.yield_safe_distance_m,
    latency_s: float = oarm_cfg.yield_latency_s,
):
    """Geometry proxy for stopping/yield feasibility labels.

    Args:
        end_velocity: [..., 3] candidate terminal velocity.
        visible_free_distance: [...] distance available in currently visible free space.
        acc_max: maximum braking acceleration.
        safe_distance: required clearance after stopping.
    """

    stop_distance = stopping_distance(end_velocity, acc_max=acc_max, safe_distance=safe_distance, latency_s=latency_s)
    return (stop_distance <= visible_free_distance).float()


stopping_yield_label = stopping_backup_label
BackupFeasibilityLoss = StoppingFeasibilityLoss
YieldFeasibilityLoss = StoppingFeasibilityLoss

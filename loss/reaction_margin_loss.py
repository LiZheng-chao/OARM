import torch
import torch.nn as nn
import torch.nn.functional as F

from OARM.config import oarm_cfg


class ReactionMarginLoss(nn.Module):
    def __init__(self, sigma: float = oarm_cfg.margin_sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, margin_pred: torch.Tensor, margin_label: torch.Tensor):
        margin_label = margin_label.reshape_as(margin_pred).float()
        margin_loss = F.smooth_l1_loss(margin_pred, margin_label)
        violation_cost = F.softplus(-margin_label / self.sigma).square()
        return margin_loss, violation_cost


def weak_margin_label_from_risk(traj_time: torch.Tensor, occlusion_risk: torch.Tensor):
    """Early weak label until privileged risk-point labels are available."""

    risk = occlusion_risk.reshape_as(traj_time).float().clamp(0.0, 1.0)
    return traj_time * (1.0 - risk) - oarm_cfg.reaction_time


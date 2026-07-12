from dataclasses import dataclass

import torch

from OARM.config import oarm_cfg


@dataclass
class RiskPointAssociation:
    associated_weight: torch.Tensor
    association_weight: torch.Tensor
    min_distance: torch.Tensor
    arrival_time: torch.Tensor
    valid_mask: torch.Tensor


def associate_risk_points_to_trajectory(
    sampled_pos_w: torch.Tensor,
    sampled_time: torch.Tensor,
    risk_points_w: torch.Tensor,
    risk_weight: torch.Tensor,
    sigma_m: float = oarm_cfg.risk_assoc_sigma_m,
    max_distance_m: float = oarm_cfg.risk_assoc_distance_m,
) -> RiskPointAssociation:
    """Candidate-aware risk weights based on closest approach distance.

    Args:
        sampled_pos_w: [N, T, 3] candidate trajectory samples.
        sampled_time: [N, T] sample times.
        risk_points_w: [N, Q, 3] risk points.
        risk_weight: [N, Q] global proposal/privileged risk weight.
    """

    dist = (sampled_pos_w[:, :, None, :] - risk_points_w[:, None, :, :]).norm(dim=-1)
    min_distance, closest_id = dist.min(dim=1)
    arrival_time = torch.gather(sampled_time, 1, closest_id)
    association_weight = torch.exp(-0.5 * (min_distance / max(sigma_m, 1e-3)).square())
    valid_mask = min_distance < max_distance_m
    association_weight = torch.where(valid_mask, association_weight, torch.zeros_like(association_weight))
    associated_weight = risk_weight.float() * association_weight
    return RiskPointAssociation(
        associated_weight=associated_weight,
        association_weight=association_weight,
        min_distance=min_distance,
        arrival_time=arrival_time,
        valid_mask=valid_mask,
    )


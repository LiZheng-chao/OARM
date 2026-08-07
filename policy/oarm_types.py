from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class OARMRawPrediction:
    """Raw per-lattice network outputs before physical decoding."""

    endstate_residual: torch.Tensor
    time_raw: torch.Tensor
    yaw_raw: torch.Tensor
    margin_raw: torch.Tensor
    risk_logit: torch.Tensor
    backup_logit: torch.Tensor
    utility_score: torch.Tensor


@dataclass
class OARMCandidate:
    """Decoded candidate set in body frame.

    Shapes follow YOPO's image-lattice convention unless noted:
    end_state_b: [B, 9, V, H]
    traj_time: [B, 1, V, H]
    yaw_terminal: [B, 1, V, H]
    margin_pred: [B, 1, V, H]
    risk_logit: [B, 1, V, H]
    backup_logit: [B, 1, V, H]
    utility_score: [B, V, H]
    candidate_type: [B, V, H]
    frontier_score: [B, V, H]
    """

    end_state_b: torch.Tensor
    traj_time: torch.Tensor
    yaw_terminal: torch.Tensor
    margin_pred: torch.Tensor
    risk_logit: torch.Tensor
    backup_logit: torch.Tensor
    utility_score: torch.Tensor
    candidate_type: torch.Tensor = None
    frontier_score: torch.Tensor = None
    time_anchor: torch.Tensor = None
    yaw_anchor: torch.Tensor = None
    utility_base: torch.Tensor = None
    utility_delta: torch.Tensor = None

    def flatten(self) -> Dict[str, torch.Tensor]:
        b, _, v, h = self.end_state_b.shape
        n = v * h
        flat = {
            "end_state_b": self.end_state_b.permute(0, 2, 3, 1).reshape(b * n, 9),
            "traj_time": self.traj_time.permute(0, 2, 3, 1).reshape(b * n),
            "yaw_terminal": self.yaw_terminal.permute(0, 2, 3, 1).reshape(b * n),
            "margin_pred": self.margin_pred.permute(0, 2, 3, 1).reshape(b * n),
            "risk_logit": self.risk_logit.permute(0, 2, 3, 1).reshape(b * n),
            "backup_logit": self.backup_logit.permute(0, 2, 3, 1).reshape(b * n),
            "yield_logit": self.backup_logit.permute(0, 2, 3, 1).reshape(b * n),
            "utility_score": self.utility_score.reshape(b * n),
        }
        if self.candidate_type is not None:
            flat["candidate_type"] = self.candidate_type.reshape(b * n)
        if self.frontier_score is not None:
            flat["frontier_score"] = self.frontier_score.reshape(b * n)
        if self.time_anchor is not None:
            flat["time_anchor"] = self.time_anchor.reshape(b * n)
        if self.yaw_anchor is not None:
            flat["yaw_anchor"] = self.yaw_anchor.reshape(b * n)
        if self.utility_base is not None:
            flat["utility_base"] = self.utility_base.reshape(b * n)
        if self.utility_delta is not None:
            flat["utility_delta"] = self.utility_delta.reshape(b * n)
        return flat

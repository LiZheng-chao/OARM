import torch
from torch import nn


def normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / 1.4142135623730951))


def risk_probability_from_window(
    reaction_window_mean: torch.Tensor,
    reaction_window_logvar: torch.Tensor,
    reaction_budget_s,
) -> torch.Tensor:
    budget = torch.as_tensor(
        reaction_budget_s,
        device=reaction_window_mean.device,
        dtype=reaction_window_mean.dtype,
    )
    while budget.dim() < reaction_window_mean.dim():
        budget = budget.unsqueeze(-1)
    sigma = torch.exp(0.5 * reaction_window_logvar).clamp_min(1e-3)
    z = (budget - reaction_window_mean) / sigma
    return normal_cdf(z).clamp(1e-6, 1.0 - 1e-6)


def risk_logit_from_window(
    reaction_window_mean: torch.Tensor,
    reaction_window_logvar: torch.Tensor,
    reaction_budget_s,
) -> torch.Tensor:
    prob = risk_probability_from_window(reaction_window_mean, reaction_window_logvar, reaction_budget_s)
    return torch.logit(prob)


class CandidateRMCritic(nn.Module):
    """Candidate-level probabilistic reaction-window critic.

    The critic predicts the reaction window distribution supplied by environment
    and candidate geometry. Online risk is computed later with the current
    reaction budget, so latency changes do not require retraining the critic.
    """

    def __init__(
        self,
        candidate_feature_dim: int,
        state_feature_dim: int = 0,
        geometry_feature_dim: int = 0,
        hidden_dim: int = 128,
        hazard_bins: int = 0,
    ):
        super().__init__()
        self.candidate_feature_dim = int(candidate_feature_dim)
        self.state_feature_dim = int(state_feature_dim)
        self.geometry_feature_dim = int(geometry_feature_dim)
        self.hazard_bins = int(hazard_bins)
        input_dim = self.candidate_feature_dim + self.state_feature_dim + self.geometry_feature_dim + 1
        output_dim = 4 + max(self.hazard_bins, 0)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        final = self.mlp[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        candidate_feature: torch.Tensor,
        state_feature: torch.Tensor = None,
        yopo_cost: torch.Tensor = None,
        candidate_geometry: torch.Tensor = None,
    ):
        if candidate_feature.dim() != 3:
            raise ValueError("candidate_feature must have shape [B, N, C]")
        b, n, c = candidate_feature.shape
        if c != self.candidate_feature_dim:
            raise ValueError(f"candidate feature dim mismatch: got {c}, expected {self.candidate_feature_dim}")
        pieces = [candidate_feature]
        if self.state_feature_dim > 0:
            if state_feature is None:
                raise ValueError("state_feature is required when state_feature_dim > 0")
            state_feature = state_feature.reshape(b, -1)
            if state_feature.shape[-1] != self.state_feature_dim:
                raise ValueError(f"state feature dim mismatch: got {state_feature.shape[-1]}, expected {self.state_feature_dim}")
            pieces.append(state_feature[:, None, :].expand(-1, n, -1))
        elif state_feature is not None:
            raise ValueError("state_feature was provided but state_feature_dim=0")
        if yopo_cost is None:
            yopo_cost = torch.zeros((b, n, 1), device=candidate_feature.device, dtype=candidate_feature.dtype)
        else:
            yopo_cost = yopo_cost.reshape(b, n, 1).to(device=candidate_feature.device, dtype=candidate_feature.dtype)
        pieces.append(yopo_cost)
        if self.geometry_feature_dim > 0:
            if candidate_geometry is None:
                raise ValueError("candidate_geometry is required when geometry_feature_dim > 0")
            candidate_geometry = candidate_geometry.reshape(b, n, -1).to(device=candidate_feature.device, dtype=candidate_feature.dtype)
            if candidate_geometry.shape[-1] != self.geometry_feature_dim:
                raise ValueError(f"geometry feature dim mismatch: got {candidate_geometry.shape[-1]}, expected {self.geometry_feature_dim}")
            pieces.append(candidate_geometry)
        elif candidate_geometry is not None:
            raise ValueError("candidate_geometry was provided but geometry_feature_dim=0")
        raw = self.mlp(torch.cat(pieces, dim=-1))
        out = {
            "reaction_window_mean": raw[..., 0],
            "reaction_window_logvar": raw[..., 1].clamp(min=-12.0, max=8.0),
            "insufficient_margin_logit": raw[..., 2],
            "validity_logit": raw[..., 3],
        }
        if self.hazard_bins > 0:
            out["hazard_logits"] = raw[..., 4 : 4 + self.hazard_bins]
        return out

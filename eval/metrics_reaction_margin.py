from typing import Dict

import torch


def reaction_margin_metrics(
    margin: torch.Tensor,
    threshold: float = 0.0,
    valid_mask: torch.Tensor = None,
) -> Dict[str, torch.Tensor]:
    margin = margin.float()
    finite = torch.isfinite(margin)
    if valid_mask is not None:
        finite = finite & valid_mask.reshape_as(margin).bool()
    zero = torch.zeros((), device=margin.device)
    if not bool(finite.any()):
        return {
            "reaction_margin_mean": zero,
            "reaction_margin_min": zero,
            "reaction_margin_violation_rate": zero,
            "negative_margin_rate": zero,
            "reaction_margin_finite_rate": zero,
        }
    margin = margin[finite]
    violation = margin < threshold
    return {
        "reaction_margin_mean": margin.mean(),
        "reaction_margin_min": margin.amin(),
        "reaction_margin_violation_rate": violation.float().mean(),
        "negative_margin_rate": violation.float().mean(),
        "reaction_margin_finite_rate": finite.float().mean(),
    }


def margin_prediction_metrics(
    pred_margin: torch.Tensor,
    label_margin: torch.Tensor,
    valid_mask: torch.Tensor = None,
) -> Dict[str, torch.Tensor]:
    pred_margin = pred_margin.reshape_as(label_margin).float()
    label_margin = label_margin.float()
    finite = torch.isfinite(pred_margin) & torch.isfinite(label_margin)
    if valid_mask is not None:
        finite = finite & valid_mask.reshape_as(label_margin).bool()
    zero = torch.zeros((), device=pred_margin.device)
    if not bool(finite.any()):
        return {
            "reaction_margin_mae": zero,
            "reaction_margin_rmse": zero,
            "negative_margin_recall": zero,
            "reaction_margin_prediction_finite_rate": zero,
        }
    pred_margin = pred_margin[finite]
    label_margin = label_margin[finite]
    error = pred_margin - label_margin
    return {
        "reaction_margin_mae": error.abs().mean(),
        "reaction_margin_rmse": error.square().mean().sqrt(),
        "negative_margin_recall": (((pred_margin < 0.0) & (label_margin < 0.0)).float().sum())
        / ((label_margin < 0.0).float().sum().clamp(min=1.0)),
        "reaction_margin_prediction_finite_rate": finite.float().mean(),
    }


def binary_auc(logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    label = label.reshape_as(logit).float()
    score = torch.sigmoid(logit.float()).reshape(-1)
    label = label.reshape(-1)
    pos = label > 0.5
    neg = ~pos
    pos_count = pos.float().sum()
    neg_count = neg.float().sum()
    if pos_count < 1 or neg_count < 1:
        return torch.zeros((), device=logit.device)
    order = torch.argsort(score)
    ranks = torch.empty_like(score)
    ranks[order] = torch.arange(1, score.numel() + 1, device=score.device, dtype=score.dtype)
    pos_rank_sum = ranks[pos].sum()
    return (pos_rank_sum - pos_count * (pos_count + 1.0) * 0.5) / (pos_count * neg_count).clamp(min=1.0)


def binary_ece(logit: torch.Tensor, label: torch.Tensor, bins: int = 10) -> torch.Tensor:
    label = label.reshape_as(logit).float()
    prob = torch.sigmoid(logit.float()).reshape(-1)
    label = label.reshape(-1)
    edges = torch.linspace(0.0, 1.0, bins + 1, device=prob.device)
    ece = torch.zeros((), device=prob.device)
    for i in range(bins):
        if i == bins - 1:
            mask = (prob >= edges[i]) & (prob <= edges[i + 1])
        else:
            mask = (prob >= edges[i]) & (prob < edges[i + 1])
        if bool(mask.any()):
            weight = mask.float().mean()
            ece = ece + weight * (prob[mask].mean() - label[mask].mean()).abs()
    return ece


def risk_calibration_metrics(risk_logit: torch.Tensor, risk_label: torch.Tensor) -> Dict[str, torch.Tensor]:
    risk_label = risk_label.reshape_as(risk_logit).float()
    prob = torch.sigmoid(risk_logit.float()).reshape_as(risk_label)
    pos = risk_label > 0.5
    neg = ~pos
    zero = torch.zeros((), device=risk_logit.device)
    return {
        "risk_auc": binary_auc(risk_logit, risk_label),
        "risk_ece": binary_ece(risk_logit, risk_label),
        "risk_label_positive_rate": pos.float().mean(),
        "risk_prob_mean": prob.mean(),
        "risk_prob_positive_mean": prob[pos].mean() if bool(pos.any()) else zero,
        "risk_prob_negative_mean": prob[neg].mean() if bool(neg.any()) else zero,
    }


def pairwise_ranking_accuracy(
    utility_score: torch.Tensor,
    margin_label: torch.Tensor,
    traj_num: int,
    margin_delta: float = 0.15,
    valid_mask: torch.Tensor = None,
) -> Dict[str, torch.Tensor]:
    if traj_num <= 1 or utility_score.numel() % traj_num != 0:
        zero = torch.zeros((), device=utility_score.device)
        return {"pairwise_ranking_accuracy": zero, "pairwise_ranking_pair_rate": zero, "pairwise_ranking_finite_rate": zero}
    batch_size = utility_score.numel() // traj_num
    utility = utility_score.reshape(batch_size, traj_num)
    margin = margin_label.reshape(batch_size, traj_num)
    finite = torch.isfinite(utility) & torch.isfinite(margin)
    if valid_mask is not None:
        finite = finite & valid_mask.reshape(batch_size, traj_num).bool()
    preference = (margin[:, :, None] - margin[:, None, :]) > margin_delta
    preference = preference & finite[:, :, None] & finite[:, None, :]
    if not bool(preference.any()):
        zero = torch.zeros((), device=utility_score.device)
        return {"pairwise_ranking_accuracy": zero, "pairwise_ranking_pair_rate": zero, "pairwise_ranking_finite_rate": zero}
    utility_delta = utility[:, :, None] - utility[:, None, :]
    return {
        "pairwise_ranking_accuracy": (utility_delta[preference] > 0.0).float().mean(),
        "pairwise_ranking_pair_rate": preference.float().mean(),
        "pairwise_ranking_finite_rate": finite.float().mean(),
    }

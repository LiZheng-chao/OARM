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
            "reaction_margin_bias": zero,
            "reaction_margin_corr": zero,
            "reaction_margin_spearman": zero,
            "reaction_margin_sign_accuracy": zero,
            "negative_margin_recall": zero,
            "reaction_margin_prediction_finite_rate": zero,
            "reaction_margin_valid_count": zero,
        }
    pred_margin_valid = pred_margin[finite]
    label_margin_valid = label_margin[finite]
    error = pred_margin_valid - label_margin_valid
    label_negative = label_margin_valid < 0.0
    pred_negative = pred_margin_valid < 0.0
    return {
        "reaction_margin_mae": error.abs().mean(),
        "reaction_margin_rmse": error.square().mean().sqrt(),
        "reaction_margin_bias": error.mean(),
        "reaction_margin_corr": _pearson_corr(pred_margin.reshape(-1), label_margin.reshape(-1), finite.reshape(-1)),
        "reaction_margin_spearman": _spearman_corr(pred_margin.reshape(-1), label_margin.reshape(-1), finite.reshape(-1)),
        "reaction_margin_sign_accuracy": ((pred_margin_valid >= 0.0) == (label_margin_valid >= 0.0)).float().mean(),
        "negative_margin_recall": ((pred_negative & label_negative).float().sum())
        / (label_negative.float().sum().clamp(min=1.0)),
        "reaction_margin_prediction_finite_rate": finite.float().mean(),
        "reaction_margin_valid_count": finite.float().sum(),
    }


def binary_auc_from_score(score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    score = score.float().reshape(-1)
    label = label.float().reshape(-1)
    pos = label > 0.5
    neg = ~pos
    pos_count = pos.float().sum()
    neg_count = neg.float().sum()
    if pos_count < 1 or neg_count < 1:
        return torch.zeros((), device=score.device)
    order = torch.argsort(score)
    ranks = torch.empty_like(score)
    ranks[order] = torch.arange(1, score.numel() + 1, device=score.device, dtype=score.dtype)
    pos_rank_sum = ranks[pos].sum()
    return (pos_rank_sum - pos_count * (pos_count + 1.0) * 0.5) / (pos_count * neg_count).clamp(min=1.0)


def binary_auc(logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    label = label.reshape_as(logit).float()
    score = torch.sigmoid(logit.float()).reshape(-1)
    return binary_auc_from_score(score, label.reshape(-1))


def binary_pr_auc_from_score(score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    score = score.float().reshape(-1)
    label = label.float().reshape(-1)
    pos = label > 0.5
    pos_count = pos.float().sum()
    if pos_count < 1:
        return torch.zeros((), device=score.device)
    order = torch.argsort(score, descending=True)
    sorted_pos = pos[order].float()
    tp = torch.cumsum(sorted_pos, dim=0)
    fp = torch.cumsum(1.0 - sorted_pos, dim=0)
    precision = tp / (tp + fp).clamp(min=1.0)
    recall = tp / pos_count.clamp(min=1.0)
    recall_prev = torch.cat((torch.zeros((1,), device=score.device, dtype=recall.dtype), recall[:-1]))
    return ((recall - recall_prev) * precision).sum()


def binary_balanced_accuracy(prob: torch.Tensor, label: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    prob = prob.float().reshape(-1)
    label = label.float().reshape(-1)
    pos = label > 0.5
    neg = ~pos
    pred_pos = prob >= threshold
    zero = torch.zeros((), device=prob.device)
    tpr = ((pred_pos & pos).float().sum() / pos.float().sum().clamp(min=1.0)) if bool(pos.any()) else zero
    tnr = (((~pred_pos) & neg).float().sum() / neg.float().sum().clamp(min=1.0)) if bool(neg.any()) else zero
    return 0.5 * (tpr + tnr)


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
    pred_pos = prob >= 0.5
    zero = torch.zeros((), device=risk_logit.device)
    return {
        "risk_auc": binary_auc(risk_logit, risk_label),
        "risk_auc_inverted": binary_auc_from_score(-prob.reshape(-1), risk_label.reshape(-1)),
        "risk_pr_auc": binary_pr_auc_from_score(prob.reshape(-1), risk_label.reshape(-1)),
        "risk_pr_auc_inverted": binary_pr_auc_from_score((-prob).reshape(-1), risk_label.reshape(-1)),
        "risk_ece": binary_ece(risk_logit, risk_label),
        "risk_label_positive_rate": pos.float().mean(),
        "risk_positive_count": pos.float().sum(),
        "risk_negative_count": neg.float().sum(),
        "risk_balanced_accuracy_05": binary_balanced_accuracy(prob, risk_label, threshold=0.5),
        "risk_positive_recall_05": ((pred_pos & pos).float().sum() / pos.float().sum().clamp(min=1.0)) if bool(pos.any()) else zero,
        "risk_negative_recall_05": (((~pred_pos) & neg).float().sum() / neg.float().sum().clamp(min=1.0)) if bool(neg.any()) else zero,
        "risk_prob_mean": prob.mean(),
        "risk_prob_positive_mean": prob[pos].mean() if bool(pos.any()) else zero,
        "risk_prob_negative_mean": prob[neg].mean() if bool(neg.any()) else zero,
    }


def pairwise_ranking_accuracy(
    utility_score: torch.Tensor,
    margin_label: torch.Tensor,
    traj_num: int,
    margin_delta: float = 0.10,
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


def matched_pairwise_ranking_accuracy(
    utility_score: torch.Tensor,
    margin_label: torch.Tensor,
    traj_num: int,
    margin_delta: float = 0.10,
    valid_mask: torch.Tensor = None,
    progress: torch.Tensor = None,
    base_cost: torch.Tensor = None,
    mean_speed: torch.Tensor = None,
    traj_time: torch.Tensor = None,
    progress_eps: float = 0.60,
    base_cost_eps: float = 1.50,
    speed_eps: float = 0.75,
    time_eps: float = 0.35,
) -> Dict[str, torch.Tensor]:
    if traj_num <= 1 or utility_score.numel() % traj_num != 0:
        zero = torch.zeros((), device=utility_score.device)
        return {'matched_pairwise_ranking_accuracy': zero, 'matched_pairwise_ranking_pair_rate': zero, 'matched_pairwise_ranking_finite_rate': zero}
    batch_size = utility_score.numel() // traj_num
    utility = utility_score.reshape(batch_size, traj_num)
    margin = margin_label.reshape(batch_size, traj_num)
    finite = torch.isfinite(utility) & torch.isfinite(margin)
    if valid_mask is not None:
        finite = finite & valid_mask.reshape(batch_size, traj_num).bool()
    comparable = torch.ones((batch_size, traj_num, traj_num), device=utility_score.device, dtype=torch.bool)
    if progress is not None:
        progress = progress.reshape(batch_size, traj_num)
        finite = finite & torch.isfinite(progress)
        comparable = comparable & ((progress[:, :, None] - progress[:, None, :]).abs() < progress_eps)
    if base_cost is not None:
        base_cost = base_cost.reshape(batch_size, traj_num)
        finite = finite & torch.isfinite(base_cost)
        comparable = comparable & ((base_cost[:, :, None] - base_cost[:, None, :]).abs() < base_cost_eps)
    if mean_speed is not None:
        mean_speed = mean_speed.reshape(batch_size, traj_num)
        finite = finite & torch.isfinite(mean_speed)
        comparable = comparable & ((mean_speed[:, :, None] - mean_speed[:, None, :]).abs() < speed_eps)
    if traj_time is not None:
        traj_time = traj_time.reshape(batch_size, traj_num)
        finite = finite & torch.isfinite(traj_time)
        comparable = comparable & ((traj_time[:, :, None] - traj_time[:, None, :]).abs() < time_eps)
    preference = (margin[:, :, None] - margin[:, None, :]) > margin_delta
    pair_mask = comparable & preference & finite[:, :, None] & finite[:, None, :]
    if not bool(pair_mask.any()):
        zero = torch.zeros((), device=utility_score.device)
        return {'matched_pairwise_ranking_accuracy': zero, 'matched_pairwise_ranking_pair_rate': zero, 'matched_pairwise_ranking_finite_rate': finite.float().mean()}
    utility_delta = utility[:, :, None] - utility[:, None, :]
    return {
        'matched_pairwise_ranking_accuracy': (utility_delta[pair_mask] > 0.0).float().mean(),
        'matched_pairwise_ranking_pair_rate': pair_mask.float().mean(),
        'matched_pairwise_ranking_finite_rate': finite.float().mean(),
    }


def _pearson_corr(a: torch.Tensor, b: torch.Tensor, valid: torch.Tensor):
    mask = valid & torch.isfinite(a) & torch.isfinite(b)
    if not bool(mask.any()) or int(mask.float().sum().item()) < 2:
        return torch.zeros((), device=a.device)
    a = a[mask].float()
    b = b[mask].float()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.square().mean().sqrt() * b.square().mean().sqrt()
    if float(denom.detach().cpu()) <= 1e-8:
        return torch.zeros((), device=a.device)
    return (a * b).mean() / denom.clamp(min=1e-8)


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(values.numel(), device=values.device, dtype=values.dtype)
    return ranks


def _spearman_corr(a: torch.Tensor, b: torch.Tensor, valid: torch.Tensor):
    mask = valid & torch.isfinite(a) & torch.isfinite(b)
    if not bool(mask.any()) or int(mask.float().sum().item()) < 2:
        return torch.zeros((), device=a.device)
    return _pearson_corr(_rankdata(a[mask].float()), _rankdata(b[mask].float()), torch.ones_like(a[mask], dtype=torch.bool))

def margin_disentanglement_metrics(margin: torch.Tensor, utility: torch.Tensor, valid_mask: torch.Tensor = None, frontier_score: torch.Tensor = None, duration: torch.Tensor = None, traj_time: torch.Tensor = None, progress: torch.Tensor = None) -> Dict[str, torch.Tensor]:
    margin = margin.float().reshape(-1)
    utility = utility.reshape_as(margin).float()
    valid = torch.isfinite(margin) & torch.isfinite(utility)
    if valid_mask is not None:
        valid = valid & valid_mask.reshape_as(margin).bool()
    out = {
        'utility_margin_corr': _pearson_corr(utility, margin, valid),
        'disentanglement_valid_rate': valid.float().mean(),
    }
    if frontier_score is not None:
        frontier_score = frontier_score.reshape_as(margin).float()
        out['margin_frontier_corr'] = _pearson_corr(margin, frontier_score, valid)
        out['utility_frontier_corr'] = _pearson_corr(utility, frontier_score, valid)
    if duration is None:
        duration = traj_time
    if duration is not None:
        duration = duration.reshape_as(margin).float()
        out['margin_duration_corr'] = _pearson_corr(margin, duration, valid)
        out['utility_duration_corr'] = _pearson_corr(utility, duration, valid)
    if progress is not None:
        progress = progress.reshape_as(margin).float()
        out['margin_progress_corr'] = _pearson_corr(margin, progress, valid)
        out['utility_progress_corr'] = _pearson_corr(utility, progress, valid)
    return out

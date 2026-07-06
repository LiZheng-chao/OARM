from typing import Dict

import torch


def backup_feasibility_metrics(
    backup_logit: torch.Tensor,
    backup_label: torch.Tensor,
    reaction_margin: torch.Tensor = None,
    margin_threshold: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Compute OARM stopping/yield feasibility metrics.

    Names keep the historical backup prefix so existing logs do not break.
    """

    backup_pred = (torch.sigmoid(backup_logit) >= 0.5).float()
    backup_label = backup_label.reshape_as(backup_pred).float()
    feasible_rate = backup_pred.mean()
    label_rate = backup_label.mean()
    accuracy = (backup_pred == backup_label).float().mean()
    false_safe_rate = ((backup_pred > 0.5) & (backup_label < 0.5)).float().mean()

    metrics = {
        "backup_feasibility_rate": feasible_rate,
        "backup_label_rate": label_rate,
        "backup_accuracy": accuracy,
        "backup_false_safe_rate": false_safe_rate,
        "yield_feasibility_rate": feasible_rate,
        "yield_label_rate": label_rate,
        "yield_accuracy": accuracy,
        "yield_false_safe_rate": false_safe_rate,
    }

    if reaction_margin is not None:
        reaction_margin = reaction_margin.reshape_as(backup_pred)
        violation = (reaction_margin < margin_threshold) | (backup_label < 0.5)
        metrics["backup_margin_violation_rate"] = violation.float().mean()
        metrics["yield_margin_violation_rate"] = metrics["backup_margin_violation_rate"]

    return metrics

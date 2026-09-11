import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import torch

from OARM.eval.check_episode_splits import check_episode_splits
from OARM.policy.oarm_risk_calibrator import (
    TemperatureCalibration,
    binary_calibration_metrics,
    fit_conformal_slack,
    fit_temperature_scaling,
    reliability_bins,
)


CANDIDATE_LABEL_KEYS = (
    "insufficient_reaction_gt",
    "rm_violation_gt",
    "selected_rm_violation_gt",
    "reaction_window_lt_budget",
)
REACTION_WINDOW_KEYS = ("reaction_window_gt", "reaction_window", "selected_reaction_window_gt", "rm_reaction_window_gt")
REACTION_BUDGET_KEYS = ("reaction_budget_s", "latency_tau_total_s", "tau_total_s", "reaction_budget", "selected_reaction_budget_s")
EPISODE_LEVEL_LABEL_KEYS = {"collision", "collision_flag", "success", "success_flag", "arrive"}
GENERIC_LABEL_KEYS = {"risk_label", "selected_risk_label", "label"}
TWO_STAGE_RISK_KEYS = (
    "hazard_risk_prob",
    "selected_hazard_risk_prob",
    "two_stage_risk_prob",
    "selected_two_stage_risk_prob",
)
AUTO_RAW_RISK_KEYS = TWO_STAGE_RISK_KEYS + (
    "raw_risk_prob",
    "selected_raw_risk_prob",
    "risk_logit_prob",
    "selected_risk_logit_prob",
    "risk_prob",
    "selected_risk_prob",
)
AUTO_FUSED_RISK_KEYS = TWO_STAGE_RISK_KEYS + (
    "validity_fused_risk_prob",
    "selected_validity_fused_risk_prob",
    "raw_risk_prob",
    "selected_raw_risk_prob",
    "risk_prob",
    "selected_risk_prob",
)
AUTO_VALIDITY_KEYS = ("validity_prob", "selected_validity_prob")


def _first_number_with_key(row: Dict, keys: Iterable[str]) -> Tuple[Optional[str], Optional[float]]:
    for key in keys:
        if key in row and row[key] is not None:
            try:
                value = float(row[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return key, value
    return None, None


def _first_number(row: Dict, keys: Iterable[str]) -> Optional[float]:
    _, value = _first_number_with_key(row, keys)
    return value


def _first_label(row: Dict, keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        if key not in row or row[key] is None:
            continue
        value = row[key]
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return 1.0 if value >= 0.5 else 0.0
    return None


def _iter_rows(paths: Iterable[Path]) -> Iterator[Dict]:
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no} is not valid JSONL") from exc


def _iter_records(row: Dict, include_candidates: bool) -> Iterator[Dict]:
    if include_candidates and isinstance(row.get("candidates"), list):
        parent = {key: value for key, value in row.items() if key != "candidates"}
        for cand in row["candidates"]:
            if isinstance(cand, dict):
                merged = dict(parent)
                merged.update(cand)
                yield merged
    else:
        yield row


def _extract_arrays(
    paths: Iterable[Path],
    label_key: str,
    risk_key: Optional[str],
    validity_key: Optional[str],
    use_validity_fusion: bool,
    validity_unknown_risk: float,
    include_candidates: bool,
    split_key: str,
    calibration_split: str,
    episode_key: str,
    require_split: bool,
    require_episode_id: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    labels: List[float] = []
    risks: List[float] = []
    budgets: List[float] = []
    if not label_key:
        raise ValueError("risk calibration requires an explicit candidate-level --label-key, e.g. insufficient_reaction_gt")
    if label_key in GENERIC_LABEL_KEYS:
        raise ValueError(f"{label_key!r} is too generic for calibration; use an explicit candidate-level RM label")
    if include_candidates and label_key in EPISODE_LEVEL_LABEL_KEYS:
        raise ValueError(f"{label_key!r} is episode-level and cannot be used for candidate-level calibration")
    stats = {
        "records_seen": 0,
        "records_used": 0,
        "missing_label": 0,
        "missing_risk": 0,
        "missing_validity": 0,
        "missing_split": 0,
        "non_calibration_split": 0,
        "missing_episode_id": 0,
        "episode_count": 0,
        "missing_reaction_window": 0,
        "missing_reaction_budget": 0,
        "missing_reaction_budget_diagnostic": 0,
        "no_entry_negative_labels": 0,
        "validity_fusion_skipped_two_stage": 0,
    }
    risk_keys = (risk_key,) if risk_key else (AUTO_RAW_RISK_KEYS if use_validity_fusion else AUTO_FUSED_RISK_KEYS)
    validity_keys = (validity_key,) if validity_key else AUTO_VALIDITY_KEYS
    episode_ids = set()

    for row in _iter_rows(paths):
        for record in _iter_records(row, include_candidates):
            stats["records_seen"] += 1
            split_value = record.get(split_key)
            if split_value is None:
                stats["missing_split"] += 1
                if require_split:
                    continue
            elif str(split_value) != str(calibration_split):
                stats["non_calibration_split"] += 1
                if require_split:
                    continue
            episode_id = record.get(episode_key)
            if episode_id is None:
                stats["missing_episode_id"] += 1
                if require_episode_id:
                    continue
            else:
                episode_ids.add(str(episode_id))

            label_budget = None
            if label_key == "reaction_window_lt_budget":
                no_entry = _first_label(record, ("rm_no_entry_gt",))
                if no_entry == 1.0:
                    label = 0.0
                    stats["no_entry_negative_labels"] += 1
                else:
                    window = _first_number(record, REACTION_WINDOW_KEYS)
                    label_budget = _first_number(record, REACTION_BUDGET_KEYS)
                    if window is None:
                        stats["missing_reaction_window"] += 1
                    if label_budget is None:
                        stats["missing_reaction_budget"] += 1
                    if window is None or label_budget is None:
                        stats["missing_label"] += 1
                        continue
                    label = 1.0 if window < label_budget else 0.0
            else:
                label = _first_label(record, (label_key,))
                if label is None:
                    stats["missing_label"] += 1
                    continue

            risk_key_used, risk = _first_number_with_key(record, risk_keys)
            if risk is None:
                stats["missing_risk"] += 1
                continue
            risk = min(max(risk, 0.0), 1.0)
            skip_validity_fusion = risk_key_used in TWO_STAGE_RISK_KEYS
            if use_validity_fusion and skip_validity_fusion:
                stats["validity_fusion_skipped_two_stage"] += 1
            elif use_validity_fusion:
                validity = _first_number(record, validity_keys)
                if validity is None:
                    stats["missing_validity"] += 1
                    continue
                validity = min(max(validity, 0.0), 1.0)
                risk = validity * risk + (1.0 - validity) * float(validity_unknown_risk)

            if label_budget is None:
                label_budget = _first_number(record, REACTION_BUDGET_KEYS)
            if label_budget is None:
                stats["missing_reaction_budget_diagnostic"] += 1
                budgets.append(math.nan)
            else:
                budgets.append(float(label_budget))
            labels.append(label)
            risks.append(risk)
            stats["records_used"] += 1

    stats["episode_count"] = len(episode_ids)
    if require_split and stats["missing_split"] > 0:
        raise ValueError(f"calibration rows missing split key {split_key!r}; pass --allow-missing-split only for ad-hoc smoke checks; stats={stats}")
    if require_split and stats["non_calibration_split"] > 0:
        raise ValueError(f"input contains non-calibration split rows; provide a clean calibration split; stats={stats}")
    if require_episode_id and stats["missing_episode_id"] > 0:
        raise ValueError(f"calibration rows missing episode id key {episode_key!r}; stats={stats}")
    if not risks:
        raise ValueError(f"no usable calibration records found; stats={stats}")

    probs = torch.tensor(risks, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    tau_s = torch.tensor(budgets, dtype=torch.float32)
    positive_count = int((y >= 0.5).sum().item())
    stats["positive_label_count"] = positive_count
    stats["negative_label_count"] = int(y.numel()) - positive_count
    stats["positive_label_rate"] = float(positive_count) / float(y.numel())
    return probs, y, tau_s, stats


def _metrics_by_tau_bin(
    before_probs: torch.Tensor,
    after_probs: torch.Tensor,
    labels: torch.Tensor,
    tau_s: torch.Tensor,
    n_bins: int,
) -> List[Dict[str, Any]]:
    edges = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, math.inf)
    rows: List[Dict[str, Any]] = []
    finite_tau = torch.isfinite(tau_s)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = finite_tau & (tau_s >= lower)
        if math.isfinite(upper):
            mask = mask & (tau_s < upper)
        count = int(mask.sum().item())
        before = binary_calibration_metrics(before_probs[mask], labels[mask], n_bins=n_bins)
        after = binary_calibration_metrics(after_probs[mask], labels[mask], n_bins=n_bins)
        rows.append(
            {
                "bin": index,
                "tau_lower_s": float(lower),
                "tau_upper_s": None if not math.isfinite(upper) else float(upper),
                "sample_count": count,
                "positive_label_rate": None if count == 0 else float(labels[mask].mean().item()),
                "metrics_before": before.__dict__,
                "metrics_after_platt": after.__dict__,
            }
        )
    return rows


def _write_reliability_diagram(path: str, bins: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required when --reliability-output is set") from exc

    populated = [row for row in bins if row["count"] > 0]
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="0.45", label="ideal")
    if populated:
        ax.plot(
            [row["confidence"] for row in populated],
            [row["empirical_rate"] for row in populated],
            marker="o",
            linewidth=1.5,
            label="calibrated",
        )
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0), xlabel="Predicted risk", ylabel="Empirical positive rate")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)

def _logit_from_prob(probabilities: torch.Tensor) -> torch.Tensor:
    probs = torch.clamp(probabilities.float(), min=1e-6, max=1.0 - 1e-6)
    return torch.logit(probs)


def fit_calibration_from_jsonl(
    inputs: Iterable[str],
    output: str,
    label_key: Optional[str] = None,
    risk_key: Optional[str] = None,
    validity_key: Optional[str] = None,
    use_validity_fusion: bool = True,
    validity_unknown_risk: float = 0.5,
    empirical_upper_alpha: float = 0.1,
    n_bins: int = 15,
    max_iter: int = 100,
    include_candidates: bool = True,
    split_key: str = "split",
    calibration_split: str = "calibration",
    episode_key: str = "episode_id",
    require_split: bool = True,
    require_episode_id: bool = True,
    train_manifest: Optional[str] = None,
    val_manifest: Optional[str] = None,
    calibration_manifest: Optional[str] = None,
    test_manifest: Optional[str] = None,
    ignore_map_overlap: bool = False,
    reliability_output: Optional[str] = None,
) -> Dict:
    paths = [Path(p) for p in inputs]
    split_check = None
    manifests = {
        "train": train_manifest,
        "val": val_manifest,
        "calibration": calibration_manifest,
        "test": test_manifest,
    }
    if any(manifests.values()):
        missing = [name for name, value in manifests.items() if not value]
        if missing:
            raise ValueError(f"split manifest check requires all four manifests; missing={missing}")
        split_check = check_episode_splits(
            {name: value for name, value in manifests.items() if value},
            episode_key=episode_key,
            check_maps=not ignore_map_overlap,
        )
    probs, labels, tau_s, stats = _extract_arrays(
        paths,
        label_key=label_key,
        risk_key=risk_key,
        validity_key=validity_key,
        use_validity_fusion=use_validity_fusion,
        validity_unknown_risk=validity_unknown_risk,
        include_candidates=include_candidates,
        split_key=split_key,
        calibration_split=calibration_split,
        episode_key=episode_key,
        require_split=require_split,
        require_episode_id=require_episode_id,
    )
    logits = _logit_from_prob(probs)
    before = binary_calibration_metrics(probs, labels, n_bins=n_bins)
    calibration = fit_temperature_scaling(logits, labels, max_iter=max_iter)
    calibrated = torch.sigmoid(
        logits / max(float(calibration.temperature), 1e-6) + float(calibration.bias)
    )
    after = binary_calibration_metrics(calibrated, labels, n_bins=n_bins)
    slack = fit_conformal_slack(
        calibrated,
        labels,
        alpha=empirical_upper_alpha,
        n_bins=n_bins,
    )
    calibration.conformal_slack = float(slack)
    calibration.fitted_on = ",".join(str(p) for p in paths)
    upper = torch.clamp(calibrated + float(slack), min=0.0, max=1.0)
    upper_metrics = binary_calibration_metrics(upper, labels, n_bins=n_bins)
    reliability_rows = reliability_bins(calibrated, labels, n_bins=n_bins)
    tau_metrics = _metrics_by_tau_bin(probs, calibrated, labels, tau_s, n_bins=n_bins)

    payload = calibration.state_dict()
    payload.update(
        {
            "calibration_version": "platt_empirical_upper_v2",
            "empirical_upper_alpha": float(empirical_upper_alpha),
            "empirical_upper_method": "binned_hoeffding_gap_v1",
            "validity_fusion": bool(use_validity_fusion),
            "validity_unknown_risk": float(validity_unknown_risk),
            "label_key": label_key,
            "split_key": split_key,
            "calibration_split": calibration_split,
            "episode_key": episode_key,
            "require_split": bool(require_split),
            "require_episode_id": bool(require_episode_id),
            "split_manifest_check": split_check,
            "sample_count": int(labels.numel()),
            "positive_label_count": int(stats["positive_label_count"]),
            "negative_label_count": int(stats["negative_label_count"]),
            "positive_label_rate": float(stats["positive_label_rate"]),
            "tau_diagnostic_sample_count": int(torch.isfinite(tau_s).sum().item()),
            "metrics_before": before.__dict__,
            "metrics_after_platt": after.__dict__,
            "metrics_after_temperature": after.__dict__,
            "metrics_after_empirical_upper": upper_metrics.__dict__,
            "reliability_bins_after_platt": reliability_rows,
            "reliability_bins_after_temperature": reliability_rows,
            "metrics_by_tau_bin": tau_metrics,
            "input_stats": stats,
            "note": "conformal_slack is a binned one-sided empirical calibration-gap bound, not a formal conformal guarantee.",
        }
    )
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    if reliability_output:
        _write_reliability_diagram(reliability_output, reliability_rows)
    return payload


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fit OARM risk temperature calibration and empirical conservative upper slack from JSONL logs.")
    p.add_argument("--input", nargs="+", required=True, help="calibration JSONL file(s); rows may contain a candidates array")
    p.add_argument("--output", required=True, help="output calibration JSON path")
    p.add_argument("--reliability-output", default=None, help="optional reliability diagram PNG path")
    p.add_argument("--label-key", required=True, choices=CANDIDATE_LABEL_KEYS, help="candidate-level calibration label; use reaction_window_lt_budget to derive y=1[reaction_window < tau]")
    p.add_argument("--risk-key", default=None, help="override risk probability key; default uses raw risk when validity fusion is enabled")
    p.add_argument("--validity-key", default=None, help="override validity probability key")
    validity = p.add_mutually_exclusive_group()
    validity.add_argument("--use-validity-fusion", dest="use_validity_fusion", action="store_true", default=True)
    validity.add_argument("--disable-validity-fusion", dest="use_validity_fusion", action="store_false")
    p.add_argument("--validity-unknown-risk", type=float, default=0.5)
    p.add_argument("--empirical-upper-alpha", type=float, default=0.1, help="quantile tail for conservative risk upper slack")
    p.add_argument("--n-bins", type=int, default=15)
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--selected-only", action="store_true", help="fit on top-level selected row fields instead of per-candidate table")
    p.add_argument("--split-key", default="split")
    p.add_argument("--calibration-split", default="calibration")
    p.add_argument("--episode-key", default="episode_id")
    p.add_argument("--allow-missing-split", action="store_true", help="ad-hoc smoke only; formal calibration should keep split metadata")
    p.add_argument("--allow-missing-episode-id", action="store_true", help="ad-hoc smoke only; formal calibration should keep episode metadata")
    p.add_argument("--train-manifest", default=None)
    p.add_argument("--val-manifest", default=None)
    p.add_argument("--calibration-manifest", default=None)
    p.add_argument("--test-manifest", default=None)
    p.add_argument("--ignore-map-overlap", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    payload = fit_calibration_from_jsonl(
        args.input,
        args.output,
        label_key=args.label_key,
        risk_key=args.risk_key,
        validity_key=args.validity_key,
        use_validity_fusion=args.use_validity_fusion,
        validity_unknown_risk=args.validity_unknown_risk,
        empirical_upper_alpha=args.empirical_upper_alpha,
        n_bins=args.n_bins,
        max_iter=args.max_iter,
        include_candidates=not args.selected_only,
        split_key=args.split_key,
        calibration_split=args.calibration_split,
        episode_key=args.episode_key,
        require_split=not args.allow_missing_split,
        require_episode_id=not args.allow_missing_episode_id,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        calibration_manifest=args.calibration_manifest,
        test_manifest=args.test_manifest,
        ignore_map_overlap=args.ignore_map_overlap,
        reliability_output=args.reliability_output,
    )
    summary = {
        "output": args.output,
        "sample_count": payload["sample_count"],
        "positive_label_rate": payload["positive_label_rate"],
        "tau_diagnostic_sample_count": payload["tau_diagnostic_sample_count"],
        "reliability_output": args.reliability_output,
        "temperature": payload["temperature"],
        "bias": payload["bias"],
        "empirical_conservative_upper_slack": payload["conformal_slack"],
        "metrics_before": payload["metrics_before"],
        "metrics_after_platt": payload["metrics_after_platt"],
        "metrics_after_empirical_upper": payload["metrics_after_empirical_upper"],
        "input_stats": payload["input_stats"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

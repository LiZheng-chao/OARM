import json
import math
from dataclasses import asdict, dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F


@dataclass
class CalibrationMetrics:
    sample_count: int
    brier: float
    ece: float
    mce: float
    nll: float


@dataclass
class TemperatureCalibration:
    temperature: float = 1.0
    bias: float = 0.0
    conformal_slack: float = 0.0
    fitted_on: str = "calibration"

    def state_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_file(cls, path: str) -> "TemperatureCalibration":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        allowed = set(cls.__dataclass_fields__.keys())
        return cls(**{key: value for key, value in payload.items() if key in allowed})

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f, indent=2, sort_keys=True)


def _tensor(values) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.detach().float().reshape(-1)
    if isinstance(values, (float, int)):
        return torch.tensor([values], dtype=torch.float32)
    return torch.tensor(list(values), dtype=torch.float32).reshape(-1)


def calibrated_probability(logits, calibration: TemperatureCalibration) -> torch.Tensor:
    temperature = max(float(calibration.temperature), 1e-6)
    return torch.sigmoid(_tensor(logits) / temperature + float(calibration.bias))


def risk_upper_bound(probabilities, calibration: TemperatureCalibration, max_prob: float = 1.0) -> torch.Tensor:
    probs = _tensor(probabilities)
    return torch.clamp(probs + max(float(calibration.conformal_slack), 0.0), min=0.0, max=float(max_prob))


def binary_calibration_metrics(probabilities, labels, n_bins: int = 15) -> CalibrationMetrics:
    probs = torch.clamp(_tensor(probabilities), min=1e-6, max=1.0 - 1e-6)
    y = _tensor(labels)
    if probs.numel() != y.numel():
        raise ValueError(f"probabilities/labels size mismatch: {probs.numel()} vs {y.numel()}")
    if probs.numel() == 0:
        return CalibrationMetrics(sample_count=0, brier=math.nan, ece=math.nan, mce=math.nan, nll=math.nan)
    y = y.clamp(0.0, 1.0)
    brier = torch.mean((probs - y) ** 2)
    nll = F.binary_cross_entropy(probs, y)
    ece = torch.tensor(0.0)
    mce = torch.tensor(0.0)
    edges = torch.linspace(0.0, 1.0, int(n_bins) + 1, device=probs.device)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs <= hi) if hi >= 1.0 else (probs >= lo) & (probs < hi)
        if not bool(mask.any()):
            continue
        gap = torch.abs(probs[mask].mean() - y[mask].mean())
        ece = ece + gap * mask.float().mean()
        mce = torch.maximum(mce, gap)
    return CalibrationMetrics(int(probs.numel()), float(brier.detach().cpu()), float(ece.detach().cpu()), float(mce.detach().cpu()), float(nll.detach().cpu()))


def fit_temperature_scaling(logits, labels, max_iter: int = 100, initial_temperature: float = 1.0) -> TemperatureCalibration:
    x = _tensor(logits)
    y = _tensor(labels).clamp(0.0, 1.0)
    if x.numel() != y.numel():
        raise ValueError(f"logits/labels size mismatch: {x.numel()} vs {y.numel()}")
    if x.numel() == 0:
        return TemperatureCalibration(temperature=float(initial_temperature))
    log_temp = torch.tensor([math.log(max(float(initial_temperature), 1e-6))], requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temp, bias], lr=0.1, max_iter=int(max_iter), line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        temp = torch.exp(log_temp).clamp(min=1e-6, max=1e6)
        loss = F.binary_cross_entropy_with_logits(x / temp + bias, y)
        loss.backward()
        return loss
    optimizer.step(closure)
    return TemperatureCalibration(
        temperature=float(torch.exp(log_temp).detach().cpu()),
        bias=float(bias.detach().cpu()),
    )


def fit_conformal_slack(probabilities, labels, alpha: float = 0.1, n_bins: int = 15) -> float:
    probs = torch.clamp(_tensor(probabilities), min=0.0, max=1.0)
    y = _tensor(labels).clamp(0.0, 1.0)
    if probs.numel() != y.numel():
        raise ValueError(f"probabilities/labels size mismatch: {probs.numel()} vs {y.numel()}")
    if probs.numel() == 0:
        return 0.0

    alpha = min(max(float(alpha), 1e-6), 1.0)
    n_bins = max(int(n_bins), 1)
    min_bin_count = max(30, int(math.ceil(probs.numel() / (n_bins * 20.0))))
    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    upper_gaps = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs <= hi) if hi >= 1.0 else (probs >= lo) & (probs < hi)
        count = int(mask.sum().detach().cpu())
        if count < min_bin_count:
            continue
        calibration_gap = y[mask].mean() - probs[mask].mean()
        radius = math.sqrt(math.log(max(n_bins / alpha, 1.0)) / (2.0 * count))
        upper_gaps.append(float(calibration_gap.detach().cpu()) + radius)

    if not upper_gaps:
        calibration_gap = float((y.mean() - probs.mean()).detach().cpu())
        radius = math.sqrt(math.log(1.0 / alpha) / (2.0 * probs.numel()))
        upper_gaps.append(calibration_gap + radius)
    return min(max(max(upper_gaps), 0.0), 1.0)


def reliability_bins(probabilities, labels, n_bins: int = 15) -> List[Dict[str, float]]:
    probs = torch.clamp(_tensor(probabilities), min=0.0, max=1.0)
    y = _tensor(labels).clamp(0.0, 1.0)
    edges = torch.linspace(0.0, 1.0, int(n_bins) + 1, device=probs.device)
    rows = []
    for idx, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (probs >= lo) & (probs <= hi) if hi >= 1.0 else (probs >= lo) & (probs < hi)
        count = int(mask.sum().detach().cpu())
        rows.append({"bin": idx, "lo": float(lo.detach().cpu()), "hi": float(hi.detach().cpu()), "count": count, "confidence": float(probs[mask].mean().detach().cpu()) if count else None, "empirical_rate": float(y[mask].mean().detach().cpu()) if count else None})
    return rows

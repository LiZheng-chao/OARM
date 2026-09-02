from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


KEEP_LOW_RISK = "KEEP_LOW_RISK"
KEEP_GRAY_NO_RISK_IMPROVEMENT = "KEEP_GRAY_NO_RISK_IMPROVEMENT"
RERANK_TOP1_UNSAFE = "RERANK_TOP1_UNSAFE"
PROBE_VISIBILITY_GAIN = "PROBE_VISIBILITY_GAIN"
BRAKE_NO_SAFE_CANDIDATE = "BRAKE_NO_SAFE_CANDIDATE"
BRAKE_HIGH_UNCERTAINTY = "BRAKE_HIGH_UNCERTAINTY"
BRAKE_LATENCY_SPIKE = "BRAKE_LATENCY_SPIKE"
BRAKE_LATCH_HOLD = "BRAKE_LATCH_HOLD"
NO_VERIFIED_SAFE_ACTION = "NO_VERIFIED_SAFE_ACTION"


@dataclass
class InterventionSelectorConfig:
    delta_keep: float = 0.10
    delta_safe: float = 0.20
    delta_probe: float = 0.25
    lambda_deviation: float = 0.25
    lambda_risk: float = 1.0
    risk_improvement_min: float = 0.02
    lambda_probe_risk: float = 1.0
    lambda_probe_margin_gain: float = 0.5
    lambda_probe_time: float = 0.05
    min_probe_margin_gain_s: float = 0.05


@dataclass
class InterventionSelection:
    selected_index: Optional[int]
    intervention_type: str
    intervention_reason: str
    risk_before: Optional[float]
    risk_after: Optional[float]
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BrakeLatchConfig:
    enabled: bool = True
    min_hold_s: float = 0.6
    release_speed_mps: float = 0.25
    release_frames: int = 3
    release_risk: float = 0.10
    require_release_evidence: bool = True


class BrakeInterventionLatch:
    """Hold a selected brake long enough to stop before accepting a new trajectory."""

    def __init__(self, config: BrakeLatchConfig = None):
        self.config = config or BrakeLatchConfig()
        self.active = False
        self.hold_until_s = 0.0
        self.safe_release_frames = 0

    def update(
        self,
        decision: InterventionSelection,
        now_s: float,
        speed_mps: float,
        selected_admissible: bool,
        brake_duration_s: float,
        brake_risk_upper_bound: Optional[float],
        release_evidence: bool = True,
    ) -> InterventionSelection:
        if not self.config.enabled:
            return decision
        now_s = float(now_s)
        if decision.intervention_type == "BRAKE":
            self.arm(now_s, brake_duration_s)
            return decision
        if not self.active:
            return decision

        release_safe = bool(
            now_s >= self.hold_until_s
            and float(speed_mps) <= float(self.config.release_speed_mps)
            and selected_admissible
            and (release_evidence or not self.config.require_release_evidence)
            and decision.risk_after is not None
            and float(decision.risk_after) <= float(self.config.release_risk)
        )
        self.safe_release_frames = self.safe_release_frames + 1 if release_safe else 0
        if self.safe_release_frames >= max(int(self.config.release_frames), 1):
            self.active = False
            self.safe_release_frames = 0
            return decision

        metadata = dict(decision.metadata or {})
        metadata.update(
            {
                "latched_from_intervention": decision.intervention_type,
                "latch_hold_until_s": self.hold_until_s,
                "latch_safe_release_frames": self.safe_release_frames,
            }
        )
        return InterventionSelection(
            None,
            "BRAKE",
            BRAKE_LATCH_HOLD,
            decision.risk_before,
            None if brake_risk_upper_bound is None else float(brake_risk_upper_bound),
            metadata=metadata,
        )

    def arm(self, now_s: float, brake_duration_s: float) -> None:
        if not self.config.enabled:
            return
        if not self.active:
            hold_s = max(float(brake_duration_s), float(self.config.min_hold_s), 0.0)
            self.hold_until_s = float(now_s) + hold_s
        self.active = True
        self.safe_release_frames = 0

    def remaining_s(self, now_s: float) -> float:
        return max(0.0, float(self.hold_until_s) - float(now_s)) if self.active else 0.0


def _as_list(values: Optional[Iterable], default_len: int = 0, default: float = 0.0) -> List:
    if values is None:
        return [default for _ in range(default_len)]
    if hasattr(values, "detach"):
        values = values.detach().cpu().reshape(-1).tolist()
    elif hasattr(values, "reshape") and not isinstance(values, list):
        values = values.reshape(-1).tolist()
    return list(values)


class OARMInterventionSelector:
    """Layered KEEP/Rerank/Probe/Brake selector driven by calibrated risk bounds."""

    def __init__(self, config: InterventionSelectorConfig = None):
        self.config = config or InterventionSelectorConfig()

    def select(
        self,
        risk_upper_bound: Iterable[float],
        yopo_cost: Optional[Iterable[float]] = None,
        geometry_admissible: Optional[Iterable[bool]] = None,
        deviation_from_top1: Optional[Iterable[float]] = None,
        probe_candidates: Optional[List[Dict[str, Any]]] = None,
        brake_feasible: bool = True,
        brake_risk_upper_bound: Optional[float] = None,
        top1_index: int = 0,
        latency_spike: bool = False,
        high_uncertainty: bool = False,
    ) -> InterventionSelection:
        risks = [float(v) for v in _as_list(risk_upper_bound)]
        if not risks:
            return self._brake(None, None, BRAKE_NO_SAFE_CANDIDATE, brake_feasible, brake_risk_upper_bound)
        n = len(risks)
        top1_index = int(max(0, min(top1_index, n - 1)))
        admissible = [bool(v) for v in _as_list(geometry_admissible, n, True)]
        costs = [float(v) for v in _as_list(yopo_cost, n, 0.0)]
        deviation = [float(v) for v in _as_list(deviation_from_top1, n, 0.0)]
        risk_before = risks[top1_index]
        if latency_spike:
            return self._brake(top1_index, risk_before, BRAKE_LATENCY_SPIKE, brake_feasible, brake_risk_upper_bound)
        if high_uncertainty:
            return self._brake(top1_index, risk_before, BRAKE_HIGH_UNCERTAINTY, brake_feasible, brake_risk_upper_bound)
        if admissible[top1_index] and risk_before <= self.config.delta_keep:
            return InterventionSelection(top1_index, "KEEP", KEEP_LOW_RISK, risk_before, risk_before, costs[top1_index])
        improvement_min = max(float(self.config.risk_improvement_min), 0.0)
        safe = [
            idx
            for idx, risk in enumerate(risks)
            if idx != top1_index
            and admissible[idx]
            and risk <= self.config.delta_safe
            and risk <= risk_before - improvement_min
        ]
        if safe:
            best = min(
                safe,
                key=lambda idx: costs[idx]
                + self.config.lambda_deviation * deviation[idx]
                + self.config.lambda_risk * risks[idx],
            )
            score = costs[best] + self.config.lambda_deviation * deviation[best] + self.config.lambda_risk * risks[best]
            return InterventionSelection(
                best,
                "RERANK",
                RERANK_TOP1_UNSAFE,
                risk_before,
                risks[best],
                score,
                metadata={"risk_improvement_min": improvement_min},
            )
        if admissible[top1_index] and risk_before <= self.config.delta_safe:
            return InterventionSelection(
                top1_index,
                "KEEP",
                KEEP_GRAY_NO_RISK_IMPROVEMENT,
                risk_before,
                risk_before,
                costs[top1_index] + self.config.lambda_risk * risk_before,
                metadata={"risk_improvement_min": improvement_min},
            )
        probe = self._select_probe(probe_candidates or [], risk_before)
        if probe is not None:
            return probe
        if not brake_feasible:
            feasible = [idx for idx in range(n) if admissible[idx]]
            if feasible:
                best = min(
                    feasible,
                    key=lambda idx: risks[idx] + 0.01 * costs[idx] + 0.01 * self.config.lambda_deviation * deviation[idx],
                )
                score = risks[best] + 0.01 * costs[best] + 0.01 * self.config.lambda_deviation * deviation[best]
                return InterventionSelection(
                    best,
                    "DEGRADED",
                    NO_VERIFIED_SAFE_ACTION,
                    risk_before,
                    risks[best],
                    score,
                    metadata={
                        "brake_feasible": False,
                        "brake_risk_upper_bound": None if brake_risk_upper_bound is None else float(brake_risk_upper_bound),
                        "fallback": "lowest_risk_admissible_candidate",
                    },
                )
            return InterventionSelection(
                top1_index,
                "DEGRADED",
                NO_VERIFIED_SAFE_ACTION,
                risk_before,
                risk_before,
                metadata={
                    "brake_feasible": False,
                    "brake_risk_upper_bound": None if brake_risk_upper_bound is None else float(brake_risk_upper_bound),
                    "fallback": "top1_no_admissible_candidate",
                },
            )
        return self._brake(top1_index, risk_before, BRAKE_NO_SAFE_CANDIDATE, brake_feasible, brake_risk_upper_bound)

    def _select_probe(self, probe_candidates: List[Dict[str, Any]], risk_before: Optional[float]):
        feasible = []
        for idx, cand in enumerate(probe_candidates):
            if not bool(cand.get("admissible", True)):
                continue
            risk = float(cand.get("risk_upper_bound", cand.get("risk", 1.0)))
            gain = float(cand.get("margin_gain_s", cand.get("visibility_gain_s", 0.0)))
            if risk > self.config.delta_probe or gain < self.config.min_probe_margin_gain_s:
                continue
            deviation = float(cand.get("deviation", 0.0))
            duration = float(cand.get("duration_s", 0.0))
            score = self.config.lambda_probe_risk * risk - self.config.lambda_probe_margin_gain * gain + self.config.lambda_deviation * deviation + self.config.lambda_probe_time * duration
            feasible.append((score, idx, risk, cand))
        if not feasible:
            return None
        score, idx, risk, cand = min(feasible, key=lambda item: item[0])
        return InterventionSelection(idx, "PROBE", PROBE_VISIBILITY_GAIN, risk_before, risk, score, dict(cand))

    @staticmethod
    def _brake(
        top1_index: Optional[int],
        risk_before: Optional[float],
        reason: str,
        brake_feasible: bool,
        brake_risk_upper_bound: Optional[float] = None,
    ) -> InterventionSelection:
        return InterventionSelection(
            None,
            "BRAKE",
            reason,
            risk_before,
            None if brake_risk_upper_bound is None else float(brake_risk_upper_bound),
            metadata={"brake_feasible": bool(brake_feasible), "top1_index": top1_index},
        )

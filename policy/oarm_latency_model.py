import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Deque, Dict, Iterable, Optional


@dataclass
class LatencyBudget:
    sensor_age_s: float
    queue_latency_s: float
    inference_latency_s: float
    selector_latency_s: float
    control_latency_s: float
    actuation_latency_s: float
    maneuver_latency_s: float
    tau_fixed_s: float
    tau_total_s: float
    brake_distance_m: float
    speed_parallel_mps: float

    def log_fields_ms(self) -> Dict[str, float]:
        return {
            "sensor_age_ms": 1000.0 * self.sensor_age_s,
            "queue_latency_ms": 1000.0 * self.queue_latency_s,
            "inference_latency_ms": 1000.0 * self.inference_latency_s,
            "selector_latency_ms": 1000.0 * self.selector_latency_s,
            "control_latency_ms": 1000.0 * self.control_latency_s,
            "estimated_brake_time_ms": 1000.0 * self.maneuver_latency_s,
            "reaction_budget_ms": 1000.0 * self.tau_total_s,
        }

    def to_dict(self) -> Dict[str, float]:
        out = asdict(self)
        out.update(self.log_fields_ms())
        return out


class RollingQuantile:
    def __init__(self, maxlen: int = 128, default: float = 0.0):
        self.values: Deque[float] = deque(maxlen=max(1, int(maxlen)))
        self.default = float(default)

    def update(self, value: Optional[float]):
        if value is None:
            return
        value = float(value)
        if math.isfinite(value) and value >= 0.0:
            self.values.append(value)

    def quantile(self, q: float = 0.95) -> float:
        if not self.values:
            return self.default
        ordered = sorted(self.values)
        idx = int(round(max(0.0, min(1.0, q)) * (len(ordered) - 1)))
        return float(ordered[idx])


class OARMLatencyModel:
    """Online reaction-time budget with a conservative braking term."""

    def __init__(self, brake_accel_mps2: float = 6.0, sensor_age_s: float = 0.0, queue_latency_s: float = 0.0, selector_latency_s: float = 0.0, control_latency_s: float = 0.02, actuation_latency_s: float = 0.03, latency_window: int = 128, quantile: float = 0.95):
        self.brake_accel_mps2 = max(float(brake_accel_mps2), 1e-3)
        self.sensor_age_s = max(float(sensor_age_s), 0.0)
        self.queue_latency_s = max(float(queue_latency_s), 0.0)
        self.selector_latency_s = max(float(selector_latency_s), 0.0)
        self.control_latency_s = max(float(control_latency_s), 0.0)
        self.actuation_latency_s = max(float(actuation_latency_s), 0.0)
        self.quantile = float(quantile)
        self.inference_history = RollingQuantile(latency_window, default=0.0)
        self.queue_history = RollingQuantile(latency_window, default=self.queue_latency_s)
        self.control_history = RollingQuantile(latency_window, default=self.control_latency_s)

    @staticmethod
    def _norm3(values: Iterable[float]) -> float:
        vals = list(values)
        if not vals:
            return 0.0
        return math.sqrt(sum(float(v) * float(v) for v in vals[:3]))

    def update(self, inference_latency_s: Optional[float] = None, queue_latency_s: Optional[float] = None, control_latency_s: Optional[float] = None):
        self.inference_history.update(inference_latency_s)
        self.queue_history.update(queue_latency_s)
        self.control_history.update(control_latency_s)

    def estimate(self, speed_parallel_mps: Optional[float] = None, velocity_body_mps: Optional[Iterable[float]] = None, inference_latency_s: Optional[float] = None, sensor_age_s: Optional[float] = None, queue_latency_s: Optional[float] = None, selector_latency_s: Optional[float] = None, control_latency_s: Optional[float] = None, actuation_latency_s: Optional[float] = None) -> LatencyBudget:
        self.update(inference_latency_s, queue_latency_s, control_latency_s)
        if speed_parallel_mps is None:
            speed_parallel_mps = self._norm3(velocity_body_mps or ())
        speed_parallel_mps = max(float(speed_parallel_mps), 0.0)
        sensor_age = self.sensor_age_s if sensor_age_s is None else max(float(sensor_age_s), 0.0)
        queue = self.queue_history.quantile(self.quantile) if queue_latency_s is None else max(float(queue_latency_s), 0.0)
        inference = self.inference_history.quantile(self.quantile) if inference_latency_s is None else max(float(inference_latency_s), 0.0)
        selector = self.selector_latency_s if selector_latency_s is None else max(float(selector_latency_s), 0.0)
        control = self.control_history.quantile(self.quantile) if control_latency_s is None else max(float(control_latency_s), 0.0)
        actuation = self.actuation_latency_s if actuation_latency_s is None else max(float(actuation_latency_s), 0.0)
        tau_fixed = sensor_age + queue + inference + selector + control + actuation
        maneuver = speed_parallel_mps / self.brake_accel_mps2
        brake_distance = speed_parallel_mps * tau_fixed + speed_parallel_mps * speed_parallel_mps / (2.0 * self.brake_accel_mps2)
        return LatencyBudget(sensor_age, queue, inference, selector, control, actuation, maneuver, tau_fixed, tau_fixed + maneuver, brake_distance, speed_parallel_mps)

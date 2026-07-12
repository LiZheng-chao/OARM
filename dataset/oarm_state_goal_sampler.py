import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class OARMStateGoalSamplerConfig:
    """Sampling policy for OARM-targeted training states.

    This module is intentionally standalone. It does not replace YOPO's dataset
    sampler unless a caller explicitly imports and uses it.
    """

    normal_speed_prob: float = 0.30
    medium_high_speed_prob: float = 0.40
    high_speed_prob: float = 0.20
    recovery_speed_prob: float = 0.10
    normal_speed_range: Tuple[float, float] = (1.5, 3.0)
    medium_high_speed_range: Tuple[float, float] = (3.0, 5.0)
    high_speed_range: Tuple[float, float] = (5.0, 6.0)
    recovery_speed_range: Tuple[float, float] = (0.0, 1.0)
    lateral_speed_std: float = 0.45
    vertical_speed_std: float = 0.25
    acc_std: float = 1.0
    acc_clip: float = 4.0
    goal_length: float = 10.0
    forward_goal_prob: float = 0.50
    occluded_goal_prob: float = 0.25
    lateral_goal_prob: float = 0.15
    near_goal_prob: float = 0.10
    forward_yaw_std_deg: float = 12.0
    occluded_yaw_range_deg: Tuple[float, float] = (18.0, 45.0)
    lateral_yaw_range_deg: Tuple[float, float] = (35.0, 70.0)
    pitch_std_deg: float = 8.0


class OARMStateGoalSampler:
    """Optional OARM-focused state/goal sampler.

    The returned velocity, acceleration, and goal vectors follow YOPO's
    convention: body-frame vectors in meters, meters/second, and
    meters/second^2. The distribution is biased toward states where reaction
    margin and yield/brake labels are informative.
    """

    def __init__(self, config: OARMStateGoalSamplerConfig = OARMStateGoalSamplerConfig(), seed=None):
        self.config = config
        self.rng = np.random.default_rng(seed)

    def sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, str]]:
        speed_mode, speed = self._sample_speed()
        vel_b = np.array(
            [
                speed,
                self.rng.normal(0.0, self.config.lateral_speed_std),
                self.rng.normal(0.0, self.config.vertical_speed_std),
            ],
            dtype=np.float32,
        )
        acc_b = self.rng.normal(0.0, self.config.acc_std, size=3)
        acc_b = np.clip(acc_b, -self.config.acc_clip, self.config.acc_clip).astype(np.float32)
        goal_mode, goal_b = self._sample_goal()
        meta = {"speed_mode": speed_mode, "goal_mode": goal_mode}
        return vel_b, acc_b, goal_b.astype(np.float32), meta

    def _sample_speed(self) -> Tuple[str, float]:
        cfg = self.config
        mode = self.rng.choice(
            ["normal", "medium_high", "high", "recovery"],
            p=[
                cfg.normal_speed_prob,
                cfg.medium_high_speed_prob,
                cfg.high_speed_prob,
                cfg.recovery_speed_prob,
            ],
        )
        ranges = {
            "normal": cfg.normal_speed_range,
            "medium_high": cfg.medium_high_speed_range,
            "high": cfg.high_speed_range,
            "recovery": cfg.recovery_speed_range,
        }
        low, high = ranges[str(mode)]
        return str(mode), float(self.rng.uniform(low, high))

    def _sample_goal(self) -> Tuple[str, np.ndarray]:
        cfg = self.config
        mode = self.rng.choice(
            ["forward", "occluded", "lateral", "near"],
            p=[
                cfg.forward_goal_prob,
                cfg.occluded_goal_prob,
                cfg.lateral_goal_prob,
                cfg.near_goal_prob,
            ],
        )
        if mode == "forward":
            yaw_deg = self.rng.normal(0.0, cfg.forward_yaw_std_deg)
            length = cfg.goal_length
        elif mode == "occluded":
            yaw_deg = self._signed_uniform(*cfg.occluded_yaw_range_deg)
            length = cfg.goal_length
        elif mode == "lateral":
            yaw_deg = self._signed_uniform(*cfg.lateral_yaw_range_deg)
            length = cfg.goal_length
        else:
            yaw_deg = self.rng.normal(0.0, cfg.forward_yaw_std_deg)
            length = self.rng.uniform(0.15, 0.45) * cfg.goal_length

        pitch_deg = self.rng.normal(0.0, cfg.pitch_std_deg)
        yaw = math.radians(float(yaw_deg))
        pitch = math.radians(float(pitch_deg))
        goal = length * np.array(
            [
                math.cos(yaw) * math.cos(pitch),
                math.sin(yaw) * math.cos(pitch),
                math.sin(pitch),
            ],
            dtype=np.float32,
        )
        return str(mode), goal

    def _signed_uniform(self, low: float, high: float) -> float:
        sign = -1.0 if self.rng.random() < 0.5 else 1.0
        return sign * float(self.rng.uniform(low, high))


if __name__ == "__main__":
    sampler = OARMStateGoalSampler(seed=0)
    counts = {}
    for _ in range(1000):
        _, _, _, meta = sampler.sample()
        key = (meta["speed_mode"], meta["goal_mode"])
        counts[key] = counts.get(key, 0) + 1
    for key, value in sorted(counts.items()):
        print(f"{key}: {value}")

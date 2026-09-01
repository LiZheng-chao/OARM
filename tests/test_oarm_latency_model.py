import math

import numpy as np

from OARM.policy.oarm_latency_model import OARMLatencyModel


def test_estimate_accepts_numpy_velocity_vector():
    model = OARMLatencyModel(brake_accel_mps2=5.0)

    budget = model.estimate(velocity_body_mps=np.array([3.0, 4.0, 0.0]))

    assert math.isclose(budget.speed_parallel_mps, 5.0)
    assert math.isclose(budget.maneuver_latency_s, 1.0)


def test_estimate_defaults_missing_velocity_to_zero():
    model = OARMLatencyModel()

    budget = model.estimate(velocity_body_mps=None)

    assert budget.speed_parallel_mps == 0.0
    assert budget.maneuver_latency_s == 0.0

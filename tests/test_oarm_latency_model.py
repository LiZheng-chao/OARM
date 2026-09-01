import math

import numpy as np

from OARM.policy.oarm_latency_model import OARMLatencyModel, is_sensor_frame_stale


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


def test_stale_sensor_frame_threshold():
    assert not is_sensor_frame_stale(None, 0.25)
    assert not is_sensor_frame_stale(0.25, 0.25)
    assert is_sensor_frame_stale(0.251, 0.25)
    assert not is_sensor_frame_stale(10.0, 0.0)

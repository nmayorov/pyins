import pandas as pd
from numpy.testing import assert_allclose, run_module_suite
import numpy as np
from pyins import sim
from pyins.transform import difference_trajectories
from pyins.integrate import coning_sculling, Integrator


def test_coning_sculling():
    # Basically a smoke test, because the function is quite simple.
    gyro = np.zeros((10, 3))
    gyro[:, 0] = 0.01
    gyro[:, 2] = -0.01

    accel = np.zeros((10, 3))
    accel[:, 2] = 0.1

    dv_true = np.empty_like(accel)
    dv_true[:, 0] = 0
    dv_true[:, 1] = -0.5e-3
    dv_true[:, 2] = 0.1
    theta, dv = coning_sculling(gyro, accel)
    assert_allclose(theta, gyro, rtol=1e-10)
    assert_allclose(dv, dv_true, rtol=1e-10)


def run_integration_test(reference_trajectory, gyro, accel, dt, sensor_type,
                         thresholds):
    theta, dv = coning_sculling(gyro, accel,
                                dt=dt if sensor_type == 'rate' else None)
    init = reference_trajectory.iloc[0]
    integrator = Integrator(dt,
                            init[['lat', 'lon', 'alt']],
                            init[['VE', 'VN', 'VU']],
                            init[['roll', 'pitch', 'heading']])
    result = integrator.integrate(theta, dv)
    diff = difference_trajectories(
        result, reference_trajectory).abs().max(axis=0)
    assert (diff < thresholds).all()


def test_integrate_stationary():
    total_time = 3600
    dt = 1e-1
    n = int(total_time / dt)

    lla = np.empty((n, 3))
    lla[:, 0] = 55.0
    lla[:, 1] = 37.0
    lla[:, 2] = 150.0

    rph = np.empty((n, 3))
    rph[:, 0] = -5.0
    rph[:, 1] = 10.0
    rph[:, 2] = 110.0

    thresholds = pd.Series({
        'east': 1e-3, 'north': 1e-3, 'up': 1e-2,
        'VE': 1e-6, 'VN': 1e-6, 'VU': 1e-5,
        'roll': 1e-8, 'pitch': 1e-8, 'heading': 1e-8
    })

    ref, gyro, accel = sim.from_position(dt, lla, rph, sensor_type='increment')
    run_integration_test(ref, gyro, accel, dt, 'increment', thresholds)

    ref, gyro, accel = sim.from_position(dt, lla, rph, sensor_type='rate')
    run_integration_test(ref, gyro, accel, dt, 'rate', thresholds)


def test_integrate_constant_velocity():
    dt = 1e-1
    total_time = 3600

    lla0 = [55.0, 37.0, 150.0]
    velocity_n = [5.0, -3.0, 0.2]

    thresholds = pd.Series({
        'east': 1, 'north': 1, 'up': 1,
        'VE': 1e-3, 'VN': 1e-3, 'VU': 1e-3,
        'roll': 1e-4, 'pitch': 1e-4, 'heading': 1e-4
    })

    ref, gyro, accel = sim.constant_velocity_motion(dt, total_time,
                                                    lla0, velocity_n,
                                                    sensor_type='increment')
    run_integration_test(ref, gyro, accel, dt, 'increment', thresholds)

    ref, gyro, accel = sim.constant_velocity_motion(dt, total_time,
                                                    lla0, velocity_n,
                                                    sensor_type='rate')
    run_integration_test(ref, gyro, accel, dt, 'rate', thresholds)


if __name__ == '__main__':
    run_module_suite()

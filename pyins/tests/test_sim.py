import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from pyins import sim
from pyins import earth
from pyins.inertial_sensor import EstimationModel
from pyins.util import GYRO_COLS, ACCEL_COLS


def test_sim_on_stationary():
    dt = 1e-1
    n_points = 1000

    time = dt * np.arange(n_points)

    lla = np.empty((n_points, 3))
    lla[:, 0] = 50.0
    lla[:, 1] = 45.0
    lla[:, 2] = 0.0
    hpr = np.zeros((n_points, 3))
    V_n = np.zeros((n_points, 3))

    slat = np.sin(np.deg2rad(50))
    clat = (1 - slat**2) ** 0.5

    gyro = earth.RATE * np.array([clat, 0, -slat])
    accel = np.array([0, 0, -earth.gravity(50, 0)])

    for lla_arg, velocity_n_arg in [(lla[0], V_n), (lla, V_n), (lla, None)]:
        for sensor_type in ['rate', 'increment']:
            trajectory, imu = sim.generate_imu(time, lla_arg, hpr, velocity_n_arg,
                                               sensor_type=sensor_type)

            assert_allclose(trajectory.lat, 50, rtol=1e-12)
            assert_allclose(trajectory.lon, 45, rtol=1e-12)
            assert_allclose(trajectory.VE, 0, atol=1e-7)
            assert_allclose(trajectory.VN, 0, atol=1e-7)
            assert_allclose(trajectory.roll, 0, atol=1e-8)
            assert_allclose(trajectory.pitch, 0, atol=1e-8)
            assert_allclose(trajectory.heading, 0, atol=1e-8)

            accel_atol = 1e-7 if sensor_type == 'increment' else 1e-6
            factor = dt if sensor_type == 'increment' else 1

            gyro_g = imu[GYRO_COLS].values
            accel_g = imu[ACCEL_COLS].values

            for i in range(3):
                assert_allclose(gyro_g[:, i], gyro[i] * factor, atol=1e-14)
                assert_allclose(accel_g[:, i], accel[i] * factor, atol=accel_atol)

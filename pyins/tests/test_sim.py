import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from pyins import sim
from pyins import earth
from pyins.util import GYRO_COLS, ACCEL_COLS


def test_sim_on_stationary():
    dt = 1e-1
    n_points = 1000

    time = dt * np.arange(n_points)

    lla = np.empty((n_points, 3))
    lla[:, 0] = 50.0
    lla[:, 1] = 45.0
    lla[:, 2] = 0.0
    rph = np.zeros((n_points, 3))
    V_n = np.zeros((n_points, 3))

    gyro = earth.rate_n(50)
    accel = -earth.gravity_n(50, 0)

    for lla_arg, velocity_n_arg in [(lla[0], V_n), (lla, V_n), (lla, None)]:
        for sensor_type in ['rate', 'increment']:
            trajectory, imu = sim.generate_imu(time, lla_arg, rph, velocity_n_arg,
                                               sensor_type=sensor_type)
            assert_array_equal(trajectory.index, time)
            assert_allclose(trajectory.lat, 50, rtol=1e-12)
            assert_allclose(trajectory.lon, 45, rtol=1e-12)
            assert_allclose(trajectory.alt, 0, atol=1e-30)
            assert_allclose(trajectory.VE, 0, atol=1e-7)
            assert_allclose(trajectory.VN, 0, atol=1e-7)
            assert_allclose(trajectory.VD, 0, atol=1e-7)
            assert_allclose(trajectory.roll, 0, atol=1e-8)
            assert_allclose(trajectory.pitch, 0, atol=1e-8)
            assert_allclose(trajectory.heading, 0, atol=1e-8)
            factor = dt if sensor_type == 'increment' else 1
            assert_allclose(imu[GYRO_COLS] - gyro * factor, 0, atol=1e-14)
            assert_allclose(imu[ACCEL_COLS] - accel * factor, 0,
                            atol=1e-7 if sensor_type == 'increment' else 1e-6)

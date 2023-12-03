import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from pyins import earth, sim, transform
from pyins.util import GYRO_COLS, ACCEL_COLS


def test_generate_imu():
    dt = 1e-1
    n_points = 1000

    time = dt * np.arange(n_points)
    lla0 = [50, 45, 0]
    rph0 = [-10, 20, 219]

    lla = np.empty((n_points, 3))
    lla[:] = lla0
    rph = np.empty((n_points, 3))
    rph[:] = rph0
    V_n = np.zeros((n_points, 3))

    mat_nb = transform.mat_from_rph(rph0)
    gyro = mat_nb.T @ earth.rate_n(lla0[0])
    accel = -mat_nb.T @ earth.gravity_n(lla0[0], lla0[2])

    for lla_arg, velocity_n_arg in [(lla[0], V_n), (lla, V_n), (lla, None)]:
        for sensor_type in ['rate', 'increment']:
            trajectory, imu = sim.generate_imu(time, lla_arg, rph, velocity_n_arg,
                                               sensor_type=sensor_type)
            assert_array_equal(trajectory.index, time)
            assert_allclose(trajectory.lat, 50, rtol=1e-16)
            assert_allclose(trajectory.lon, 45, rtol=1e-16)
            assert_allclose(trajectory.alt, 0, atol=1e-30)
            assert_allclose(trajectory.VE, 0, atol=1e-7)
            assert_allclose(trajectory.VN, 0, atol=1e-7)
            assert_allclose(trajectory.VD, 0, atol=1e-7)
            assert_allclose(trajectory.roll, rph0[0], rtol=1e-16)
            assert_allclose(trajectory.pitch, rph0[1], rtol=1e-16)
            assert_allclose(trajectory.heading, rph0[2], rtol=1e-16)
            factor = dt if sensor_type == 'increment' else 1
            assert_allclose(imu[GYRO_COLS] - gyro * factor, 0, atol=1e-14)
            assert_allclose(imu[ACCEL_COLS] - accel * factor, 0,
                            atol=1e-7 if sensor_type == 'increment' else 1e-6)

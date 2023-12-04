import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from pyins import earth, sim, transform, util
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


def test_generate_sine_velocity_motion():
    trajectory, imu = sim.generate_sine_velocity_motion(
        0.1, 10.1, [55, 37, 0], 0, [1, 1, 0], 10)
    assert_allclose(trajectory.lat.iloc[0], trajectory.lat.iloc[-1])
    assert_allclose(trajectory.lon.iloc[0], trajectory.lon.iloc[-1])
    assert_array_equal(trajectory.alt, 0)
    assert_allclose(trajectory[['VN', 'VE']].max(), 1)
    assert_allclose(trajectory[['VN', 'VE']].min(), -1)
    assert_allclose(trajectory.VN.iloc[0], trajectory.VN.iloc[-1], atol=1e-15)
    assert_allclose(trajectory.VE.iloc[0], trajectory.VE.iloc[-1])
    assert_array_equal(trajectory.VD, 0)
    assert_array_equal(trajectory[['roll', 'pitch']], 0)
    assert_allclose(
        util.to_180_range(trajectory.heading - (90 - 360 * trajectory.index / 10)), 0,
        atol=1e-13)
    assert_allclose(imu[['gyro_x', 'gyro_y']].mean(), 0, atol=1e-6)
    assert_allclose(imu.gyro_z, -360 / 10 * transform.DEG_TO_RAD, rtol=1e-4)
    assert_allclose(imu.accel_x, 0, atol=1e-4)
    assert_allclose(imu.accel_y, -360 / 10 * transform.DEG_TO_RAD, rtol=1e-3)
    assert_allclose(imu.accel_z, -earth.gravity(55, 0), rtol=1e-5)


def test_generate_measurements():
    trajectory, _ = sim.generate_sine_velocity_motion(0.1, 1000, [55, 37, 0],
                                                      1, 0.1, 20)
    rng = np.random.RandomState(0)
    position_obs = sim.generate_position_measurements(trajectory, 2.0, rng)
    assert_allclose(
        util.compute_rms(transform.compute_state_difference(position_obs, trajectory)),
        2.0, 1e-1)

    ned_velocity_obs = sim.generate_ned_velocity_measurements(trajectory, 0.2, rng)
    assert_allclose(
        util.compute_rms(transform.compute_state_difference(ned_velocity_obs,
                                                            trajectory)), 0.2, 1e-1)

    mat_nb = transform.mat_from_rph(trajectory[util.RPH_COLS])
    velocity_b = pd.DataFrame(util.mv_prod(mat_nb, trajectory[util.VEL_COLS], True),
                              columns=['VX', 'VY', 'VZ'])
    body_velocity_obs = sim.generate_body_velocity_measurements(trajectory, 0.3, rng)
    assert_allclose(util.compute_rms(transform.compute_state_difference(
        velocity_b, body_velocity_obs)), 0.3, 1.2e-1)


def test_generate_pva_error():
    pva_error = sim.generate_pva_error(1.0, 0.2, 0.05, 0.3, 0)
    assert pva_error[util.NED_COLS].abs().max() < 4.0
    assert pva_error[util.VEL_COLS].abs().max() < 0.8
    assert pva_error[['roll', 'pitch']].abs().max() < 0.2
    assert abs(pva_error.heading) < 1.2

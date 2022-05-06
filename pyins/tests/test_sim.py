import numpy as np
from numpy.testing import assert_allclose, run_module_suite
from scipy.spatial.transform import Rotation
from pyins import sim
from pyins import earth
from pyins import dcm


def test_sim_on_stationary():
    dt = 1e-1
    n_points = 1000

    lla = np.empty((n_points, 3))
    lla[:, 0] = 50.0
    lla[:, 1] = 45.0
    lla[:, 2] = 0.0
    hpr = np.zeros((n_points, 3))
    V_n = np.zeros((n_points, 3))

    slat = np.sin(np.deg2rad(50))
    clat = (1 - slat**2) ** 0.5

    gyro = earth.RATE * np.array([0, clat, slat])
    accel = np.array([0, 0, earth.gravity(50)])

    traj, gyro_g, accel_g = sim.from_position(dt, lla, hpr)
    assert_allclose(traj.lat, 50, rtol=1e-12)
    assert_allclose(traj.lon, 45, rtol=1e-12)
    assert_allclose(traj.VE, 0, atol=1e-7)
    assert_allclose(traj.VN, 0, atol=1e-7)
    assert_allclose(traj.h, 0, atol=1e-8)
    assert_allclose(traj.p, 0, atol=1e-8)
    assert_allclose(traj.r, 0, atol=1e-8)

    for i in range(3):
        assert_allclose(gyro_g[:, i], gyro[i] * dt, atol=1e-14)
        assert_allclose(accel_g[:, i], accel[i] * dt, atol=1e-7)

    traj, gyro_g, accel_g = sim.from_position(dt, lla, hpr, sensor_type='rate')
    assert_allclose(traj.lat, 50, rtol=1e-12)
    assert_allclose(traj.lon, 45, rtol=1e-12)
    assert_allclose(traj.VE, 0, atol=1e-7)
    assert_allclose(traj.VN, 0, atol=1e-7)
    assert_allclose(traj.h, 0, atol=1e-8)
    assert_allclose(traj.p, 0, atol=1e-8)
    assert_allclose(traj.r, 0, atol=1e-8)

    for i in range(3):
        assert_allclose(gyro_g[:, i], gyro[i], atol=1e-14)
        assert_allclose(accel_g[:, i], accel[i], atol=1e-6)

    traj, gyro_g, accel_g = sim.from_velocity(dt, lla[0], V_n, hpr)
    assert_allclose(traj.lat, 50, rtol=1e-12)
    assert_allclose(traj.lon, 45, rtol=1e-12)
    assert_allclose(traj.VE, 0, atol=1e-7)
    assert_allclose(traj.VN, 0, atol=1e-7)
    assert_allclose(traj.h, 0, atol=1e-8)
    assert_allclose(traj.p, 0, atol=1e-8)
    assert_allclose(traj.r, 0, atol=1e-8)

    for i in range(3):
        assert_allclose(gyro_g[:, i], gyro[i] * dt, atol=1e-14)
        assert_allclose(accel_g[:, i], accel[i] * dt, atol=1e-7)

    traj, gyro_g, accel_g = sim.from_velocity(dt, lla[0], V_n, hpr,
                                              sensor_type='rate')
    assert_allclose(traj.lat, 50, rtol=1e-12)
    assert_allclose(traj.lon, 45, rtol=1e-12)
    assert_allclose(traj.VE, 0, atol=1e-7)
    assert_allclose(traj.VN, 0, atol=1e-7)
    assert_allclose(traj.h, 0, atol=1e-8)
    assert_allclose(traj.p, 0, atol=1e-8)
    assert_allclose(traj.r, 0, atol=1e-8)

    for i in range(3):
        assert_allclose(gyro_g[:, i], gyro[i], atol=1e-14)
        assert_allclose(accel_g[:, i], accel[i], atol=1e-6)


def test_stationary():
    dt = 0.1
    n_points = 100
    t = dt * np.arange(n_points)
    lat = 58
    alt = -10
    hpr = np.empty((n_points, 3))
    hpr[:, 0] = 45
    hpr[:, 1] = -10
    hpr[:, 2] = 5
    Cnb = dcm.from_hpr(hpr)

    slat = np.sin(np.deg2rad(lat))
    clat = (1 - slat ** 2) ** 0.5
    omega_n = earth.RATE * np.array([0, clat, slat])
    g_n = np.array([0, 0, -earth.gravity(lat, alt)])
    Omega_b = Cnb[0].T.dot(omega_n)
    g_b = Cnb[0].T.dot(g_n)

    gyro, accel = sim.stationary_rotation(0.1, lat, alt, Cnb)
    gyro_true = np.tile(Omega_b * dt, (n_points - 1, 1))
    accel_true = np.tile(-g_b * dt, (n_points - 1, 1))
    assert_allclose(gyro, gyro_true)
    assert_allclose(accel, accel_true, rtol=1e-5, atol=1e-8)

    # Rotate around Earth's axis with additional rate.
    rate = 6
    rate_n = rate * np.array([0, clat, slat])
    rate_s = Cnb[0].T.dot(rate_n)
    Cbs = dcm.from_rv(rate_s * t[:, None])

    gyro, accel = sim.stationary_rotation(0.1, lat, alt, Cnb, Cbs)
    gyro_true = np.tile((Omega_b + rate_s) * dt, (n_points - 1, 1))
    assert_allclose(gyro, gyro_true)

    # Place IMU horizontally and rotate around vertical axis.
    # Gravity components should be identically 0.
    Cnb = Rotation.from_euler('Z', -hpr[:, 0], True).as_matrix()
    Cbs = Rotation.from_euler('Z', rate * t, True).as_matrix()
    gyro, accel = sim.stationary_rotation(0.1, lat, alt, Cnb, Cbs)
    accel_true = np.tile(-g_n * dt, (n_points - 1, 1))
    assert_allclose(accel, accel_true, rtol=1e-5, atol=1e-7)


if __name__ == '__main__':
    run_module_suite()

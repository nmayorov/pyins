import numpy as np
from numpy.testing import assert_allclose, run_module_suite
from pyins import sim
from pyins import earth


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

    gyro = earth.RATE * np.array([clat, 0, -slat])
    accel = np.array([0, 0, -earth.gravity(50, 0)])

    traj, gyro_g, accel_g = sim.from_position(dt, lla, hpr)
    assert_allclose(traj.lat, 50, rtol=1e-12)
    assert_allclose(traj.lon, 45, rtol=1e-12)
    assert_allclose(traj.VE, 0, atol=1e-7)
    assert_allclose(traj.VN, 0, atol=1e-7)
    assert_allclose(traj.roll, 0, atol=1e-8)
    assert_allclose(traj.pitch, 0, atol=1e-8)
    assert_allclose(traj.heading, 0, atol=1e-8)

    for i in range(3):
        assert_allclose(gyro_g[:, i], gyro[i] * dt, atol=1e-14)
        assert_allclose(accel_g[:, i], accel[i] * dt, atol=1e-7)

    traj, gyro_g, accel_g = sim.from_position(dt, lla, hpr, sensor_type='rate')
    assert_allclose(traj.lat, 50, rtol=1e-12)
    assert_allclose(traj.lon, 45, rtol=1e-12)
    assert_allclose(traj.VE, 0, atol=1e-7)
    assert_allclose(traj.VN, 0, atol=1e-7)
    assert_allclose(traj.roll, 0, atol=1e-8)
    assert_allclose(traj.pitch, 0, atol=1e-8)
    assert_allclose(traj.heading, 0, atol=1e-8)

    for i in range(3):
        assert_allclose(gyro_g[:, i], gyro[i], atol=1e-14)
        assert_allclose(accel_g[:, i], accel[i], atol=1e-6)

    traj, gyro_g, accel_g = sim.from_velocity(dt, lla[0], V_n, hpr)
    assert_allclose(traj.lat, 50, rtol=1e-12)
    assert_allclose(traj.lon, 45, rtol=1e-12)
    assert_allclose(traj.VE, 0, atol=1e-7)
    assert_allclose(traj.VN, 0, atol=1e-7)
    assert_allclose(traj.roll, 0, atol=1e-8)
    assert_allclose(traj.pitch, 0, atol=1e-8)
    assert_allclose(traj.heading, 0, atol=1e-8)

    for i in range(3):
        assert_allclose(gyro_g[:, i], gyro[i] * dt, atol=1e-14)
        assert_allclose(accel_g[:, i], accel[i] * dt, atol=1e-7)

    traj, gyro_g, accel_g = sim.from_velocity(dt, lla[0], V_n, hpr,
                                              sensor_type='rate')
    assert_allclose(traj.lat, 50, rtol=1e-12)
    assert_allclose(traj.lon, 45, rtol=1e-12)
    assert_allclose(traj.VE, 0, atol=1e-7)
    assert_allclose(traj.VN, 0, atol=1e-7)
    assert_allclose(traj.roll, 0, atol=1e-8)
    assert_allclose(traj.pitch, 0, atol=1e-8)
    assert_allclose(traj.heading, 0, atol=1e-8)

    for i in range(3):
        assert_allclose(gyro_g[:, i], gyro[i], atol=1e-14)
        assert_allclose(accel_g[:, i], accel[i], atol=1e-6)


if __name__ == '__main__':
    run_module_suite()

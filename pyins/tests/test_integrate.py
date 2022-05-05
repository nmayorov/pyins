from numpy.testing import assert_allclose, run_module_suite
import numpy as np
from pyins import earth
from pyins.integrate import coning_sculling, Integrator
from pyins import dcm


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


def test_integrate():
    # Test on the static bench.
    dt = 1e-1
    n = 100

    Cnb = dcm.from_hpr(45, -30, 60)
    gyro = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2)]) * earth.RATE * dt
    gyro = Cnb.T.dot(gyro)
    gyro = np.resize(gyro, (n, 3))

    accel = np.array([0, 0, earth.gravity(45)]) * dt
    accel = Cnb.T.dot(accel)
    accel = np.resize(accel, (n, 3))

    theta, dv = coning_sculling(gyro, accel)

    Integrator.INITIAL_SIZE = 50
    I = Integrator(dt, [45, 50, 0], [0, 0, 0], [45, -30, 60])
    I.integrate(theta[:n//2], dv[:n//2])
    I.integrate(theta[n//2:], dv[n//2:])

    assert_allclose(I.traj.lat, 45, rtol=1e-12)
    assert_allclose(I.traj.lon, 50, rtol=1e-12)
    assert_allclose(I.traj.VE, 0, atol=1e-8)
    assert_allclose(I.traj.VN, 0, atol=1e-8)
    assert_allclose(I.traj.h, 45, rtol=1e-12)
    assert_allclose(I.traj.p, -30, rtol=1e-12)
    assert_allclose(I.traj.r, 60, rtol=1e-12)


def test_integrate_rate_sensors():
    # Test on the static bench.
    dt = 1e-1
    n = 100

    Cnb = dcm.from_hpr(45, -30, 60)
    gyro = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2)]) * earth.RATE
    gyro = Cnb.T.dot(gyro)
    gyro = np.resize(gyro, (n, 3))

    accel = np.array([0, 0, earth.gravity(45)])
    accel = Cnb.T.dot(accel)
    accel = np.resize(accel, (n, 3))

    theta, dv = coning_sculling(gyro, accel, dt=dt)

    Integrator.INITIAL_SIZE = 50
    I = Integrator(dt, [45, 50, 0], [0, 0, 0], [45, -30, 60])
    I.integrate(theta[:n//2], dv[:n//2])
    I.integrate(theta[n//2:], dv[n//2:])

    assert_allclose(I.traj.lat, 45, rtol=1e-12)
    assert_allclose(I.traj.lon, 50, rtol=1e-12)
    assert_allclose(I.traj.VE, 0, atol=1e-8)
    assert_allclose(I.traj.VN, 0, atol=1e-8)
    assert_allclose(I.traj.h, 45, rtol=1e-12)
    assert_allclose(I.traj.p, -30, rtol=1e-12)
    assert_allclose(I.traj.r, 60, rtol=1e-12)


if __name__ == '__main__':
    run_module_suite()

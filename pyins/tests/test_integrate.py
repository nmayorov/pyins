from numpy.testing import assert_allclose, run_module_suite
import numpy as np
from pyins import earth
from pyins.integrate import coning_sculling, integrate, Integrator
from pyins.integrate import _integrate_py_fast, integrate3d, Integrator3d
from pyins import dcm, sim


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

    accel = np.array([0, 0, earth.gravity(np.rad2deg(1/np.sqrt(2)))]) * dt
    accel = Cnb.T.dot(accel)
    accel = np.resize(accel, (n, 3))

    theta, dv = coning_sculling(gyro, accel)

    traj = integrate(dt, 45, 50, 0, 0, 45, -30, 60, theta, dv)

    assert_allclose(traj.lat, 45, rtol=1e-12)
    assert_allclose(traj.lon, 50, rtol=1e-12)
    assert_allclose(traj.VE, 0, atol=1e-8)
    assert_allclose(traj.VN, 0, atol=1e-8)
    assert_allclose(traj.h, 45, rtol=1e-12)
    assert_allclose(traj.p, -30, rtol=1e-12)
    assert_allclose(traj.r, 60, rtol=1e-12)

    Integrator.INITIAL_SIZE = 50
    I = Integrator(dt, 45, 50, 0, 0, 45, -30, 60)
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

    accel = np.array([0, 0, earth.gravity(np.rad2deg(1/np.sqrt(2)))])
    accel = Cnb.T.dot(accel)
    accel = np.resize(accel, (n, 3))

    theta, dv = coning_sculling(gyro, accel, dt=dt)

    traj = integrate(dt, 45, 50, 0, 0, 45, -30, 60, theta, dv)

    assert_allclose(traj.lat, 45, rtol=1e-12)
    assert_allclose(traj.lon, 50, rtol=1e-12)
    assert_allclose(traj.VE, 0, atol=1e-8)
    assert_allclose(traj.VN, 0, atol=1e-8)
    assert_allclose(traj.h, 45, rtol=1e-12)
    assert_allclose(traj.p, -30, rtol=1e-12)
    assert_allclose(traj.r, 60, rtol=1e-12)

    Integrator.INITIAL_SIZE = 50
    I = Integrator(dt, 45, 50, 0, 0, 45, -30, 60)
    I.integrate(theta[:n//2], dv[:n//2])
    I.integrate(theta[n//2:], dv[n//2:])

    assert_allclose(I.traj.lat, 45, rtol=1e-12)
    assert_allclose(I.traj.lon, 50, rtol=1e-12)
    assert_allclose(I.traj.VE, 0, atol=1e-8)
    assert_allclose(I.traj.VN, 0, atol=1e-8)
    assert_allclose(I.traj.h, 45, rtol=1e-12)
    assert_allclose(I.traj.p, -30, rtol=1e-12)
    assert_allclose(I.traj.r, 60, rtol=1e-12)


def sim_traj():
    dt = 0.01
    n = 36000
    t = dt * np.arange(n)

    lat, lon, alt = (50, 35, 1000)
    VE = 10 * np.ones(n)
    VN = 10 * np.ones(n)
    VU = np.cos(2 * np.pi * t / 60)

    h = 10 + 5 * np.sin(2 * np.pi * t / 30)
    p = 5 * np.cos(2 * np.pi * t / 30)
    r = 5 * np.sin(2 * np.pi * t / 30)

    return  dt, *sim.from_velocity(dt, lat, lon, alt, VE, VN, VU, h, p, r)


def test_integrate_py_fast():
    dt, traj, gyro, accel = sim_traj()

    lla0 = traj.loc[0, ['lat', 'lon', 'alt']]
    Vn0 = traj.loc[0, ['VE', 'VN', 'VU']]
    hpr0 = traj.loc[0, ['h', 'p', 'r']]

    lla = np.zeros((traj.shape[0], 3))
    Vn = np.zeros((traj.shape[0], 3))
    Cnb = np.zeros((traj.shape[0], 3, 3))

    lla[0] = lla0
    lla[0, :2] = np.deg2rad(lla[0, :2])
    Vn[0] = Vn0
    Cnb[0] = dcm.from_hpr(hpr0[0], hpr0[1], hpr0[2])

    theta, dv = coning_sculling(gyro, accel)

    _integrate_py_fast(lla, Vn, Cnb, theta, dv, dt)
    lla[:, :2] = np.rad2deg(lla[:, :2])
    h, p, r = dcm.to_hpr(Cnb)

    assert_allclose(traj.lat, lla[:, 0], atol=2e-8)
    assert_allclose(traj.lon, lla[:, 1], atol=2e-8)
    assert_allclose(traj.alt, lla[:, 2], atol=4e-3)

    assert_allclose(traj.VE, Vn[:, 0], atol=1e-5)
    assert_allclose(traj.VN, Vn[:, 1], atol=1e-5)
    assert_allclose(traj.VU, Vn[:, 2], atol=3e-5)

    assert_allclose(traj.h, h, atol=3e-8)
    assert_allclose(traj.p, p, atol=3e-8)
    assert_allclose(traj.r, r, atol=3e-8)


def test_integrate3d():
    dt, traj_ref, gyro, accel = sim_traj()
    theta, dv = coning_sculling(gyro, accel)

    lla0 = traj_ref.loc[0, ['lat', 'lon', 'alt']]
    Vn0 = traj_ref.loc[0, ['VE', 'VN', 'VU']]
    hpr0 = traj_ref.loc[0, ['h', 'p', 'r']]

    traj_test = integrate3d(dt, lla0, Vn0, hpr0, theta, dv)

    assert_allclose(traj_test.lat, traj_ref.lat, atol=2e-8)
    assert_allclose(traj_test.lon, traj_ref.lon, atol=2e-8)
    assert_allclose(traj_test.alt, traj_ref.alt, atol=4e-3)

    assert_allclose(traj_test.VE, traj_ref.VE, atol=1e-5)
    assert_allclose(traj_test.VN, traj_ref.VN, atol=1e-5)
    assert_allclose(traj_test.VU, traj_ref.VU, atol=3e-5)

    assert_allclose(traj_test.h, traj_ref.h, atol=3e-8)
    assert_allclose(traj_test.p, traj_ref.p, atol=3e-8)
    assert_allclose(traj_test.r, traj_ref.r, atol=3e-8)


    Integrator.INITIAL_SIZE = 10000
    I = Integrator3d(dt, lla0, Vn0, hpr0)
    n = theta.shape[0]
    I.integrate(theta[:n//2], dv[:n//2])
    I.integrate(theta[n//2:], dv[n//2:])

    assert_allclose(I.traj.lat, traj_ref.lat, atol=2e-8)
    assert_allclose(I.traj.lon, traj_ref.lon, atol=2e-8)
    assert_allclose(I.traj.alt, traj_ref.alt, atol=4e-3)

    assert_allclose(I.traj.VE, traj_ref.VE, atol=1e-5)
    assert_allclose(I.traj.VN, traj_ref.VN, atol=1e-5)
    assert_allclose(I.traj.VU, traj_ref.VU, atol=3e-5)

    assert_allclose(I.traj.h, traj_ref.h, atol=3e-8)
    assert_allclose(I.traj.p, traj_ref.p, atol=3e-8)
    assert_allclose(I.traj.r, traj_ref.r, atol=3e-8)


def test_integrate3d_rate_sensors():
    # Test on the static bench.
    dt = 1e-2
    n = 100

    Cnb = dcm.from_hpr(45, -30, 60)
    gyro = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2)]) * earth.RATE
    gyro = Cnb.T.dot(gyro)
    gyro = np.resize(gyro, (n, 3))

    accel = np.array([0, 0, earth.gravity(45, 500)])
    accel = Cnb.T.dot(accel)
    accel = np.resize(accel, (n, 3))

    theta, dv = coning_sculling(gyro, accel, dt=dt)

    lla0 = [45, 50, 500]
    Vn0 = [0, 0, 0]
    hpr0 = [45, -30, 60]
    traj = integrate3d(dt, lla0, Vn0, hpr0, theta, dv)

    assert_allclose(traj.lat, 45, rtol=1e-20)
    assert_allclose(traj.lon, 50, rtol=1e-20)
    assert_allclose(traj.alt, 500, rtol=1e-20)
    assert_allclose(traj.VE, 0, atol=1e-12)
    assert_allclose(traj.VN, 0, atol=1e-12)
    assert_allclose(traj.VU, 0, atol=1e-12)
    assert_allclose(traj.h, 45, atol=1e-12)
    assert_allclose(traj.p, -30, atol=1e-12)
    assert_allclose(traj.r, 60, atol=1e-12)

    Integrator.INITIAL_SIZE = 50
    I = Integrator3d(dt, lla0, Vn0, hpr0)
    I.integrate(theta[:n//2], dv[:n//2])
    I.integrate(theta[n//2:], dv[n//2:])

    assert_allclose(I.traj.lat, 45, rtol=1e-20)
    assert_allclose(I.traj.lon, 50, rtol=1e-20)
    assert_allclose(I.traj.alt, 500, rtol=1e-20)
    assert_allclose(I.traj.VE, 0, atol=1e-12)
    assert_allclose(I.traj.VN, 0, atol=1e-12)
    assert_allclose(I.traj.VU, 0, atol=1e-12)
    assert_allclose(I.traj.h, 45, atol=1e-12)
    assert_allclose(I.traj.p, -30, atol=1e-12)
    assert_allclose(I.traj.r, 60, atol=1e-12)


if __name__ == '__main__':
    run_module_suite()

import numpy as np
from numpy.testing import assert_allclose, run_module_suite
from pyins.sim import from_position
from pyins import earth


def test_from_position():
    lat = [50, 50]
    lon = [45, 45]
    dt = 1e-1

    slat = np.sin(np.deg2rad(50))
    clat = (1 - slat**2) ** 0.5

    gyro = earth.RATE * np.array([0, clat, slat]) * dt
    accel = np.array([0, 0, earth.gravity(slat)]) * dt

    traj, gyro_g, accel_g = from_position(dt, lat, lon, [0, 10],
                                          h=[0, 0], p=[0, 0], r=[0, 0])
    assert_allclose(traj.lat, 50, rtol=1e-12)
    assert_allclose(traj.lon, 45, rtol=1e-12)
    assert_allclose(traj.VE, 0, atol=1e-7)
    assert_allclose(traj.VN, 0, atol=1e-7)
    assert_allclose(traj.h, 0, atol=1e-8)
    assert_allclose(traj.p, 0, atol=1e-8)
    assert_allclose(traj.r, 0, atol=1e-8)

    for i in range(3):
        assert_allclose(gyro_g[:, i], gyro[i], atol=1e-16)
        assert_allclose(accel_g[:, i], accel[i], atol=1e-7)


if __name__ == '__main__':
    run_module_suite()

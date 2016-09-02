import numpy as np
from numpy.testing import assert_approx_equal, assert_allclose
from pyins import align, dcm, earth
from pyins.integrate import coning_sculling


def test_wahba():
    lat = 45
    Cnb = dcm.from_hpr(45, -30, 60)

    dt = 1e-1
    n = 1000

    gyro = np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2)]) * earth.RATE * dt
    gyro = Cnb.T.dot(gyro)
    gyro = np.resize(gyro, (n, 3))

    accel = np.array([0, 0, earth.gravity(1 / np.sqrt(2))]) * dt
    accel = Cnb.T.dot(accel)
    accel = np.resize(accel, (n, 3))

    np.random.seed(0)
    gyro += 1e-6 * np.random.randn(*gyro.shape) * dt
    accel += 1e-4 * np.random.randn(*accel.shape) * dt

    phi, dv = coning_sculling(gyro, accel)
    hpr, P = align.align_wahba(dt, phi, dv, lat)

    assert_allclose(hpr, [45, -30, 60], rtol=1e-3)


if __name__ == '__main__':
    test_wahba()

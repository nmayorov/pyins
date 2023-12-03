import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from scipy.spatial.transform import Rotation
from pyins import earth, transform
from pyins.sim import sinusoid_velocity_motion
from pyins import util
from pyins.util import (LLA_COLS, NED_COLS, VEL_COLS, RPH_COLS,
                        GYRO_COLS, RATE_COLS)

def test_lla_to_ecef():
    r_e = transform.lla_to_ecef([0, 0, 10])
    assert_allclose(r_e, [earth.A + 10, 0, 0])

    r_e = transform.lla_to_ecef([-90, 0, -10])
    b = (1 - earth.E2) ** 0.5 * earth.A
    assert_allclose(r_e, [0, 0, -b + 10], atol=1e-9)

    r_e = transform.lla_to_ecef([[0, 0, 10], [-90, 0, -10]])
    assert_allclose(r_e, [[earth.A + 10, 0, 0], [0, 0, -b + 10]], atol=1e-9)


def test_lla_to_ned():
    lla = [[0, 0, 1000], [90, 90, 0], [0, -90, -1000]]
    ned = transform.lla_to_ned(lla)
    a = earth.A + 1000
    b = (1 - earth.E2) ** 0.5 * earth.A
    test = [[0, 0, 0], [b, 0, a], [0, -earth.A + 1000, a]]
    assert_allclose(ned, test, atol=1e-8)
    assert(isinstance(ned, np.ndarray))

    lla = pd.DataFrame(data=lla, columns=LLA_COLS)
    ned = transform.lla_to_ned(lla)
    test = pd.DataFrame(data=test, columns=NED_COLS)
    assert_allclose(ned, test, atol=1e-8)
    assert(isinstance(ned, pd.DataFrame))

    lla = [[0, 0, 1000], [90, 90, 0], [0, -90, -1000]]
    ned = transform.lla_to_ned(lla, [0, -90, -1000])
    a = earth.A - 1000
    test = [[0, earth.A + 1000, a], [b, 0, a], [0, 0, 0]]
    assert_allclose(ned, test, atol=1e-8)
    assert(isinstance(ned, np.ndarray))

    lla = pd.DataFrame(data=lla, columns=LLA_COLS)
    ned = transform.lla_to_ned(lla, [0, -90, -1000])
    test = pd.DataFrame(data=test, columns=NED_COLS)
    assert_allclose(ned, test, atol=1e-8)
    assert(isinstance(ned, pd.DataFrame))


def test_perturb_ll():
    lla = [40, 50, 0]
    lla_new = transform.perturb_lla(lla, [10, -20, 5])
    lla_back = transform.perturb_lla(lla_new, [-10, 20, -5])
    assert_allclose(lla_back, lla, rtol=1e-11)

    lla = [[40, 50, 0], [-40, 50, 10]]
    lla_new = transform.perturb_lla(lla, [10, -20, 5])
    lla_back = transform.perturb_lla(lla_new, [-10, 20, -5])
    assert_allclose(lla_back, lla, rtol=1e-11)


def test_translate_trajectory():
    traj, imu = sinusoid_velocity_motion(0.1, 60, [10, 20, -3], [5, 7, 3])
    traj_new = transform.translate_trajectory(traj, [10, -20, 5])
    mat = Rotation.from_euler('xyz', traj[RPH_COLS], degrees=True).as_matrix()
    ned = util.mv_prod(mat, [10, -20, 5])
    lla_test = transform.perturb_lla(traj[LLA_COLS], ned)
    assert_allclose(traj_new[LLA_COLS], lla_test, rtol=1e-11)
    assert_allclose(traj_new[VEL_COLS], traj[VEL_COLS], rtol=1e-11)
    assert_allclose(traj_new[RPH_COLS], traj[RPH_COLS], rtol=1e-11)

    traj[RATE_COLS] = imu[GYRO_COLS].values
    traj_new = transform.translate_trajectory(traj, [10, -20, 5])
    vel_b = np.cross(traj[RATE_COLS], [10, -20, 5])
    vel_test = traj[VEL_COLS] + util.mv_prod(mat, vel_b)
    assert_allclose(traj_new[VEL_COLS], vel_test)

    traj_back = transform.translate_trajectory(traj_new, [-10, 20, -5])
    assert_allclose(traj_back, traj)


def test_compute_lla_difference():
    lla0 = [40, 50, 0]
    lla1 = transform.perturb_lla(lla0, [10, -20, 5])
    ned = transform.compute_lla_difference(lla1, lla0)
    assert_allclose(ned, [10, -20, 5], atol=1e-4)

    lla0 = [[40, 50, 0], [-40, 50, 10]]
    lla1 = transform.perturb_lla(lla0, [10, -20, 5])
    ned = transform.compute_lla_difference(lla1, lla0)
    assert_allclose(ned, [[10, -20, 5], [10, -20, 5]], atol=1e-4)


def test_resample_state():
    traj, _ = sinusoid_velocity_motion(0.02, 60.01, [40, -50, 100], [5, -7, 0],
                                       velocity_change_amplitude=[10, 15, 2])
    ref_traj, _ = sinusoid_velocity_motion(0.01, 60.01, [40, -50, 100], [5, -7, 0],
                                           velocity_change_amplitude=[10, 15, 2])
    test_traj = transform.resample_state(traj, ref_traj.index)

    assert_allclose(test_traj.lat, ref_traj.lat)
    assert_allclose(test_traj.lon, ref_traj.lon)
    assert_allclose(test_traj.alt, ref_traj.alt, atol=1e-4)

    assert_allclose(test_traj[VEL_COLS], ref_traj[VEL_COLS], atol=1e-5)
    assert_allclose(test_traj[RPH_COLS], ref_traj[RPH_COLS], atol=3e-4)


def test_mat_en_from_ll():
    A1 = np.eye(3)
    A2 = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])

    assert_allclose(transform.mat_en_from_ll(-90, 0), A1,
                    rtol=1e-10, atol=1e-10)
    assert_allclose(transform.mat_en_from_ll(0, 0), A2, rtol=1e-10, atol=1e-10)
    assert_allclose(transform.mat_en_from_ll([-90, 0], [0, 0]),
                    np.stack([A1, A2]), rtol=1e-10, atol=1e-10)

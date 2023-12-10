import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from scipy.spatial.transform import Rotation
from pyins import earth, sim, transform, util
from pyins.util import LLA_COLS, NED_COLS, VEL_COLS, RPH_COLS, GYRO_COLS, RATE_COLS


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
    expected = [[0, 0, 0], [b, 0, a], [0, -earth.A + 1000, a]]
    assert_allclose(ned, expected, atol=1e-8)
    assert isinstance(ned, np.ndarray)

    lla = pd.DataFrame(data=lla, columns=LLA_COLS)
    ned = transform.lla_to_ned(lla)
    expected = pd.DataFrame(data=expected, columns=NED_COLS)
    assert_allclose(ned, expected, atol=1e-8)
    assert isinstance(ned, pd.DataFrame)

    lla = [[0, 0, 1000], [90, 90, 0], [0, -90, -1000]]
    ned = transform.lla_to_ned(lla, [0, -90, -1000])
    a = earth.A - 1000
    expected = [[0, earth.A + 1000, a], [b, 0, a], [0, 0, 0]]
    assert_allclose(ned, expected, atol=1e-8)
    assert isinstance(ned, np.ndarray)

    lla = pd.DataFrame(data=lla, columns=LLA_COLS)
    ned = transform.lla_to_ned(lla, [0, -90, -1000])
    expected = pd.DataFrame(data=expected, columns=NED_COLS)
    assert_allclose(ned, expected, atol=1e-8)
    assert isinstance(ned, pd.DataFrame)


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
    traj, imu = sim.generate_sine_velocity_motion(0.1, 60, [10, 20, -3], [5, 7, 3])
    traj_new = transform.translate_trajectory(traj, [10, -20, 5])
    mat_nb = Rotation.from_euler('xyz', traj[RPH_COLS], degrees=True).as_matrix()
    ned = util.mv_prod(mat_nb, [10, -20, 5])
    lla = transform.perturb_lla(traj[LLA_COLS], ned)
    assert_allclose(traj_new[LLA_COLS], lla, rtol=1e-11)
    assert_allclose(traj_new[VEL_COLS], traj[VEL_COLS], rtol=1e-11)
    assert_allclose(traj_new[RPH_COLS], traj[RPH_COLS], rtol=1e-11)

    traj[RATE_COLS] = imu[GYRO_COLS].values
    traj_new = transform.translate_trajectory(traj, [10, -20, 5])
    vel_b = np.cross(traj[RATE_COLS], [10, -20, 5])
    vel = traj[VEL_COLS] + util.mv_prod(mat_nb, vel_b)
    assert_allclose(traj_new[VEL_COLS], vel)

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
    traj, _ = sim.generate_sine_velocity_motion(
        0.02, 60.01, [40, -50, 100], [5, -7, 0], velocity_change_amplitude=[10, 15, 2])
    ref_traj, _ = sim.generate_sine_velocity_motion(
        0.01, 60.01, [40, -50, 100], [5, -7, 0], velocity_change_amplitude=[10, 15, 2])
    test_traj = transform.resample_state(traj, ref_traj.index)

    assert_allclose(test_traj.lat, ref_traj.lat)
    assert_allclose(test_traj.lon, ref_traj.lon)
    assert_allclose(test_traj.alt, ref_traj.alt, atol=1e-4)

    assert_allclose(test_traj[VEL_COLS], ref_traj[VEL_COLS], atol=1e-5)
    assert_allclose(test_traj[RPH_COLS], ref_traj[RPH_COLS], atol=3e-4)


def test_compute_state_difference():
    traj1, _ = sim.generate_sine_velocity_motion(
        0.02, 60.01, [40, -50, 100], [5, -7, 0], velocity_change_amplitude=[10, 15, 2])
    traj2, _ = sim.generate_sine_velocity_motion(
        0.01, 60.01, [40, -50, 100], [5, -7, 0], velocity_change_amplitude=[10, 15, 2])
    traj2 = traj2.iloc[1::2]
    traj2[LLA_COLS] = transform.perturb_lla(traj2[LLA_COLS], [-3, 5, 7])
    traj2[VEL_COLS] += [1, -2, 3]
    traj2[RPH_COLS] += [359, 3, -3]

    diff = transform.compute_state_difference(traj1, traj2)
    assert_allclose(diff[NED_COLS] - [3, -5, -7], 0, atol=1e-4)
    assert_allclose(diff[VEL_COLS] - [-1, 2, -3], 0, atol=1e-5)
    assert_allclose(diff[RPH_COLS] - [1, -3, 3], 0, atol=3e-4)


def test_mat_en_from_ll():
    A1 = np.eye(3)
    A2 = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])

    assert_allclose(transform.mat_en_from_ll(-90, 0), A1,
                    rtol=1e-10, atol=1e-10)
    assert_allclose(transform.mat_en_from_ll(0, 0), A2, rtol=1e-10, atol=1e-10)
    assert_allclose(transform.mat_en_from_ll([-90, 0], [0, 0]),
                    np.stack([A1, A2]), rtol=1e-10, atol=1e-10)


def test_mat_from_rph():
    assert_allclose(transform.mat_from_rph([0, 0, 0]), np.eye(3))

    rph1 = [90, 0, 0]
    mat1 = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    assert_allclose(transform.mat_from_rph(rph1), mat1, atol=1e-15)

    rph2 = [0, -90, 0]
    mat2 = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
    assert_allclose(transform.mat_from_rph(rph2), mat2, atol=1e-15)

    rph3 = [0, 0, 180]
    mat3 = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    assert_allclose(transform.mat_from_rph(rph3), mat3, atol=1e-15)

    rph = np.asarray([rph1, rph2, rph3])
    mat = np.asarray([mat1, mat2, mat3])
    assert_allclose(transform.mat_from_rph(rph), mat, atol=1e-15)


def test_mat_to_rph():
    assert_allclose(transform.mat_to_rph(np.eye(3)), 0)

    np.random.seed(0)
    for i in range(10):
        rph = 30 * np.random.randn(3)
        mat = Rotation.from_euler('xyz', rph, degrees=True).as_matrix()
        assert_allclose(transform.mat_to_rph(mat), rph)

    rph = 30 * np.random.randn(10, 3)
    mat = Rotation.from_euler('xyz', rph, degrees=True).as_matrix()
    assert_allclose(transform.mat_to_rph(mat), rph)


def test_smooth_rotations():
    def generate_envelope(dt, total_time, rest_time, raise_time):
        time = np.arange(0, total_time, dt)
        result = np.zeros_like(time)

        mask = (time >= rest_time) & (time < rest_time + raise_time)
        result[mask] = (time[mask] - rest_time) / raise_time

        steady_time = total_time - 2 * (rest_time + raise_time)
        mask = (time >= rest_time + raise_time) & (
                    time < rest_time + raise_time + steady_time)
        result[mask] = 1.0

        mask = (time >= rest_time + raise_time + steady_time) & (
                    time < total_time - rest_time)
        result[mask] = (-time[mask] + (total_time - rest_time)) / raise_time

        return result

    dt = 0.01
    total_time = 60
    time = np.arange(0, total_time, dt)
    smoothing_time = 1
    envelope = generate_envelope(dt, total_time, smoothing_time, 20 * smoothing_time)

    rph = np.empty((len(time), 3))
    rph[:, 0] = 30 * envelope * np.sin(2 * np.pi * time / 0.1)
    rph[:, 1] = 70 * envelope * np.cos(2 * np.pi * time / 10)
    rph[:, 2] = 90 * envelope * np.sin(2 * np.pi * time / 20)

    rotations = Rotation.from_euler('xyz', rph, True)
    smoothed_rotation, num_taps = transform.smooth_rotations(rotations, dt,
                                                             smoothing_time)
    rph[:, 0] = 0.0
    rotations_expected = Rotation.from_euler('xyz', rph, True)
    rotation_diff = (smoothed_rotation[num_taps - 1:] *
                     rotations_expected[num_taps // 2:-(num_taps // 2)].inv())
    assert np.max(rotation_diff.magnitude()) < 0.4 * transform.DEG_TO_RAD


def test_smooth_state():
    trajectory, _ = sim.generate_sine_velocity_motion(
        0.1, 600.0, [55, 37, 150], [1, 2, -0.1], 0.5, 1)
    trajectory_smoothed = transform.smooth_state(trajectory, 10)
    trajectory_expected, _ = sim.generate_sine_velocity_motion(
        0.1, 600.0, [55, 37, 150], [1, 2, -0.1])
    difference = transform.compute_state_difference(trajectory_smoothed,
                                                    trajectory_expected)
    assert difference[NED_COLS].abs().max(None) < 0.1
    assert difference[VEL_COLS].abs().max(None) < 5e-5
    assert difference[RPH_COLS].abs().max(None) < 0.7

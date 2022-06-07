import numpy as np
from numpy.testing import assert_allclose, run_module_suite
import pandas as pd
from scipy.spatial.transform import Rotation
from pyins import earth, transform


def test_lla_to_ecef():
    r_e = transform.lla_to_ecef([0, 0, 10])
    assert_allclose(r_e, [earth.R0 + 10, 0, 0])

    r_e = transform.lla_to_ecef([-90, 0, -10])
    b = (1 - earth.E2) ** 0.5 * earth.R0
    assert_allclose(r_e, [0, 0, -b + 10], atol=1e-9)

    r_e = transform.lla_to_ecef([[0, 0, 10], [-90, 0, -10]])
    assert_allclose(r_e, [[earth.R0 + 10, 0, 0],
                          [0, 0, -b + 10]], atol=1e-9)


def test_perturb_ll():
    lla = [40, 50, 0]
    lla_new = transform.perturb_lla(lla, [10, -20, 5])
    lla_back = transform.perturb_lla(lla_new, [-10, 20, -5])
    assert_allclose(lla_back, lla, rtol=1e-11)


def test_phi_to_delta_rph():
    rph = [10, -20, 30]
    mat = transform.mat_from_rph(rph)
    phi = np.array([-0.02, 0.01, -0.03])
    mat_perturbed = Rotation.from_rotvec(-phi).as_matrix() @ mat

    rph_perturbed = transform.mat_to_rph(mat_perturbed)
    delta_rph_true = rph_perturbed - rph

    T = transform.phi_to_delta_rph(rph)
    delta_rph_linear = np.rad2deg(T @ phi)

    assert_allclose(delta_rph_linear, delta_rph_true, rtol=1e-1)


def test_mat_en_from_ll():
    A1 = np.eye(3)
    A2 = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])

    assert_allclose(transform.mat_en_from_ll(-90, 0), A1,
                    rtol=1e-10, atol=1e-10)
    assert_allclose(transform.mat_en_from_ll(0, 0), A2, rtol=1e-10, atol=1e-10)
    assert_allclose(transform.mat_en_from_ll([-90, 0], [0, 0]),
                    np.stack([A1, A2]), rtol=1e-10, atol=1e-10)


def test_correct_trajectory():
    traj = pd.DataFrame(index=range(2))
    traj[['lat', 'lon', 'alt']] = [[30, 30, 100], [-30, -30, -100]]
    traj[['VN', 'VE', 'VD']] = [[-1, 2, 3], [3, -1, -2]]
    traj[['roll', 'pitch', 'heading']] = [[10, -20, 45], [-30, -50, 180]]

    error = pd.DataFrame(index=range(2))
    error[['north', 'east', 'down']] = [[12000, -20000, 50000],
                                       [-20000, 15000, -70000]]
    error[['VN', 'VE', 'VD']] = [[-2, 1, 2], [2, -2, -3]]
    error[['roll', 'pitch', 'heading']] = [[10, -20, 45], [-30, -50, 180]]

    traj_new = transform.correct_trajectory(traj, error)
    assert_allclose(traj_new[['VN', 'VE', 'VD']], 1)
    assert_allclose(traj_new[['roll', 'pitch', 'heading']], 0)

    error_back = transform.difference_trajectories(traj, traj_new)
    print(error)
    print(error_back)
    assert_allclose(error, error_back)


if __name__ == '__main__':
    run_module_suite()

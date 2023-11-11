import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation
from pyins import earth, transform


def test_lla_to_ecef():
    r_e = transform.lla_to_ecef([0, 0, 10])
    assert_allclose(r_e, [earth.A + 10, 0, 0])

    r_e = transform.lla_to_ecef([-90, 0, -10])
    b = (1 - earth.E2) ** 0.5 * earth.A
    assert_allclose(r_e, [0, 0, -b + 10], atol=1e-9)

    r_e = transform.lla_to_ecef([[0, 0, 10], [-90, 0, -10]])
    assert_allclose(r_e, [[earth.A + 10, 0, 0], [0, 0, -b + 10]], atol=1e-9)


def test_perturb_ll():
    lla = [40, 50, 0]
    lla_new = transform.perturb_lla(lla, [10, -20, 5])
    lla_back = transform.perturb_lla(lla_new, [-10, 20, -5])
    assert_allclose(lla_back, lla, rtol=1e-11)


def test_mat_en_from_ll():
    A1 = np.eye(3)
    A2 = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])

    assert_allclose(transform.mat_en_from_ll(-90, 0), A1,
                    rtol=1e-10, atol=1e-10)
    assert_allclose(transform.mat_en_from_ll(0, 0), A2, rtol=1e-10, atol=1e-10)
    assert_allclose(transform.mat_en_from_ll([-90, 0], [0, 0]),
                    np.stack([A1, A2]), rtol=1e-10, atol=1e-10)

import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation
from pyins import earth
from pyins._numba_integrate import gravity, mat_from_rotvec
import pytest


def test_gravity():
    np.random.seed(0)
    for i in range(10):
        lat, alt = [180, 1000] * np.random.rand(2) - [90, 500]
        assert_allclose(gravity(lat, alt), earth.gravity(lat, alt))


@pytest.mark.parametrize("sd", [1.0, 0.01])
def test_mat_from_rotvec(sd):
    mat_ref = np.eye(3)
    mat_test = np.zeros((3, 3))
    rv = np.asarray([0, 0, 0])
    mat_from_rotvec(rv, mat_test)
    assert_allclose(mat_ref, mat_ref)

    np.random.seed(0)
    for i in range(10):
        rv = sd * np.random.randn(3)
        mat_ref = Rotation.from_rotvec(rv).as_matrix()
        mat_test = np.zeros((3, 3))
        mat_from_rotvec(rv, mat_test)
        assert_allclose(mat_test, mat_ref)

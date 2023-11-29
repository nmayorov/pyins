import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation
from pyins import earth
from pyins._numba_integrate import gravity, mat_from_rotvec


def test_earth():
    assert_allclose(gravity(0, 100), earth.gravity(0, 100))
    assert_allclose(gravity(90, -100), earth.gravity(90, -100))
    assert_allclose(gravity(-90, 1000), earth.gravity(-90, 1000))

    for i in range(10):
        lat, alt = [180, 10000] * (np.random.rand(2) - [90, 0])
        assert_allclose(gravity(lat, alt), earth.gravity(lat, alt))


def test_mat_from_rotvec():
    m_ref = Rotation.from_rotvec([0, 0, 0]).as_matrix()
    m_test = np.zeros((3, 3))
    rv = np.asarray([0, 0, 0])
    mat_from_rotvec(rv, m_test)
    assert_allclose(m_test, m_ref)

    for i in range(10):
        rv = np.random.randn(3)
        m_ref = Rotation.from_rotvec(rv).as_matrix()
        m_test = np.zeros((3, 3))
        mat_from_rotvec(rv, m_test)
        assert_allclose(m_test, m_ref)

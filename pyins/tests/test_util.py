import numpy as np
from numpy.testing import assert_allclose
from pyins import util


def test_mm_prod():
    np.random.seed(0)
    a = np.random.randn(3, 4)
    b = np.random.randn(4, 3)
    assert_allclose(util.mm_prod(a, b), a @ b)
    assert_allclose(util.mm_prod(a, b, at=True, bt=True), a.T @ b.T)

    a = np.random.randn(10, 3, 4)
    at = a.transpose((0, 2, 1))
    assert_allclose(util.mm_prod(a, b), a @ b)
    assert_allclose(util.mm_prod(a, b, at=True, bt=True), at @ b.T)

    b = np.random.randn(10, 4, 3)
    bt = b.transpose((0, 2, 1))
    assert_allclose(util.mm_prod(a, b), a @ b)
    assert_allclose(util.mm_prod(a, b, at=True, bt=True), at @ bt)


def test_skew_matrix():
    vec = np.array([
        [0, 1, 2],
        [-2, 3, 5]
    ])
    check = np.array([
        [-2, 3, 6],
        [0, -2, 3]
    ])
    assert_allclose(util.skew_matrix(vec[0]) @ check[0],
                    np.cross(vec[0], check[0]))
    assert_allclose(util.skew_matrix(vec[1]) @ check[1],
                    np.cross(vec[1], check[1]))
    assert_allclose(util.mv_prod(util.skew_matrix(vec), check),
                    np.cross(vec, check))

import numpy as np
from numpy.testing import assert_allclose
from pyins import util


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

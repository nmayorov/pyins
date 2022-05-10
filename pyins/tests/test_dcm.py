from numpy.testing import assert_allclose, run_module_suite
import numpy as np
from pyins import dcm, util


def test_skew_matrix():
    vec = np.array([
        [0, 1, 2],
        [-2, 3, 5]
    ])
    check = np.array([
        [-2, 3, 6],
        [0, -2, 3]
    ])
    assert_allclose(dcm.skew_matrix(vec[0]) @ check[0],
                    np.cross(vec[0], check[0]))
    assert_allclose(dcm.skew_matrix(vec[1]) @ check[1],
                    np.cross(vec[1], check[1]))
    assert_allclose(util.mv_prod(dcm.skew_matrix(vec), check),
                    np.cross(vec, check))


if __name__ == '__main__':
    run_module_suite()

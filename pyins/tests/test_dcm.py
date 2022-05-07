import pytest
from numpy.testing import assert_allclose, run_module_suite
import numpy as np
from pyins import dcm, util


def test_from_rv():
    rv1 = np.array([1, 0, 0]) * np.pi / 3
    A1 = dcm.from_rv(rv1)
    A1_true = np.array([[1, 0, 0],
                       [0, 0.5, -0.5 * np.sqrt(3)],
                       [0, 0.5 * np.sqrt(3), 0.5]])
    assert_allclose(A1, A1_true, rtol=1e-10)

    rv2 = np.array([1, 1, 1]) * 1e-10
    A2 = dcm.from_rv(rv2)
    A2_true = np.array([[1, -1e-10, 1e-10],
                        [1e-10, 1, -1e-10],
                        [-1e-10, 1e-10, 1]])
    assert_allclose(A2, A2_true, rtol=1e-10)

    n = np.array([-0.5, 1/np.sqrt(2), 0.5])
    theta = np.pi / 6
    rv3 = n * theta
    s = np.sin(theta)
    c = np.cos(theta)

    A3 = dcm.from_rv(rv3)
    A3_true = np.array([
        [(1-c)*n[0]*n[0] + c, (1-c)*n[0]*n[1] - n[2]*s,
         (1-c)*n[0]*n[2] + s*n[1]],
        [(1-c)*n[1]*n[0] + s*n[2], (1-c)*n[1]*n[1] + c,
         (1-c)*n[1]*n[2] - s*n[0]],
        [(1-c)*n[2]*n[0] - s*n[1], (1-c)*n[2]*n[1] + s*n[0],
         (1-c)*n[2]*n[2] + c]
    ])
    assert_allclose(A3, A3_true, rtol=1e-10)

    rv = np.empty((30, 3))
    rv[:10] = rv1
    rv[10:20] = rv2
    rv[20:] = rv3
    A_true = np.empty((30, 3, 3))
    A_true[:10] = A1_true
    A_true[10:20] = A2_true
    A_true[20:] = A3_true
    A = dcm.from_rv(rv)
    assert_allclose(A, A_true, rtol=1e-8)

    rv = rv[::4]
    A_true = A_true[::4]
    A = dcm.from_rv(rv)
    assert_allclose(A, A_true, rtol=1e-10)


def test_from_rph():
    rph1 = [0, 0, 30]
    A_true1 = np.array([[np.sqrt(3)/2, 0.5, 0],
                        [-0.5, np.sqrt(3)/2, 0],
                        [0, 0, 1]])
    assert_allclose(dcm.from_rph(rph1), A_true1, rtol=1e-10)

    rph2 = np.rad2deg([-1e-10, 3e-10, 1e-10])
    A_true2 = np.array([[1, 1e-10, -1e-10],
                        [-1e-10, 1, -3e-10],
                        [1e-10, 3e-10, 1]])
    assert_allclose(dcm.from_rph(rph2), A_true2, rtol=1e-8)

    rph3 = [60, -30, 45]
    A_true3 = np.array([
        [-np.sqrt(6)/8 + np.sqrt(2)/4, np.sqrt(6)/4,
         np.sqrt(2)/8 + np.sqrt(6)/4],
        [-np.sqrt(2)/4 - np.sqrt(6)/8, np.sqrt(6)/4,
         -np.sqrt(6)/4 + np.sqrt(2)/8],
        [-0.75, -0.5, np.sqrt(3)/4]
    ])
    assert_allclose(dcm.from_rph(rph3), A_true3, rtol=1e-8)

    rph = np.vstack((rph1, rph2, rph3))
    A = np.array((A_true1, A_true2, A_true3))
    assert_allclose(dcm.from_rph(rph), A, rtol=1e-8)


def test_to_rph():
    A1 = np.identity(3)
    rph1 = np.zeros(3)
    assert_allclose(dcm.to_rph(A1), rph1, atol=1e-10)

    A2 = np.array([[1, 1e-10, -2e-10],
                   [-1e-10, 1, 3e-10],
                   [2e-10, -3e-10, 1]])
    rph2 = np.rad2deg([-2e-10, -3e-10, 1e-10])
    assert_allclose(dcm.to_rph(A2), rph2, atol=1e-10)

    A3 = np.array([
        [1/np.sqrt(2), 0, 1/np.sqrt(2)],
        [0, 1, 0],
        [-1/np.sqrt(2), 0, 1/np.sqrt(2)]
    ])
    rph3 = np.array([45, 0, 0])
    assert_allclose(dcm.to_rph(A3), rph3, rtol=1e-10)

    A4 = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
    rph4 = np.array([-180, -90, 0])
    assert_allclose(dcm.to_rph(A4), rph4, rtol=1e-7)

    A = np.empty((20, 3, 3))
    A[:5] = A1
    A[5:10] = A2
    A[10:15] = A3
    A[15:] = A4
    rph = np.empty((20, 3))
    rph[:5] = rph1
    rph[5:10] = rph2
    rph[10:15] = rph3
    rph[15:20] = rph4

    ret = dcm.to_rph(A)
    assert_allclose(ret, rph)


def test_dcm_hpr_conversion():
    rng = np.random.RandomState(0)

    hpr = np.empty((20, 3))
    hpr[:, 0] = rng.uniform(-180, 180, 20)
    hpr[:, 1] = rng.uniform(-90, 90, 20)
    hpr[:, 2] = rng.uniform(0, 360, 20)

    A = dcm.from_rph(hpr)
    hpr_r = dcm.to_rph(A)

    assert_allclose(hpr, hpr_r, rtol=1e-10)


def test_from_ll():
    ll1 = np.array([90, -90])
    A1 = np.identity(3)
    assert_allclose(dcm.from_ll(*ll1), A1, rtol=1e-10, atol=1e-10)

    ll2 = np.array([-30, -45])
    A2 = np.array([[2**0.5/2, 2**0.5/4, 6**0.5/4],
                   [2**0.5/2, -2**0.5/4, -6**0.5/4],
                   [0, 3**0.5/2, -0.5]])
    assert_allclose(dcm.from_ll(*ll2), A2, rtol=1e-10, atol=1e-10)

    ll = np.empty((10, 2))
    ll[:5] = ll1
    ll[5:10] = ll2
    A = np.empty((10, 3, 3))
    A[:5] = A1
    A[5:10] = A2
    assert_allclose(dcm.from_ll(*ll.T), A, rtol=1e-10, atol=1e-10)


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

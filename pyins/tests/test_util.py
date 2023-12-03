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


def test_mm_prod_symmetric():
    np.random.seed(0)
    a = np.random.randn(3, 3)
    b = np.random.randn(3, 3)
    assert_allclose(util.mm_prod_symmetric(a, b), a @ b @ a.T)

    a = np.random.randn(10, 3, 3)
    at = a.transpose((0, 2, 1))
    b = np.random.randn(3, 3)
    assert_allclose(util.mm_prod_symmetric(a, b), a @ b @ at)

    b = np.random.randn(10, 3, 3)
    assert_allclose(util.mm_prod_symmetric(a, b), a @ b @ at)


def test_mv_prod():
    np.random.seed(0)
    a = np.random.randn(3, 3)
    b = np.random.randn(3)
    assert_allclose(util.mv_prod(a, b), a @ b)
    assert_allclose(util.mv_prod(a, b, at=True), a.T @ b)

    a = np.random.randn(10, 3, 3)
    b = np.random.randn(3)
    at = a.transpose((0, 2, 1))
    assert_allclose(util.mv_prod(a, b), a @ b)
    assert_allclose(util.mv_prod(a, b, at=True), at @ b)

    b = np.random.randn(10, 3)
    assert_allclose(util.mv_prod(a, b), (a @ b[:, :, None]).reshape(10, 3))
    assert_allclose(util.mv_prod(a, b, at=True),
                   (at @ b[:, :, None]).reshape(10, 3))


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


def test_compute_rms():
    a = np.zeros(10)
    assert_allclose(util.compute_rms(a), 0)
    a[::2] = 2
    assert_allclose(util.compute_rms(a), 2**0.5)
    a[1::2] = -2
    assert_allclose(util.compute_rms(a), 2)


def test_bunch():
    dict = {'a': 1, 'b': None, 'c': 3 * np.ones(3)}
    bunch = util.Bunch(dict)
    assert(bunch.a == 1)
    assert(bunch.b == None)
    assert((bunch.c == 3).all())

    assert(dir(bunch) == ['a', 'b', 'c'])
    assert(isinstance(repr(bunch), str))

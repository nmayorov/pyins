import numpy as np
from numpy.testing import assert_allclose
from pyins import kalman


def test_kalman_correct():
    # As the implementation of standard Kalman correction formulas is
    # straightforward we use a sanity check, when the correct answer is
    # computed without complete formulas.
    P0 = np.array([[2, 0], [0, 1]], dtype=float)
    x0 = np.array([0, 0], dtype=float)

    z = np.array([1, 2])
    R = np.array([[3, 0], [0, 2]])
    H = np.identity(2)

    x_true = np.array([1 * 2 / (2 + 3), 2 * 1 / (1 + 2)])
    P_true = np.diag([1 / (1/2 + 1/3), 1 / (1/1 + 1/2)])

    x = x0.copy()
    P = P0.copy()
    kalman.correct(x, P, z, H, R)
    assert_allclose(x, x_true)
    assert_allclose(P, P_true)


def test_process_matrices():
    F = np.array([[0, 1], [0, 0]])
    Q = np.array([[0, 0], [0, 1]])

    Phi, Qd = kalman.compute_process_matrices(F, Q, 0.1, 'first-order')
    assert_allclose(Phi, np.eye(2) + F * 0.1)
    assert_allclose(Qd, Q * 0.1)

    Phi, Qd = kalman.compute_process_matrices(F, Q, 1, 'expm')
    assert_allclose(Phi, np.array([[1, 1], [0, 1]]))
    assert_allclose(Qd, np.array([[1 / 3, 0.5], [0.5, 1]]))

    np.random.seed(0)
    F = np.random.randn(15, 15)
    Q = np.eye(15)

    Phi, Qd = kalman.compute_process_matrices(F, Q, 1, 'expm')
    test = F.dot(Qd) + Qd.dot(F.T) + Q - Phi.dot(Q).dot(Phi.T)
    assert_allclose(test, 0, atol=1E-12)

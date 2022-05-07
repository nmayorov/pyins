"""Kalman filter functions."""
import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular


def correct(x, P, z, H, R, gain_factor, gain_curve):
    PHT = np.dot(P, H.T)

    S = np.dot(H, PHT) + R
    e = z - H.dot(x)
    L = cholesky(S, lower=True)
    inn = solve_triangular(L, e, lower=True)

    if gain_curve is not None:
        q = (np.dot(inn, inn) / inn.shape[0]) ** 0.5
        f = gain_curve(q)
        if f == 0:
            return inn
        L *= (q / f) ** 0.5

    K = cho_solve((L, True), PHT.T, overwrite_b=True).T
    if gain_factor is not None:
        K *= gain_factor[:, None]

    U = -K.dot(H)
    U[np.diag_indices_from(U)] += 1
    x += K.dot(z - H.dot(x))
    P[:] = U.dot(P).dot(U.T) + K.dot(R).dot(K.T)

    return inn


def smooth_rts(x, P, xa, Pa, Phi):
    """Smooth state according to Rauch-Tung-Striebel algorithm [1]_.

    Parameters
    ----------
    x : ndarray, shape (n_points, n_states)
        States from the forward pass. On exit will contain smoothed, improved
        estimates.
    P : ndarray, shape (n_points, n_states, n_states)
        Covariance from forward pass. On exit will contain smoothed, improved
        covariances.
    xa : ndarray, shape (n_points, n_states)
        States from the forward pass before measurements at each point
        was applied, i. e. `x[i]` is `xa[i]` after measurement at `i` was
        applied.
    Pa : ndarray, shape (n_points, n_states, n_states)
        Covariances from forward pass before measurements at each point was
        applied, i. e. `P[i]` is `Pa[i]` after measurement at `i` was applied.
    Phi : ndarray, shape (n_points - 1, n_states, n_states)
        Transition matrices between states.

    References
    ----------
    .. [1] H. E. Rauch, F. Tung and C.T. Striebel, "Maximum Likelihood
           Estimates of Linear Dynamic Systems", AIAA Journal, Vol. 3,
           No. 8, August 1965.
    """
    n_points, n_states = x.shape
    I = np.identity(n_states)
    for i in reversed(range(n_points - 1)):
        L = cholesky(Pa[i + 1], check_finite=False)
        Pa_inv = cho_solve((L, False), I, check_finite=False)

        C = P[i].dot(Phi[i].T).dot(Pa_inv)

        x[i] += C.dot(x[i + 1] - xa[i + 1])
        P[i] += C.dot(P[i + 1] - Pa[i + 1]).dot(C.T)
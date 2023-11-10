"""Kalman filter functions."""
import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular, expm


def compute_process_matrices(F, Q, dt, algorithm='first-order'):
    """Compute discrete process matrices for Kalman filter prediction.

    First order approximation and the algorithm with matrix exponential [1]_ are
    available.

    Parameters
    ----------
    F : ndarray, shape (n_states, n_states)
        Continuous process transition matrix.
    Q : ndarray, shape (n_states, n_states)
        Continuous process noise matrix.
    dt : float
        Time step.
    algorithm : 'first-order' or 'expm', optional
        Algorith to use: first order approximation or matrix exponential.
        Default is 'first-order'

    References
    ----------
    .. [1] CHARLES F. VAN LOAN, "Computing Integrals Involving the
           Matrix Exponential", IEEE TRANSACTIONS ON AUTOMATIC CONTROL,
           VOL. AC-23, NO. 3, JUNE 1978.
    """
    n = len(F)
    if algorithm == 'first-order':
        return np.eye(n) + F * dt, Q * dt
    elif algorithm == 'expm':
        H = np.zeros((2 * n, 2 * n))
        H[:n, :n] = F
        H[:n, n:] = Q
        H[n:, n:] = -F.T
        H = expm(H * dt)
        return H[:n, :n], H[:n, n:] @ H[:n, :n].T
    else:
        raise ValueError("`algorithm` must be 'first-order' or 'expm'")


def correct(x, P, z, H, R):
    """Perform Kalman correction.

    The correction obtains a posteriori state and covariance given observation
    of the form::

        z = H @ x + v, with v ~ N(0, R)

    Parameters
    ----------
    x : ndarray, shape (n_states,)
        State vector. On exit will contain corrected value.
    P : ndarray, shape (n_states, n_states)
        Covariance matrix. On exit will contain corrected value.
    z : ndarray, shape (n_obs,)
        Observation vector.
    H : ndarray, shape (n_obs, n_states)
        Matrix which relates state and observation vectors.
    R : ndarray, shape (n_obs, n_obs)
        Positive semi-definite observation noise matrix.

    Returns
    -------
    innovation : ndarray, shape (n_obs,)
        Standardized innovation vector with theoretical zero mean and identity
        covariance matrix.
    """
    HP = H @ P
    S = HP @ H.T + R

    e = z - H.dot(x)
    L = cholesky(S, lower=True)
    K = cho_solve((L, True), HP, overwrite_b=True).T

    x += K.dot(z - H.dot(x))
    U = np.eye(len(x)) - K.dot(H)
    P[:] = U.dot(P).dot(U.T) + K.dot(R).dot(K.T)

    return solve_triangular(L, e, lower=True)

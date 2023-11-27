"""Kalman filter functions.

Module contains abstract functions for linear Kalman filter operations.
Refer to [1]_ for the theory of Kalman filters.

Functions
---------
.. autosummary::
    :toctree: generated/

    compute_process_matrices
    correct

References
----------
.. [1] P\. S\. Maybeck, "Stochastic Models, Estimation and Control", volume 1
"""
import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular, expm


def compute_process_matrices(F, Q, dt):
    """Compute discrete process matrices for Kalman filter prediction.

    The algorithm with matrix exponential described in [1]_ is used.

    Parameters
    ----------
    F : ndarray, shape (n_states, n_states)
        Continuous process transition matrix.
    Q : ndarray, shape (n_states, n_states)
        Continuous process noise matrix.
    dt : float
        Time step.

    References
    ----------
    .. [1] Charles F. van Loan, "Computing Integrals Involving the Matrix Exponential"
    """
    n = len(F)
    H = np.zeros((2 * n, 2 * n))
    H[:n, :n] = F
    H[:n, n:] = Q
    H[n:, n:] = -F.T
    H = expm(H * dt)
    return H[:n, :n], H[:n, n:] @ H[:n, :n].T


def correct(x, P, z, H, R):
    """Perform Kalman correction.

    The correction obtains a posteriori state and covariance given measurement
    of the form::

        z = H @ x + v, with v ~ N(0, R)

    Parameters
    ----------
    x : ndarray, shape (n_states,)
        State vector.
    P : ndarray, shape (n_states, n_states)
        Covariance matrix.
    z : ndarray, shape (n_obs,)
        Observation vector.
    H : ndarray, shape (n_obs, n_states)
        Matrix which relates state and measurement vectors.
    R : ndarray, shape (n_obs, n_obs)
        Positive semi-definite measurement noise matrix.

    Returns
    -------
    x : ndarray, shape (n_states,)
        Corrected state vector.
    P : ndarray, shape (n_states, n_states)
        A posteriori covariance matrix.
    innovation : ndarray, shape (n_obs,)
        Standardized innovation vector with theoretical zero mean and identity
        covariance matrix.
    """
    HP = H @ P
    S = HP @ H.T + R

    e = z - H.dot(x)
    L = cholesky(S, lower=True)
    K = cho_solve((L, True), HP, overwrite_b=True).T
    U = np.eye(len(x)) - K.dot(H)

    return (x + K @ (z - H @ x), U.dot(P).dot(U.T) + K.dot(R).dot(K.T),
            solve_triangular(L, e, lower=True))

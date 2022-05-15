"""Kalman filter functions."""
import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular, expm


def compute_process_matrices(F, Q, dt, algorithm='first-order'):
    """Compute discrete process matrices for Kalman filter prediction.

    Support first order approximation and the algorithm with matrix exponential
    [1]_.

    Parameters
    ----------
    F : array_like, shape (n_states, n_states)
        Continuous process transition matrix.
    Q : array_like, shape (n_states, n_states)
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
    F = np.asarray(F)
    Q = np.asarray(Q)
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
        assert False


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
        Positive definite observation noise matrix.

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
    K = cho_solve((L, True), HP, overwrite_b=True).transpose()

    U = -K.dot(H)
    U[np.diag_indices_from(U)] += 1
    x += K.dot(z - H.dot(x))
    P[:] = U.dot(P).dot(U.T) + K.dot(R).dot(K.T)

    return solve_triangular(L, e, lower=True)


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

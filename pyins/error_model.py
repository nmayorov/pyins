"""INS error model to use in navigation Kalman filters.

An INS error model is a system of non-stationary (depends on trajectory) linear
differential equations which describe time evolution of INS errors.

An error model vector has 9 total states: 3 for position, velocity and attitude
errors. The states can be selected in different ways and several error models were
proposed in the literature. Here the "modified phi-angle" model proposed in [1]_ is
used. The key feature of it is that the velocity error is measured relative to the true
velocity resolved in the "platform" frame which eliminates specific force from the
system matrix.

Functions
---------
.. autosummary::
    :toctree: generated/

    system_matrices
    transform_matrix
    correct_pva
    position_error_jacobian
    ned_velocity_error_jacobian
    body_velocity_error_jacobian
    propagate_errors

References
----------
.. [1] Bruno M. Scherzinger and D.Blake Reid "Modified Strapdown Inertial Navigator
       Error Models"
"""
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from . import earth, util, transform
from .util import (LLA_COLS, VEL_COLS, RPH_COLS, RATE_COLS, NED_COLS,
                   TRAJECTORY_ERROR_COLS)


STATES = ['DR1', 'DR2', 'DR3', 'DV1', 'DV2', 'DV3', 'PHI1', 'PHI2', 'PHI3']
N_STATES = len(STATES)

DRN = 0
DRE = 1
DRD = 2
DVN = 3
DVE = 4
DVD = 5
DROLL = 6
DPITCH = 7
DHEADING = 8
DR_OUT = [DRN, DRE, DRD]
DV_OUT = [DVN, DVE, DVD]
DRPH = [DROLL, DPITCH, DHEADING]

DR1 = 0
DR2 = 1
DR3 = 2
DV1 = 3
DV2 = 4
DV3 = 5
PHI1 = 6
PHI2 = 7
PHI3 = 8
DR = [DR1, DR2, DR3]
DV = [DV1, DV2, DV3]
PHI = [PHI1, PHI2, PHI3]


def _phi_to_delta_rph(rph):
    rph = np.asarray(rph)
    single = rph.ndim == 1
    rph = np.atleast_2d(rph)
    result = np.zeros((len(rph), 3, 3))

    sin = np.sin(np.deg2rad(rph))
    cos = np.cos(np.deg2rad(rph))

    result[:, 0, 0] = -cos[:, 2] / cos[:, 1]
    result[:, 0, 1] = -sin[:, 2] / cos[:, 1]
    result[:, 1, 0] = sin[:, 2]
    result[:, 1, 1] = -cos[:, 2]
    result[:, 2, 0] = -cos[:, 2] * sin[:, 1] / cos[:, 1]
    result[:, 2, 1] = -sin[:, 2] * sin[:, 1] / cos[:, 1]
    result[:, 2, 2] = -1

    return result[0] if single else result


def system_matrices(trajectory):
    """Compute matrices which govern the error model differential equations.

    The system of differential equations has the form::

        dx/dt = F @ x + B_gyro @ gyro_error + B_accel @ accel_error

    Where

        -``x`` - error vector
        -``gyro_error``, ``accel_error`` - vectors with gyro and accelerometer errors
        -``F`` - error dynamics matrix
        -``B_gyro`` - gyro error coupling matrix
        -``B_accel`` - accel error coupling matrix

    Parameters
    ----------
    trajectory : Trajectory or Pva
        Either full trajectory dataframe or single position-velocity-attitude.

    Returns
    -------
    F : ndarray, shape (n_points, 9, 9) or (9, 9)
        Error dynamics matrix.
    B_gyro : ndarray, shape (n_points, 9, 3) or (9, 3)
        Gyro error coupling matrix.
    B_accel : ndarray, shape (n_points, 9, 3) or (9, 3)
        Accelerometer error coupling matrix.
    """
    is_series = isinstance(trajectory, pd.Series)
    n_samples = 1 if is_series else len(trajectory)

    V_skew = util.skew_matrix(trajectory[VEL_COLS])
    R = earth.curvature_matrix(trajectory.lat, trajectory.alt)
    Omega_n = earth.rate_n(trajectory.lat)
    rho_n = util.mv_prod(R, trajectory[VEL_COLS])
    g_n = earth.gravity_n(trajectory.lat, trajectory.alt)
    mat_nb = transform.mat_from_rph(trajectory[RPH_COLS])

    F = np.zeros((n_samples, N_STATES, N_STATES))
    samples = np.arange(n_samples)

    F[np.ix_(samples, DR, DV)] = np.eye(3)
    F[np.ix_(samples, DR, PHI)] = V_skew

    F[np.ix_(samples, DV, DV)] = -util.skew_matrix(2 * Omega_n + rho_n)
    F[np.ix_(samples, DV, PHI)] = -util.skew_matrix(g_n)
    F[:, DV3, DR3] = 2 * earth.gravity(trajectory.lat, 0) / earth.A

    F[np.ix_(samples, PHI, DR)] = util.mm_prod(util.skew_matrix(Omega_n), R)
    F[np.ix_(samples, PHI, DV)] = R
    F[np.ix_(samples, PHI, PHI)] = (-util.skew_matrix(rho_n + Omega_n) +
                                    util.mm_prod(R, V_skew))

    B_gyro = np.zeros((n_samples, N_STATES, 3))
    B_gyro[np.ix_(samples, DV, [0, 1, 2])] = util.mm_prod(V_skew, mat_nb)
    B_gyro[np.ix_(samples, PHI, [0, 1, 2])] = -mat_nb

    B_accel = np.zeros((n_samples, N_STATES, 3))
    B_accel[np.ix_(samples, DV, [0, 1, 2])] = mat_nb

    if is_series:
        F = F[0]
        B_gyro = B_gyro[0]
        B_accel = B_accel[0]

    return F, B_gyro, B_accel


def transform_matrix(trajectory):
    """Compute matrix transforming the internal states into output states.

    Output states are comprised of NED position errors, NED velocity errors, roll,
    pitch and heading errors.

    Parameters
    ----------
    trajectory : Trajectory or Pva
        Either full trajectory dataframe or single position-velocity-attitude.

    Returns
    -------
    ndarray, shape (9, 9) or (n_points, 9, 9)
        Transformation matrix or matrices.
    """
    series = isinstance(trajectory, pd.Series)
    if series:
        trajectory = trajectory.to_frame().transpose()

    T = np.zeros((trajectory.shape[0], N_STATES, N_STATES))
    samples = np.arange(len(trajectory))
    T[np.ix_(samples, DR_OUT, DR)] = np.eye(3)
    T[np.ix_(samples, DV_OUT, DV)] = np.eye(3)
    T[np.ix_(samples, DV_OUT, PHI)] = util.skew_matrix(trajectory[VEL_COLS])
    T[np.ix_(samples, DRPH, PHI)] = _phi_to_delta_rph(trajectory[RPH_COLS])

    return T[0] if series else T


def correct_pva(pva, x):
    """Correct position-velocity-attitude with estimated errors.

    Parameters
    ----------
    pva : Pva
        Position-velocity-attitude.
    x : ndarray, shape (9,)
        Error vector.
        Estimates errors in the internal representation.

    Returns
    -------
    Pva
        Corrected position-velocity-attitude.
    """
    mat_tp = Rotation.from_rotvec(x[PHI]).as_matrix()
    lla = transform.perturb_lla(pva[LLA_COLS], -x[DR])
    velocity_n = mat_tp @ (pva[VEL_COLS] - x[DV])
    rph = transform.mat_to_rph(mat_tp @ transform.mat_from_rph(pva[RPH_COLS]))

    return pd.Series(data=np.hstack((lla, velocity_n, rph)), index=pva.index)


def position_error_jacobian(pva, imu_to_antenna_b=None):
    """Compute position error Jacobian matrix.

    This is the matrix which linearly relates the position error in
    NED frame and the error state vector.

    Parameters
    ----------
    pva : Pva
        Position-velocity-attitude.
    imu_to_antenna_b : array_like, shape (3,) or None, optional
        Vector from IMU to antenna (measurement point) expressed in body
        frame. If None, assumed to be zero.

    Returns
    -------
    ndarray, shape (3, 9)
        Jacobian matrix.
    """
    result = np.zeros((3, N_STATES))
    result[:, DR] = np.eye(3)
    if imu_to_antenna_b is not None:
        mat_nb = transform.mat_from_rph(pva[RPH_COLS])
        result[:, PHI] = util.skew_matrix(mat_nb @ imu_to_antenna_b)
    return result


def ned_velocity_error_jacobian(pva, imu_to_antenna_b=None):
    """Compute NED velocity error Jacobian matrix.

    This is the matrix which linearly relates the velocity error in
    NED frame and the error state vector.

    Parameters
    ----------
    pva : Pva
        Position-velocity-attitude. To account for `imu_to_antenna_b` must
        additionally contain elements 'rate_x', 'rate_y', 'rate_z'.
    imu_to_antenna_b : array_like, shape (3,) or None, optional
        Vector from IMU to antenna (measurement point) expressed in body
        frame. If None (default), assumed to be zero.

    Returns
    -------
    ndarray, shape (3, 9)
        Jacobian matrix.
    """
    result = np.zeros((3, N_STATES))
    result[:, DV] = np.eye(3)
    result[:, PHI] = util.skew_matrix(pva[VEL_COLS])
    if imu_to_antenna_b is not None and all(col in pva for col in RATE_COLS):
        mat_nb = transform.mat_from_rph(pva[RPH_COLS])
        result[:, PHI] += mat_nb @ np.cross(pva[RATE_COLS], imu_to_antenna_b)
    return result


def body_velocity_error_jacobian(pva):
    """Compute body velocity error Jacobian matrix.

    This is the matrix which linearly relates the velocity error in body frame and the
    internal error state vector.

    Parameters
    ----------
    pva : Pva
        Position-velocity-attitude.

    Returns
    -------
    ndarray, shape (3, 9)
        Jacobian matrix.
    """
    mat_nb = transform.mat_from_rph(pva[RPH_COLS])
    result = np.zeros((3, N_STATES))
    result[:, DV] = mat_nb.transpose()
    return result


def propagate_errors(trajectory, pva_error=None,
                     gyro_error=np.zeros(3), accel_error=np.zeros(3)):
    """Deterministic linear propagation of INS errors.

    Parameters
    ----------
    trajectory : Trajectory
        Trajectory.
    pva_error : PvaError, optional
        Initial position-velocity-attitude error. If None (default) zeros will be used.
    gyro_error, accel_error : array_like
        Gyro and accelerometer errors (in SI units). Can be constant or
        specified for each time stamp in `trajectory`.

    Returns
    -------
    trajectory_error : TrajectoryError
        Trajectory errors.
    model_error : DataFrame
        Errors expressed using internal states of `error_model`.
    """
    dt = np.diff(trajectory.index)
    Fi, Fig, Fia = system_matrices(trajectory)
    Phi = 0.5 * (Fi[1:] + Fi[:-1]) * dt.reshape(-1, 1, 1)
    Phi[:] += np.identity(Phi.shape[-1])

    gyro_error = util.mv_prod(Fig, gyro_error)
    accel_error = util.mv_prod(Fia, accel_error)
    delta_sensor = 0.5 * (gyro_error[1:] + gyro_error[:-1] +
                          accel_error[1:] + accel_error[:-1])

    T = transform_matrix(trajectory)
    if pva_error is None:
        pva_error = pd.Series(data=np.zeros(9), index=TRAJECTORY_ERROR_COLS)

    x0 = np.hstack([pva_error[NED_COLS].values,
                    pva_error[VEL_COLS].values,
                    pva_error[RPH_COLS].values * transform.DEG_TO_RAD])
    x0 = np.linalg.solve(T[0], x0)

    n_samples = Fi.shape[0]
    x = np.empty((n_samples, N_STATES))
    x[0] = x0
    for i in range(n_samples - 1):
        x[i + 1] = Phi[i].dot(x[i]) + delta_sensor[i] * dt[i]

    model_error = pd.DataFrame(data=x, index=trajectory.index, columns=STATES)
    trajectory_error = pd.DataFrame(data=util.mv_prod(T, x), index=trajectory.index,
                                    columns=TRAJECTORY_ERROR_COLS)
    trajectory_error[RPH_COLS] *= transform.RAD_TO_DEG

    return trajectory_error, model_error

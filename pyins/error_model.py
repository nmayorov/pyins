"""INS error model to use in navigation Kalman filters.

An INS error model is a system of non-stationary (depends on the trajectory) linear
differential equations which describe time evolution of INS errors.

An error model (for full 3D INS) vector has 9 total states: 3 for position, velocity
and attitude errors. The states can be selected in different ways and several error
models were proposed in the literature. Here one particular model is implemented
in class `InsErrorModel,` the details are given there.

Classes
-------
.. autosummary::
    :toctree: generated/

    InsErrorModel

Functions
---------
.. autosummary::
    :toctree: generated/

    propagate_errors
"""
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from . import earth, util, transform
from .util import (LLA_COLS, VEL_COLS, RPH_COLS, RATE_COLS, NED_COLS,
                   TRAJECTORY_ERROR_COLS)


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
    result *= transform.RAD_TO_DEG

    return result[0] if single else result


class InsErrorModel:
    """INS error model.

    The "modified phi-angle" model proposed in [1]_ is used. The key feature of it is
    that the velocity error is measured relative to the true velocity resolved in the
    "platform" frame which eliminates specific force from the system matrix.

    If `with_altitude` is False, the reduced set of 7 errors are used with the
    assumption that altitude and vertical velocity errors are zero.

    Parameters
    ----------
    with_altitude : bool, optional
        Whether to model altitude and vertical velocity errors. Default is True.

    Attributes
    ----------
    with_altitude : bool
        Whether altitude and vertical velocity errors are modelled.
    n_states : int
        Number of states used: 9 or 7 depending on `with_altitude`.

    References
    ----------
    .. [1] Bruno M. Scherzinger and D.Blake Reid "Modified Strapdown Inertial
           Navigator Error Models"
    """
    def __init__(self, with_altitude=True):
        self.with_altitude = with_altitude
        if with_altitude:
            self.states = ['DR1', 'DR2', 'DR3', 'DV1', 'DV2', 'DV3',
                           'PHI1', 'PHI2', 'PHI3']
        else:
            self.states = ['DR1', 'DR2', 'DV1', 'DV2', 'PHI1', 'PHI2', 'PHI3']

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

    TRANSFORM_2D_3D = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
    ])

    def _transform_3d_2d(self, VN, VE):
        is_series = isinstance(VN, pd.Series)
        n = len(VN) if is_series else 1
        result = np.zeros((n, 9, 7))
        result[:] = self.TRANSFORM_2D_3D.transpose()
        result[:, 5, 4] = VE
        result[:, 5, 5] = -VN
        return result if is_series else result[0]

    @property
    def n_states(self):
        return len(self.states)

    def system_matrices(self, trajectory):
        """Compute matrices which govern the error model differential equations.

        The system of differential equations has the form::

            dx/dt = F @ x + B_gyro @ gyro_error + B_accel @ accel_error

        Where

            - ``x`` - error vector
            - ``gyro_error``, ``accel_error`` - vectors with gyro and accelerometer errors
            - ``F`` - error dynamics matrix
            - ``B_gyro`` - gyro error coupling matrix
            - ``B_accel`` - accel error coupling matrix

        Parameters
        ----------
        trajectory : Trajectory or Pva
            Either full trajectory dataframe or single position-velocity-attitude.

        Returns
        -------
        F : ndarray, shape (n, n_states, n_states) or (n_states, n_states)
            Error dynamics matrix.
        B_gyro : ndarray, shape (n, n_states, 3) or (n_states, 3)
            Gyro error coupling matrix.
        B_accel : ndarray, shape (n, n_states, 3) or (n_states, 3)
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

        F = np.zeros((n_samples, 9, 9))
        samples = np.arange(n_samples)

        F[np.ix_(samples, self.DR, self.DV)] = np.eye(3)
        F[np.ix_(samples, self.DR, self.PHI)] = V_skew

        F[np.ix_(samples, self.DV, self.DV)] = -util.skew_matrix(2 * Omega_n + rho_n)
        F[np.ix_(samples, self.DV, self.PHI)] = -util.skew_matrix(g_n)
        F[:, self.DV3, self.DR3] = 2 * earth.gravity(trajectory.lat, 0) / earth.A

        F[np.ix_(samples, self.PHI, self.DR)] = util.mm_prod(util.skew_matrix(Omega_n),
                                                             R)
        F[np.ix_(samples, self.PHI, self.DV)] = R
        F[np.ix_(samples, self.PHI, self.PHI)] = (-util.skew_matrix(rho_n + Omega_n) +
                                                  util.mm_prod(R, V_skew))

        B_gyro = np.zeros((n_samples, 9, 3))
        B_gyro[np.ix_(samples, self.DV, [0, 1, 2])] = util.mm_prod(V_skew, mat_nb)
        B_gyro[np.ix_(samples, self.PHI, [0, 1, 2])] = -mat_nb

        B_accel = np.zeros((n_samples, 9, 3))
        B_accel[np.ix_(samples, self.DV, [0, 1, 2])] = mat_nb

        if is_series:
            F = F[0]
            B_gyro = B_gyro[0]
            B_accel = B_accel[0]

        if not self.with_altitude:
            T_2d_3d = self.TRANSFORM_2D_3D
            T_3d_2d = self._transform_3d_2d(trajectory.VN, trajectory.VE)
            F = util.mm_prod(util.mm_prod(T_2d_3d, F), T_3d_2d)
            B_gyro = util.mm_prod(T_2d_3d, B_gyro)
            B_accel = util.mm_prod(T_2d_3d, B_accel)

        return F, B_gyro, B_accel

    @classmethod
    def _transform_to_output_3d(cls, trajectory):
        series = isinstance(trajectory, pd.Series)
        if series:
            trajectory = trajectory.to_frame().transpose()

        result = np.zeros((trajectory.shape[0], 9, 9))
        samples = np.arange(len(trajectory))
        result[np.ix_(samples, cls.DR_OUT, cls.DR)] = np.eye(3)
        result[np.ix_(samples, cls.DV_OUT, cls.DV)] = np.eye(3)
        result[np.ix_(samples, cls.DV_OUT, cls.PHI)] = util.skew_matrix(
            trajectory[VEL_COLS])
        result[np.ix_(samples, cls.DRPH, cls.PHI)] = _phi_to_delta_rph(
            trajectory[RPH_COLS])

        if series:
            result = result[0]

        return result

    def transform_to_output(self, trajectory):
        """Compute matrix transforming the internal states into output states.

        Output states are comprised of NED position errors, NED velocity errors, roll,
        pitch and heading errors.

        Parameters
        ----------
        trajectory : Trajectory or Pva
            Either full trajectory dataframe or single position-velocity-attitude.

        Returns
        -------
        ndarray, shape (9, n_states) or (n, 9, n_states)
            Transformation matrix or matrices.
        """
        result = self._transform_to_output_3d(trajectory)
        if not self.with_altitude:
            result = util.mm_prod(result,
                                  self._transform_3d_2d(trajectory.VN, trajectory.VE))
        return result

    def transform_to_internal(self, pva):
        """Compute matrix transforming the output states into internal states.

        Output states are comprised of NED position errors, NED velocity errors, roll,
        pitch and heading errors.

        Parameters
        ----------
        pva : Pva
            Position-velocity-attitude.

        Returns
        -------
        ndarray, shape (n_states, 9)
            Transformation matrix or matrices.
        """
        result = np.linalg.inv(self._transform_to_output_3d(pva))
        if not self.with_altitude:
            result = self.TRANSFORM_2D_3D @ result
        return result

    def correct_pva(self, pva, x):
        """Correct position-velocity-attitude with estimated errors.

        Parameters
        ----------
        pva : Pva
            Position-velocity-attitude.
        x : ndarray, shape (n_states,)
            Error vector. Estimates errors in the internal representation.

        Returns
        -------
        Pva
            Corrected position-velocity-attitude.
        """
        if not self.with_altitude:
            x = self._transform_3d_2d(pva.VN, pva.VE) @ x
        mat_tp = Rotation.from_rotvec(x[self.PHI]).as_matrix()
        lla = transform.perturb_lla(pva[LLA_COLS], -x[self.DR])
        velocity_n = mat_tp @ (pva[VEL_COLS] - x[self.DV])
        if not self.with_altitude:
            velocity_n[2] = pva.VD
        rph = transform.mat_to_rph(mat_tp @ transform.mat_from_rph(pva[RPH_COLS]))
        return pd.Series(data=np.hstack((lla, velocity_n, rph)), index=pva.index)

    def position_error_jacobian(self, pva, imu_to_antenna_b=None):
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
        ndarray, shape (3, 9) or (2, 7)
            Jacobian matrix. When `with_altitude` is False it has only 2 rows for
            North and East position errors.
        """
        result = np.zeros((3, 9))
        result[:, self.DR] = np.eye(3)
        if imu_to_antenna_b is not None:
            mat_nb = transform.mat_from_rph(pva[RPH_COLS])
            result[:, self.PHI] = util.skew_matrix(mat_nb @ imu_to_antenna_b)
        if not self.with_altitude:
            result = result @ self._transform_3d_2d(pva.VN, pva.VE)
            result = result[:2]
        return result

    def ned_velocity_error_jacobian(self, pva, imu_to_antenna_b=None):
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
        ndarray, shape (3, 9) or (2, 7)
            Jacobian matrix. When `with_altitude` is False it has only 2 rows for
            North and East velocity errors.
        """
        result = np.zeros((3, 9))
        result[:, self.DV] = np.eye(3)
        velocity_n = pva[VEL_COLS].values.copy()
        if imu_to_antenna_b is not None and all(col in pva for col in RATE_COLS):
            mat_nb = transform.mat_from_rph(pva[RPH_COLS])
            velocity_n += mat_nb @ np.cross(pva[RATE_COLS], imu_to_antenna_b)
        result[:, self.PHI] = util.skew_matrix(velocity_n)
        if not self.with_altitude:
            result = result @ self._transform_3d_2d(pva.VN, pva.VE)
            result = result[:2]
        return result

    def body_velocity_error_jacobian(self, pva):
        """Compute body velocity error Jacobian matrix.

        This is the matrix which linearly relates the velocity error in body frame and
        the internal error state vector.

        Parameters
        ----------
        pva : Pva
            Position-velocity-attitude.

        Returns
        -------
        ndarray, shape (3, n_states)
            Jacobian matrix.
        """
        mat_nb = transform.mat_from_rph(pva[RPH_COLS])
        result = np.zeros((3, 9))
        result[:, self.DV] = mat_nb.transpose()
        if not self.with_altitude:
            result = result @ self._transform_3d_2d(pva.VN, pva.VE)
        return result


def propagate_errors(trajectory, pva_error=None,
                     gyro_error=np.zeros(3), accel_error=np.zeros(3),
                     with_altitude=True):
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
    with_altitude : bool, optional
        Whether to consider altitude and vertical velocity errors.
        Default is True.

    Returns
    -------
    trajectory_error : TrajectoryError
        Trajectory errors.
    model_error : DataFrame
        Errors expressed using internal states of `error_model`.
    """
    error_model = InsErrorModel(with_altitude)
    dt = np.diff(trajectory.index)
    Fi, Fig, Fia = error_model.system_matrices(trajectory)
    Phi = 0.5 * (Fi[1:] + Fi[:-1]) * dt.reshape(-1, 1, 1)
    Phi[:] += np.identity(Phi.shape[-1])

    gyro_error = util.mv_prod(Fig, gyro_error)
    accel_error = util.mv_prod(Fia, accel_error)
    delta_sensor = 0.5 * (gyro_error[1:] + gyro_error[:-1] +
                          accel_error[1:] + accel_error[:-1])

    if pva_error is None:
        pva_error = pd.Series(data=np.zeros(9), index=TRAJECTORY_ERROR_COLS)

    x0 = error_model.transform_to_internal(trajectory.iloc[0]) @ pva_error.values

    n_samples = Fi.shape[0]
    x = np.empty((n_samples, error_model.n_states))
    x[0] = x0
    for i in range(n_samples - 1):
        x[i + 1] = Phi[i].dot(x[i]) + delta_sensor[i] * dt[i]

    model_error = pd.DataFrame(data=x, index=trajectory.index,
                               columns=error_model.states)
    T = error_model.transform_to_output(trajectory)
    trajectory_error = pd.DataFrame(data=util.mv_prod(T, x), index=trajectory.index,
                                    columns=TRAJECTORY_ERROR_COLS)

    return trajectory_error, model_error

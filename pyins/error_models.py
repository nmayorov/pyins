"""INS error models to use in EKF-like estimation filters."""
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

    return result[0] if single else result


class InsErrorModel:
    """INS error model interface.

    INS error model is a system of non-stationary linear differential equations
    which describe time evolution on INS errors. The system matrix depends
    on the trajectory.

    We consider error models consisting of 9 total states: 3 for position,
    velocity and attitude errors.

    The states can be selected in different  manners and several error models
    were proposed in the literature.

    The output error states are always NED position error, NED velocity error and errors
    of roll, pitch and heading angles.

    For education purposes two models implementing this interface are
    provided:

        - `ModifiedPhiModel`
        - `ModifiedPsiModel`

    Attributes
    ----------
    N_STATES : int
        Number of states used in error models. This value is always equal to 9.
    STATES : dict
        Mapping from the internal error states names to their indices in the
        state vector.
    """
    N_STATES = 9
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

    STATES = None

    def system_matrix(self, trajectory):
        """Compute the system matrix.

        Parameters
        ----------
        trajectory : Trajectory or Pva
            Either full trajectory dataframe or single position-velocity-attitude.

        Returns
        -------
        system_matrix : ndarray, shape (n_points, N_STATES, N_STATES) or (N_STATES, N_STATES)
        """
        raise NotImplementedError

    def transform_to_output(self, trajectory):
        """Compute matrix transforming the internal states into output states.

        Parameters
        ----------
        trajectory : Trajectory or Pva
            Either full trajectory dataframe or single position-velocity-attitude.

        Returns
        -------
        transform_matrix : ndarray, shape (N_STATES, N_STATES) or (n_points, N_STATES, N_STATES)
        """
        raise NotImplementedError

    def correct_pva(self, pva, error):
        """Correct position-velocity-attitude with estimated errors.

        Parameters
        ----------
        pva : Pva
            Position-velocity-attitude.
        error : array_like, shape (N_STATES,)
            Estimates errors in the internal representation.

        Returns
        -------
        corrected_pva : pd.Series
        """
        raise NotImplementedError

    def position_error_jacobian(self, pva, imu_to_antenna_b=None):
        """Compute position error Jacobian matrix.

        This is the matrix which linearly relates the position error in
        NED frame and the internal error state vector.

        Parameters
        ----------
        pva : Pva
            Position-velocity-attitude.
        imu_to_antenna_b : array_like, shape (3,) or None, optional
            Vector from IMU to antenna (measurement point) expressed in body
            frame. If None, assumed to be zero.

        Returns
        -------
        jacobian : ndarray, shape (3, N_STATES)
        """
        raise NotImplementedError

    def ned_velocity_error_jacobian(self, pva, imu_to_antenna_b=None):
        """Compute NED velocity error Jacobian matrix.

        This is the matrix which linearly relates the velocity error in
        NED frame and the internal error state vector.

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
        jacobian : ndarray, shape (3, N_STATES)
        """
        raise NotImplementedError

    def body_velocity_error_jacobian(self, pva):
        """Compute body velocity error Jacobian matrix.

        This is the matrix which linearly relates the velocity error in
        body frame and the internal error state vector.

        Parameters
        ----------
        pva : Pva
            Position-velocity-attitude.

        Returns
        -------
        jacobian : ndarray, shape (3, N_STATES)
        """
        raise NotImplementedError


class ModifiedPhiModel(InsErrorModel):
    """Error model with phi-angle error and modified velocity errors.

    The phi-angle is used to describe attitude errors and velocity error is
    measured relative to the true velocity resolved in the platform frame.
    The latter trick eliminates specific force from the equations which makes
    the implementation much more convenient. See [1]_ for a detailed discussion
    and derivation.

    References
    ----------
    .. [1] Bruno M. Scherzinger and D.Blake Reid
           "Modified Strapdown Inertial Navigator Error Models".
    """
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

    STATES = dict(DR1=DR1, DR2=DR2, DR3=DR3,
                  DV1=DV1, DV2=DV2, DV3=DV3,
                  PHI1=PHI1, PHI2=PHI2, PHI3=PHI3)

    def system_matrix(self, trajectory):
        is_series = isinstance(trajectory, pd.Series)
        n_samples = 1 if is_series else len(trajectory)

        V_skew = util.skew_matrix(trajectory[VEL_COLS])
        R = earth.curvature_matrix(trajectory.lat, trajectory.alt)
        Omega_n = earth.rate_n(trajectory.lat)
        rho_n = util.mv_prod(R, trajectory[VEL_COLS])
        g_n = earth.gravity_n(trajectory.lat, trajectory.alt)
        mat_nb = transform.mat_from_rph(trajectory[RPH_COLS])

        F = np.zeros((n_samples, self.N_STATES, self.N_STATES))
        samples = np.arange(n_samples)

        F[np.ix_(samples, self.DR, self.DV)] = np.eye(3)
        F[np.ix_(samples, self.DR, self.PHI)] = V_skew

        F[np.ix_(samples, self.DV, self.DV)] = -util.skew_matrix(
            2 * Omega_n + rho_n)
        F[np.ix_(samples, self.DV, self.PHI)] = -util.skew_matrix(g_n)
        F[:, self.DV3, self.DR3] = (2 * earth.gravity(trajectory.lat, 0)
                                    / earth.A)

        F[np.ix_(samples, self.PHI, self.DR)] = util.mm_prod(
            util.skew_matrix(Omega_n), R)
        F[np.ix_(samples, self.PHI, self.DV)] = R
        F[np.ix_(samples, self.PHI, self.PHI)] = \
            -util.skew_matrix(rho_n + Omega_n) + util.mm_prod(R, V_skew)

        B_gyro = np.zeros((n_samples, self.N_STATES, 3))
        B_gyro[np.ix_(samples, self.DV, [0, 1, 2])] = util.mm_prod(V_skew, mat_nb)
        B_gyro[np.ix_(samples, self.PHI, [0, 1, 2])] = -mat_nb

        B_accel = np.zeros((n_samples, self.N_STATES, 3))
        B_accel[np.ix_(samples, self.DV, [0, 1, 2])] = mat_nb

        if is_series:
            F = F[0]
            B_gyro = B_gyro[0]
            B_accel = B_accel[0]

        return F, B_gyro, B_accel

    def transform_to_output(self, trajectory):
        series = isinstance(trajectory, pd.Series)
        if series:
            trajectory = trajectory.to_frame().transpose()

        T = np.zeros((trajectory.shape[0], self.N_STATES, self.N_STATES))
        samples = np.arange(len(trajectory))
        T[np.ix_(samples, self.DR_OUT, self.DR)] = np.eye(3)
        T[np.ix_(samples, self.DV_OUT, self.DV)] = np.eye(3)
        T[np.ix_(samples, self.DV_OUT, self.PHI)] = util.skew_matrix(
            trajectory[VEL_COLS])
        T[np.ix_(samples, self.DRPH, self.PHI)] = _phi_to_delta_rph(
            trajectory[RPH_COLS])

        return T[0] if series else T

    def correct_pva(self, pva, error):
        Ctp = Rotation.from_rotvec(error[self.PHI]).as_matrix()
        lla = transform.perturb_lla(pva[LLA_COLS], -error[self.DR])
        velocity_n = Ctp @ (pva[VEL_COLS] - error[self.DV])
        rph = transform.mat_to_rph(Ctp @ transform.mat_from_rph(
            pva[RPH_COLS]))

        return pd.Series(data=np.hstack((lla, velocity_n, rph)),
                         index=pva.index)

    def position_error_jacobian(self, pva, imu_to_antenna_b=None):
        result = np.zeros((3, self.N_STATES))
        result[:, self.DR] = np.eye(3)
        if imu_to_antenna_b is not None:
            mat_nb = transform.mat_from_rph(pva[RPH_COLS])
            result[:, self.PHI] = util.skew_matrix(mat_nb @ imu_to_antenna_b)
        return result

    def ned_velocity_error_jacobian(self, pva, imu_to_antenna_b=None):
        result = np.zeros((3, self.N_STATES))
        result[:, self.DV] = np.eye(3)
        result[:, self.PHI] = util.skew_matrix(pva[VEL_COLS])
        if imu_to_antenna_b is not None and all(col in pva for col in RATE_COLS):
            mat_nb = transform.mat_from_rph(pva[RPH_COLS])
            result[:, self.PHI] += mat_nb @ np.cross(pva[RATE_COLS], imu_to_antenna_b)
        return result

    def body_velocity_error_jacobian(self, pva):
        mat_nb = transform.mat_from_rph(pva[RPH_COLS])
        result = np.zeros((3, self.N_STATES))
        result[:, self.DV] = mat_nb.transpose()
        return result


class ModifiedPsiModel(InsErrorModel):
    """Error model with psi-angle error and modified velocity errors.

    The psi-angle is used to describe attitude errors and velocity error is
    measured relative to the true velocity resolved in the platform frame.
    The latter trick eliminates specific force from the equations which makes
    the implementation much more convenient. See [1]_ for a detailed discussion
    and derivation.

    References
    ----------
    .. [1] Bruno M. Scherzinger and D.Blake Reid
           "Modified Strapdown Inertial Navigator Error Models".
    """
    DR1 = 0
    DR2 = 1
    DR3 = 2
    DV1 = 3
    DV2 = 4
    DV3 = 5
    PSI1 = 6
    PSI2 = 7
    PSI3 = 8

    DR = [DR1, DR2, DR3]
    DV = [DV1, DV2, DV3]
    PSI = [PSI1, PSI2, PSI3]

    STATES = dict(DR1=DR1, DR2=DR2, DR3=DR3,
                  DV1=DV1, DV2=DV2, DV3=DV3,
                  PSI1=PSI1, PSI2=PSI1, PSI3=PSI1)

    def system_matrix(self, trajectory):
        is_series = isinstance(trajectory, pd.Series)
        n_samples = 1 if is_series else len(trajectory)

        V_skew = util.skew_matrix(trajectory[VEL_COLS])
        R = earth.curvature_matrix(trajectory.lat, trajectory.alt)
        Omega_n = earth.rate_n(trajectory.lat)
        rho_n = util.mv_prod(R, trajectory[VEL_COLS])
        g_n_skew = util.skew_matrix(earth.gravity_n(trajectory.lat,
                                                    trajectory.alt))
        mat_nb = transform.mat_from_rph(trajectory[RPH_COLS])

        F = np.zeros((n_samples, self.N_STATES, self.N_STATES))
        samples = np.arange(n_samples)

        F[np.ix_(samples, self.DR, self.DV)] = np.eye(3)
        F[np.ix_(samples, self.DR, self.PSI)] = V_skew

        F[np.ix_(samples, self.DV, self.DV)] = -util.skew_matrix(
            2 * Omega_n + rho_n)
        F[np.ix_(samples, self.DV, self.DR)] = -g_n_skew @ R
        F[np.ix_(samples, self.DV, self.PSI)] = -g_n_skew
        F[:, self.DV3, self.DR3] += (2 * earth.gravity(trajectory.lat, 0)
                                     / earth.A)

        F[np.ix_(samples, self.PSI, self.PSI)] = -util.skew_matrix(rho_n +
                                                                   Omega_n)

        B_gyro = np.zeros((n_samples, self.N_STATES, 3))
        B_gyro[np.ix_(samples, self.DV, [0, 1, 2])] = util.mm_prod(V_skew, mat_nb)
        B_gyro[np.ix_(samples, self.PSI, [0, 1, 2])] = -mat_nb

        B_accel = np.zeros((n_samples, self.N_STATES, 3))
        B_accel[np.ix_(samples, self.DV, [0, 1, 2])] = mat_nb

        if is_series:
            F = F[0]
            B_gyro = B_gyro[0]
            B_accel = B_accel[0]

        return F, B_gyro, B_accel

    def transform_to_output(self, trajectory):
        series = isinstance(trajectory, pd.Series)
        if series:
            trajectory = trajectory.to_frame().transpose()

        T = np.zeros((trajectory.shape[0], self.N_STATES, self.N_STATES))
        samples = np.arange(len(trajectory))
        T[np.ix_(samples, self.DR_OUT, self.DR)] = np.eye(3)
        T[np.ix_(samples, self.DV_OUT, self.DV)] = np.eye(3)
        T[np.ix_(samples, self.DV_OUT, self.PSI)] = util.skew_matrix(
            trajectory[VEL_COLS])

        T_rph_phi = _phi_to_delta_rph(trajectory[RPH_COLS])
        R = earth.curvature_matrix(trajectory.lat, trajectory.alt)
        T[np.ix_(samples, self.DRPH, self.PSI)] = T_rph_phi
        T[np.ix_(samples, self.DRPH, self.DR)] = util.mm_prod(T_rph_phi, R)

        return T[0] if series else T

    def correct_pva(self, pva, error):
        R = earth.curvature_matrix(pva.lat, pva.alt)
        Ctp = Rotation.from_rotvec(error[self.PSI] +
                                   R @ error[self.DR]).as_matrix()
        lla = transform.perturb_lla(pva[LLA_COLS], -error[self.DR])
        velocity_n = Ctp @ (pva[VEL_COLS] - error[self.DV])
        rph = transform.mat_to_rph(Ctp @ transform.mat_from_rph(
            pva[RPH_COLS]))

        return pd.Series(data=np.hstack((lla, velocity_n, rph)),
                         index=pva.index)

    def position_error_jacobian(self, pva,
                                imu_to_antenna_b=None):
        result = np.zeros((3, self.N_STATES))
        result[:, self.DR] = np.eye(3)
        if imu_to_antenna_b is not None:
            mat_nb = transform.mat_from_rph(pva[RPH_COLS])
            S = util.skew_matrix(mat_nb @ imu_to_antenna_b)
            F = earth.curvature_matrix(pva.lat,
                                       pva.alt)
            result[:, self.DR] += S @ F
            result[:, self.PSI] = S

        return result

    def ned_velocity_error_jacobian(self, pva, imu_to_antenna_b=None):
        result = np.zeros((3, self.N_STATES))
        result[:, self.DV] = np.eye(3)
        result[:, self.PSI] = util.skew_matrix(pva[VEL_COLS])
        if imu_to_antenna_b is not None and all(col in pva for col in RATE_COLS):
            mat_nb = transform.mat_from_rph(pva[RPH_COLS])
            result[:, self.PHI] += mat_nb @ np.cross(pva[RATE_COLS], imu_to_antenna_b)
        return result

    def body_velocity_error_jacobian(self, pva):
        mat_nb = transform.mat_from_rph(pva[RPH_COLS])
        result = np.zeros((3, self.N_STATES))
        result[:, self.DV] = mat_nb.transpose()
        return result


def propagate_errors(trajectory, pva_error=None, gyro_error=np.zeros(3),
                     accel_error=np.zeros(3), error_model=ModifiedPhiModel()):
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
    error_model : InsErrorModel
        Error model object to use for the propagation.

    Returns
    -------
    trajectory_error : TrajectoryError
        Trajectory errors.
    model_error : DataFrame
        Errors expressed using internal states of `error_model`.
    """
    dt = np.diff(trajectory.index)
    Fi, Fig, Fia = error_model.system_matrix(trajectory)
    Phi = 0.5 * (Fi[1:] + Fi[:-1]) * dt.reshape(-1, 1, 1)
    Phi[:] += np.identity(Phi.shape[-1])

    gyro_error = util.mv_prod(Fig, gyro_error)
    accel_error = util.mv_prod(Fia, accel_error)
    delta_sensor = 0.5 * (gyro_error[1:] + gyro_error[:-1] +
                          accel_error[1:] + accel_error[:-1])

    T = error_model.transform_to_output(trajectory)
    if pva_error is None:
        pva_error = pd.Series(data=np.zeros(9), index=TRAJECTORY_ERROR_COLS)

    x0 = np.hstack([pva_error[NED_COLS].values,
                    pva_error[VEL_COLS].values,
                    pva_error[RPH_COLS].values * transform.DEG_TO_RAD])
    x0 = np.linalg.solve(T[0], x0)

    n_samples = Fi.shape[0]
    x = np.empty((n_samples, error_model.N_STATES))
    x[0] = x0
    for i in range(n_samples - 1):
        x[i + 1] = Phi[i].dot(x[i]) + delta_sensor[i] * dt[i]

    model_error = pd.DataFrame(data=x, index=trajectory.index,
                               columns=list(error_model.STATES.keys()))
    trajectory_error = pd.DataFrame(data=util.mv_prod(T, x), index=trajectory.index,
                                    columns=TRAJECTORY_ERROR_COLS)
    trajectory_error[RPH_COLS] *= transform.RAD_TO_DEG

    return trajectory_error, model_error

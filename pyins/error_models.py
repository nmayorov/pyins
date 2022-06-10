"""INS error models to use in EKF-like estimation filters."""
from collections import OrderedDict
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from . import earth, util, transform


class InsErrorModel:
    """INS error model interface.

    INS error model is a system of non-stationary linear differential equations
    which describe time evolution on INS errors. The system matrix depends
    on the navigation state (trajectory).

    We consider error models consisting of 9 total states: 3 for position,
    velocity and attitude errors.

    The states can be selected in different  manners and several error models
    were proposed in the literature.

    For education purposes two models implementing this interface are
    provided:

        - `ModifiedPhiModel`
        - `ModifiedPsiModel`

    Attributes
    ----------
    N_STATES : int
        Number of states used in error models. This value is always equal to 9.
    STATES : OrderedDict
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
        trajectory : pd.DataFrame
            Trajectory.

        Returns
        -------
        system_matrix : ndarray, shape (n_points, N_STATES, N_STATES)
        """
        raise NotImplementedError

    def transform_to_output(self, trajectory):
        """Compute matrix transforming the internal states into output states.

        Parameters
        ----------
        trajectory : pd.DataFrame or pd.Series
            Trajectory or a single trajectory point.

        Returns
        -------
        transform_matrix : ndarray, shape (N_STATES, N_STATES) or (n_points, N_STATES, N_STATES)
        """
        raise NotImplementedError

    def correct_state(self, trajectory_point, error):
        """Correct navigation state with estimated errors.

        Parameters
        ----------
        trajectory_point : pd.Series
            Point of trajectory.
        error : ndarray
            Estimates errors. First N_STATES components are assumed to
            contain error states.

        Returns
        -------
        corrected_trajectory_point : pd.Series
        """
        raise NotImplementedError

    def position_error_jacobian(self, trajectory_point,
                                imu_to_antenna_b=None):
        """Compute position error Jacobian matrix.

        This is the matrix which linearly relates the position error in
        NED frame and the internal error state vector.

        Parameters
        ----------
        trajectory_point : pd.Series
            Point of trajectory.
        imu_to_antenna_b : array_like or None, optional
            Vector from IMU to antenna (measurement point) expressed in body
            frame. If None, assumed to be zero.

        Returns
        -------
        jacobian : ndarray, shape (3, N_STATES)
        """
        raise NotImplementedError

    def ned_velocity_error_jacobian(self, trajectory_point):
        """Compute NED velocity error Jacobian matrix.

        This is the matrix which linearly relates the velocity error in
        NED frame and the internal error state vector.

        Parameters
        ----------
        trajectory_point : pd.Series
            Point of trajectory.

        Returns
        -------
        jacobian : ndarray, shape (3, N_STATES)
        """

        raise NotImplementedError

    def body_velocity_error_jacobian(self, trajectory_point):
        """Compute body velocity error Jacobian matrix.

        This is the matrix which linearly relates the velocity error in
        body frame and the internal error state vector.

        Parameters
        ----------
        trajectory_point : pd.Series
            Point of trajectory.

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

    STATES = OrderedDict(DR1=DR1, DR2=DR2, DR3=DR3,
                         DV1=DV1, DV2=DV2, DV3=DV3,
                         PHI1=PHI1, PHI2=PHI2, PHI3=PHI3)

    def system_matrix(self, trajectory):
        n_samples = trajectory.shape[0]

        V_skew = util.skew_matrix(trajectory[['VN', 'VE', 'VD']])
        R = earth.curvature_matrix(trajectory.lat, trajectory.alt)
        Omega_n = earth.rate_n(trajectory.lat)
        rho_n = util.mv_prod(R, trajectory[['VN', 'VE', 'VD']])
        g_n = earth.gravity_n(trajectory.lat, trajectory.alt)
        Cnb = transform.mat_from_rph(trajectory[['roll', 'pitch', 'heading']])

        F = np.zeros((n_samples, self.N_STATES, self.N_STATES))
        samples = np.arange(n_samples)

        F[np.ix_(samples, self.DR, self.DV)] = np.eye(3)
        F[np.ix_(samples, self.DR, self.PHI)] = V_skew

        F[np.ix_(samples, self.DV, self.DV)] = -util.skew_matrix(
            2 * Omega_n + rho_n)
        F[np.ix_(samples, self.DV, self.PHI)] = -util.skew_matrix(g_n)
        F[:, self.DV3, self.DR3] = (2 * earth.gravity(trajectory.lat, 0)
                                    / earth.R0)

        F[np.ix_(samples, self.PHI, self.DR)] = util.mm_prod(
            util.skew_matrix(Omega_n), R)
        F[np.ix_(samples, self.PHI, self.DV)] = R
        F[np.ix_(samples, self.PHI, self.PHI)] = \
            -util.skew_matrix(rho_n + Omega_n) + util.mm_prod(R, V_skew)

        B_gyro = np.zeros((n_samples, self.N_STATES, 3))
        B_gyro[np.ix_(samples, self.DV, [0, 1, 2])] = util.mm_prod(V_skew, Cnb)
        B_gyro[np.ix_(samples, self.PHI, [0, 1, 2])] = -Cnb

        B_accel = np.zeros((n_samples, self.N_STATES, 3))
        B_accel[np.ix_(samples, self.DV, [0, 1, 2])] = Cnb

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
            trajectory[['VN', 'VE', 'VD']])
        T[np.ix_(samples, self.DRPH, self.PHI)] = transform.phi_to_delta_rph(
            trajectory[['roll', 'pitch', 'heading']])

        return T[0] if series else T

    def correct_state(self, trajectory_point, error):
        Ctp = Rotation.from_rotvec(error[self.PHI]).as_matrix()
        lla = transform.perturb_lla(trajectory_point[['lat', 'lon', 'alt']],
                                    -error[self.DR])
        velocity_n = Ctp @ (trajectory_point[['VN', 'VE', 'VD']]
                            - error[self.DV])
        rph = transform.mat_to_rph(Ctp @ transform.mat_from_rph(
            trajectory_point[['roll', 'pitch', 'heading']]))

        return pd.Series(data=np.hstack((lla, velocity_n, rph)),
                         index=trajectory_point.index)

    def position_error_jacobian(self, trajectory_point, imu_to_antenna_b=None):
        result = np.zeros((3, self.N_STATES))
        result[:, self.DR] = np.eye(3)
        if imu_to_antenna_b is not None:
            mat_nb = transform.mat_from_rph(
                trajectory_point[['roll', 'pitch', 'heading']])
            result[:, self.PHI] = util.skew_matrix(mat_nb @ imu_to_antenna_b)
        return result

    def ned_velocity_error_jacobian(self, trajectory_point):
        result = np.zeros((3, self.N_STATES))
        result[:, self.DV] = np.eye(3)
        result[:, self.PHI] = util.skew_matrix(
            trajectory_point[['VN', 'VE', 'VD']])
        return result

    def body_velocity_error_jacobian(self, trajectory_point):
        Cnb = transform.mat_from_rph(
            trajectory_point[['roll', 'pitch', 'heading']])
        result = np.zeros((3, self.N_STATES))
        result[:, self.DV] = Cnb.transpose()
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

    STATES = OrderedDict(DR1=DR1, DR2=DR2, DR3=DR3,
                         DV1=DV1, DV2=DV2, DV3=DV3,
                         PSI1=PSI1, PSI2=PSI1, PSI3=PSI1)

    def system_matrix(self, trajectory):
        n_samples = trajectory.shape[0]

        V_skew = util.skew_matrix(trajectory[['VN', 'VE', 'VD']])
        R = earth.curvature_matrix(trajectory.lat, trajectory.alt)
        Omega_n = earth.rate_n(trajectory.lat)
        rho_n = util.mv_prod(R, trajectory[['VN', 'VE', 'VD']])
        g_n_skew = util.skew_matrix(earth.gravity_n(trajectory.lat,
                                                    trajectory.alt))
        Cnb = transform.mat_from_rph(trajectory[['roll', 'pitch', 'heading']])

        F = np.zeros((n_samples, self.N_STATES, self.N_STATES))
        samples = np.arange(n_samples)

        F[np.ix_(samples, self.DR, self.DV)] = np.eye(3)
        F[np.ix_(samples, self.DR, self.PSI)] = V_skew

        F[np.ix_(samples, self.DV, self.DV)] = -util.skew_matrix(
            2 * Omega_n + rho_n)
        F[np.ix_(samples, self.DV, self.DR)] = -g_n_skew @ R
        F[np.ix_(samples, self.DV, self.PSI)] = -g_n_skew
        F[:, self.DV3, self.DR3] += (2 * earth.gravity(trajectory.lat, 0)
                                     / earth.R0)

        F[np.ix_(samples, self.PSI, self.PSI)] = -util.skew_matrix(rho_n +
                                                                   Omega_n)

        B_gyro = np.zeros((n_samples, self.N_STATES, 3))
        B_gyro[np.ix_(samples, self.DV, [0, 1, 2])] = util.mm_prod(V_skew, Cnb)
        B_gyro[np.ix_(samples, self.PSI, [0, 1, 2])] = -Cnb

        B_accel = np.zeros((n_samples, self.N_STATES, 3))
        B_accel[np.ix_(samples, self.DV, [0, 1, 2])] = Cnb

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
            trajectory[['VN', 'VE', 'VD']])

        T_rph_phi = transform.phi_to_delta_rph(
            trajectory[['roll', 'pitch', 'heading']])
        R = earth.curvature_matrix(trajectory.lat, trajectory.alt)
        T[np.ix_(samples, self.DRPH, self.PSI)] = T_rph_phi
        T[np.ix_(samples, self.DRPH, self.DR)] = util.mm_prod(T_rph_phi, R)

        return T[0] if series else T

    def correct_state(self, trajectory_point, error):
        R = earth.curvature_matrix(trajectory_point.lat, trajectory_point.alt)
        Ctp = Rotation.from_rotvec(error[self.PSI] +
                                   R @ error[self.DR]).as_matrix()
        lla = transform.perturb_lla(trajectory_point[['lat', 'lon', 'alt']],
                                    -error[self.DR])
        velocity_n = Ctp @ (trajectory_point[['VN', 'VE', 'VD']]
                            - error[self.DV])
        rph = transform.mat_to_rph(Ctp @ transform.mat_from_rph(
            trajectory_point[['roll', 'pitch', 'heading']]))

        return pd.Series(data=np.hstack((lla, velocity_n, rph)),
                         index=trajectory_point.index)

    def position_error_jacobian(self, trajectory_point,
                                imu_to_antenna_b=None):
        result = np.zeros((3, self.N_STATES))
        result[:, self.DR] = np.eye(3)
        if imu_to_antenna_b is not None:
            mat_nb = transform.mat_from_rph(
                trajectory_point[['roll', 'pitch', 'heading']])
            S = util.skew_matrix(mat_nb @ imu_to_antenna_b)
            F = earth.curvature_matrix(trajectory_point.lat,
                                       trajectory_point.alt)
            result[:, self.DR] += S @ F
            result[:, self.PSI] = S

        return result

    def ned_velocity_error_jacobian(self, trajectory_point):
        result = np.zeros((3, self.N_STATES))
        result[:, self.DV] = np.eye(3)
        result[:, self.PSI] = util.skew_matrix(
            trajectory_point[['VN', 'VE', 'VD']])
        return result

    def body_velocity_error_jacobian(self, trajectory_point):
        Cnb = transform.mat_from_rph(
            trajectory_point[['roll', 'pitch', 'heading']])
        result = np.zeros((3, self.N_STATES))
        result[:, self.DV] = Cnb.transpose()
        return result


def propagate_errors(dt, trajectory,
                     delta_position_n=np.zeros(3),
                     delta_velocity_n=np.zeros(3),
                     delta_rph=np.zeros(3),
                     delta_gyro=np.zeros(3),
                     delta_accel=np.zeros(3),
                     error_model=ModifiedPhiModel()):
    """Deterministic linear propagation of INS errors.

    Parameters
    ----------
    dt : float
        Time step per stamp.
    trajectory : DataFrame
        Trajectory.
    delta_position_n : array_like, shape (3,)
        Initial position errors in meters resolved in NED.
    delta_velocity_n : array_like, shape (3,)
        Initial velocity errors resolved in NED.
    delta_rph : array_like, shape (3,)
        Initial heading, pitch and roll errors.
    delta_gyro, delta_accel : float or array_like
        Gyro and accelerometer errors (in SI units). Can be constant or
        specified for each time stamp in `trajectory`.
    error_model : InsErrorModel
        Error model object to use for the propagation.

    Returns
    -------
    traj_err : DataFrame
        Trajectory errors.
    """
    Fi, Fig, Fia = error_model.system_matrix(trajectory)
    Phi = 0.5 * (Fi[1:] + Fi[:-1]) * dt
    Phi[:] += np.identity(Phi.shape[-1])

    delta_gyro = util.mv_prod(Fig, delta_gyro)
    delta_accel = util.mv_prod(Fia, delta_accel)
    delta_sensor = 0.5 * (delta_gyro[1:] + delta_gyro[:-1] +
                          delta_accel[1:] + delta_accel[:-1])

    T = error_model.transform_to_output(trajectory)
    x0 = np.hstack([delta_position_n, delta_velocity_n, np.deg2rad(delta_rph)])
    x0 = np.linalg.inv(T[0]).dot(x0)

    n_samples = Fi.shape[0]
    x = np.empty((n_samples, error_model.N_STATES))
    x[0] = x0
    for i in range(n_samples - 1):
        x[i + 1] = Phi[i].dot(x[i]) + delta_sensor[i] * dt

    state = pd.DataFrame(data=x, index=trajectory.index,
                         columns=list(error_model.STATES.keys()))

    x_out = util.mv_prod(T, x)
    x_out[:, error_model.DRPH] = np.rad2deg(x_out[:, error_model.DRPH])
    error_out = pd.DataFrame(data=x_out,
                             index=trajectory.index,
                             columns=['north', 'east', 'down',
                                      'VN', 'VE', 'VD',
                                      'roll', 'pitch', 'heading'])

    return error_out, state

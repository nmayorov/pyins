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

    Parameters
    ----------
    with_altitude : bool
        Whether to compute altitude and vertical velocity erros.
        If False, vertical channel components (velocity and position errors)
        will be excluded from the state vector.
    """
    def __init__(self, with_altitude):
        self.with_altitude=with_altitude
        if with_altitude:
            self.n_states = 9
            self.dr_range = [0, 1, 2]
            self.dv_range = [0, 1, 2]

            self.drn = 0
            self.dre = 1
            self.drd = 2
            self.dvn = 3
            self.dve = 4
            self.dvd = 5
            self.droll = 6
            self.dpitch = 7
            self.dheading = 8

            self.dr_out = [self.drn, self.dre, self.drd]
            self.dv_out = [self.dvn, self.dve, self.dvd]
        else:
            self.n_states = 7
            self.dr_range = [0, 1]
            self.dv_range = [0, 1]

            self.drn = 0
            self.dre = 1
            self.dvn = 2
            self.dve = 3
            self.droll = 4
            self.dpitch = 5
            self.dheading = 6

            self.dr_out = [self.drn, self.dre]
            self.dv_out = [self.dvn, self.dve]

        self.drph = [self.droll, self.dpitch, self.dheading]
        self.states = None

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

    def position_error_jacobian(self, trajectory_point):
        """Compute position error Jacobian matrix.

        This is the matrix which linearly relates the position error in
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

    Parameters
    ----------
    with_altitude : bool, optional
        Whether to compute altitude and vertical velocity erros. Default is
        True. If False, vertical channel components
        (velocity and position errors) will be excluded from the state vector.

    References
    ----------
    .. [1] Bruno M. Scherzinger and D.Blake Reid
           "Modified Strapdown Inertial Navigator Error Models".
    """
    def __init__(self, with_altitude=True):
        super().__init__(with_altitude)
        if with_altitude:
            self.dr1 = 0
            self.dr2 = 1
            self.dr3 = 2
            self.dv1 = 3
            self.dv2 = 4
            self.dv3 = 5
            self.phi1 = 6
            self.phi2 = 7
            self.phi3 = 8

            self.dr = [self.dr1, self.dr2, self.dr3]
            self.dv = [self.dv1, self.dv2, self.dv3]

            self.states = OrderedDict(DR1=self.dr1, DR2=self.dr2, DR3=self.dr3,
                                      DV1=self.dv1, DV2=self.dv2, DV3=self.dv3,
                                      PHI1=self.phi1, PHI2=self.phi2,
                                      PHI3=self.phi3)
        else:
            self.dr1 = 0
            self.dr2 = 1
            self.dv1 = 2
            self.dv2 = 3
            self.phi1 = 4
            self.phi2 = 5
            self.phi3 = 6

            self.dr = [self.dr1, self.dr2]
            self.dv = [self.dv1, self.dv2]

            self.states = OrderedDict(DR1=self.dr1, DR2=self.dr2,
                                      DV1=self.dv1, DV2=self.dv2,
                                      PHI1=self.phi1, PHI2=self.phi2,
                                      PHI3=self.phi3)
        self.phi = [self.phi1, self.phi2, self.phi3]

    def system_matrix(self, trajectory):
        if self.with_altitude == False:
            trajectory = trajectory.copy()
            trajectory.VD = 0

        n_samples = trajectory.shape[0]

        V_skew = util.skew_matrix(trajectory[['VN', 'VE', 'VD']])
        R = earth.curvature_matrix(trajectory.lat, trajectory.alt)
        Omega_n = earth.rate_n(trajectory.lat)
        rho_n = util.mv_prod(R, trajectory[['VN', 'VE', 'VD']])
        g_n = earth.gravity_n(trajectory.lat, trajectory.alt)
        Cnb = transform.mat_from_rph(trajectory[['roll', 'pitch', 'heading']])

        F = np.zeros((n_samples, self.n_states, self.n_states))
        samples = np.arange(n_samples)

        F[np.ix_(samples, self.dr, self.dv)] = np.eye(len(self.dr),
                                                      len(self.dv))
        F[np.ix_(samples, self.dr, self.phi)] = V_skew[:, self.dr_range]

        F[np.ix_(samples, self.dv, self.dv)] = -util.skew_matrix(
            2 * Omega_n + rho_n)[np.ix_(samples, self.dv_range, self.dv_range)]
        F[np.ix_(samples, self.dv, self.phi)] = -util.skew_matrix(
            g_n)[:, self.dv_range]
        if self.with_altitude:
            F[:, self.dv3, self.dr3] = (2 * earth.gravity(trajectory.lat,
                                                          trajectory.alt)
                                        / earth.R0)

        F[np.ix_(samples, self.phi, self.dr)] = util.mm_prod(
            util.skew_matrix(Omega_n), R)[:, :, self.dr_range]
        F[np.ix_(samples, self.phi, self.dv)] = R[:, :, self.dv_range]
        F[np.ix_(samples, self.phi, self.phi)] = -util.skew_matrix(
            rho_n + Omega_n) + util.mm_prod(R, V_skew)

        B_gyro = np.zeros((n_samples, self.n_states, 3))
        B_gyro[:, self.dv] = util.mm_prod(
            V_skew, Cnb)[:, self.dv_range]
        B_gyro[:, self.phi] = -Cnb

        B_accel = np.zeros((n_samples, self.n_states, 3))
        B_accel[:, self.dv] = Cnb[:, self.dv_range]

        return F, B_gyro, B_accel

    def transform_to_output(self, trajectory):
        series = isinstance(trajectory, pd.Series)
        if series:
            trajectory = trajectory.to_frame().transpose()
        if self.with_altitude == False:
            trajectory = trajectory.copy()
            trajectory.VD = 0

        T = np.zeros((trajectory.shape[0], self.n_states, self.n_states))
        samples = np.arange(len(trajectory))
        T[np.ix_(samples, self.dr_out, self.dr)] = np.eye(len(self.dr))
        T[np.ix_(samples, self.dv_out, self.dv)] = np.eye(len(self.dv))
        T[np.ix_(samples, self.dv_out, self.phi)] = util.skew_matrix(
            trajectory[['VN', 'VE', 'VD']])[:, self.dv_range]
        T[np.ix_(samples, self.drph, self.phi)] = transform.phi_to_delta_rph(
            trajectory[['roll', 'pitch', 'heading']])

        return T[0] if series else T

    def correct_state(self, trajectory_point, error):
        Ctp = Rotation.from_rotvec(error[self.phi]).as_matrix()
        if self.with_altitude:
            lla = transform.perturb_lla(trajectory_point[['lat', 'lon', 'alt']],
                                        -error[self.dr])
            velocity_n = Ctp @ (trajectory_point[['VN', 'VE', 'VD']]
                                - error[self.dv])
        else:
            trajectory_point = trajectory_point.copy()
            lla = transform.perturb_lla(trajectory_point[['lat', 'lon', 'alt']],
                                        [-error[self.dr1], -error[self.dr2], 0])
            velocity_n = Ctp @ (trajectory_point[['VN', 'VE', 'VD']]
                                - [error[self.dv1], error[self.dv2], 0])
            velocity_n[2] = 0

        rph = transform.mat_to_rph(Ctp @ transform.mat_from_rph(
            trajectory_point[['roll', 'pitch', 'heading']]))

        return pd.Series(data=np.hstack((lla, velocity_n, rph)),
                         index=trajectory_point.index)

    def position_error_jacobian(self, trajectory_point):
        result = np.zeros((3, self.n_states))
        result[:, self.dr] = np.eye(3, len(self.dr))
        return result

    def ned_velocity_error_jacobian(self, trajectory_point):
        result = np.zeros((3, self.n_states))
        result[:, self.dv] = np.eye(3, len(self.dv))
        if self.with_altitude == False:
            trajectory_point = trajectory_point.copy()
            trajectory_point.DV = 0
        result[:, self.phi] = util.skew_matrix(
            trajectory_point[['VN', 'VE', 'VD']])

        return result

    def body_velocity_error_jacobian(self, trajectory_point):
        Cnb = transform.mat_from_rph(
            trajectory_point[['roll', 'pitch', 'heading']])
        result = np.zeros((3, self.n_states))
        result[:, self.dv] = Cnb.transpose()[:, self.dv_range]
        return result


class ModifiedPsiModel(InsErrorModel):
    """Error model with psi-angle error and modified velocity errors.

    The psi-angle is used to describe attitude errors and velocity error is
    measured relative to the true velocity resolved in the platform frame.
    The latter trick eliminates specific force from the equations which makes
    the implementation much more convenient. See [1]_ for a detailed discussion
    and derivation.

    Parameters
    ----------
    with_altitude : bool, optional
        Whether to compute altitude and vertical velocity erros. Default is
        True. If False, vertical channel components
        (velocity and position errors) will be excluded from the state vector.

    References
    ----------
    .. [1] Bruno M. Scherzinger and D.Blake Reid
           "Modified Strapdown Inertial Navigator Error Models".
    """
    def __init__(self, with_altitude=True):
        super().__init__(with_altitude)
        if with_altitude:
            self.dr1 = 0
            self.dr2 = 1
            self.dr3 = 2
            self.dv1 = 3
            self.dv2 = 4
            self.dv3 = 5
            self.psi1 = 6
            self.psi2 = 7
            self.psi3 = 8

            self.dr = [self.dr1, self.dr2, self.dr3]
            self.dv = [self.dv1, self.dv2, self.dv3]

            self.states = OrderedDict(DR1=self.dr1, DR2=self.dr2, DR3=self.dr3,
                                      DV1=self.dv1, DV2=self.dv2, DV3=self.dv3,
                                      PSI1=self.psi1, PSI2=self.psi2,
                                      PSI3=self.psi3)
        else:
            self.dr1 = 0
            self.dr2 = 1
            self.dv1 = 2
            self.dv2 = 3
            self.psi1 = 4
            self.psi2 = 5
            self.psi3 = 6

            self.dr = [self.dr1, self.dr2]
            self.dv = [self.dv1, self.dv2]

            self.states = OrderedDict(DR1=self.dr1, DR2=self.dr2,
                                      DV1=self.dv1, DV2=self.dv2,
                                      PSI1=self.psi1, PSI2=self.psi2,
                                      PSI3=self.psi3)
        self.psi = [self.psi1, self.psi2, self.psi3]

    def system_matrix(self, trajectory):
        if self.with_altitude == False:
            trajectory = trajectory.copy()
            trajectory.VD = 0

        n_samples = trajectory.shape[0]

        V_skew = util.skew_matrix(trajectory[['VN', 'VE', 'VD']])
        R = earth.curvature_matrix(trajectory.lat, trajectory.alt)
        Omega_n = earth.rate_n(trajectory.lat)
        rho_n = util.mv_prod(R, trajectory[['VN', 'VE', 'VD']])
        g_n_skew = util.skew_matrix(earth.gravity_n(trajectory.lat,
                                                    trajectory.alt))
        Cnb = transform.mat_from_rph(trajectory[['roll', 'pitch', 'heading']])

        F = np.zeros((n_samples, self.n_states, self.n_states))
        samples = np.arange(n_samples)

        F[np.ix_(samples, self.dr, self.dv)] = np.eye(len(self.dr),
                                                      len(self.dv))
        F[np.ix_(samples, self.dr, self.psi)] = V_skew[:, self.dr_range]

        F[np.ix_(samples, self.dv, self.dv)] = -util.skew_matrix(
            2 * Omega_n + rho_n)[np.ix_(samples, self.dv_range, self.dv_range)]
        F[np.ix_(samples, self.dv, self.dr)] = \
            (-g_n_skew @ R)[np.ix_(samples, self.dv_range, self.dr_range)]
        F[np.ix_(samples, self.dv, self.psi)] = -g_n_skew[:, self.dv_range]
        if self.with_altitude:
            F[:, self.dv3, self.dr3] += (2 * earth.gravity(trajectory.lat,
                                                           trajectory.alt)
                                         / earth.R0)

        F[np.ix_(samples, self.psi, self.psi)] = -util.skew_matrix(rho_n +
                                                                   Omega_n)
        B_gyro = np.zeros((n_samples, self.n_states, 3))
        B_gyro[:, self.dv] = util.mm_prod(V_skew, Cnb)[:, self.dv_range]
        B_gyro[:, self.psi] = -Cnb

        B_accel = np.zeros((n_samples, self.n_states, 3))
        B_accel[:, self.dv] = Cnb[:, self.dv_range]

        return F, B_gyro, B_accel

    def transform_to_output(self, trajectory):
        if self.with_altitude == False:
            trajectory = trajectory.copy()
            trajectory.VD = 0
        series = isinstance(trajectory, pd.Series)
        if series:
            trajectory = trajectory.to_frame().transpose()

        T = np.zeros((trajectory.shape[0], self.n_states, self.n_states))
        samples = np.arange(len(trajectory))
        T[np.ix_(samples, self.dr_out, self.dr)] = np.eye(len(self.dr))
        T[np.ix_(samples, self.dv_out, self.dv)] = np.eye(len(self.dv))
        T[np.ix_(samples, self.dv_out, self.psi)] = util.skew_matrix(
            trajectory[['VN', 'VE', 'VD']])[:, self.dv_range]

        T_rph_phi = transform.phi_to_delta_rph(
            trajectory[['roll', 'pitch', 'heading']])
        R = earth.curvature_matrix(trajectory.lat, trajectory.alt)
        T[np.ix_(samples, self.drph, self.psi)] = T_rph_phi
        T[np.ix_(samples, self.drph, self.dr)] = util.mm_prod(
            T_rph_phi, R)[:, :, self.dr_range]

        return T[0] if series else T

    def correct_state(self, trajectory_point, error):
        R = earth.curvature_matrix(trajectory_point.lat, trajectory_point.alt)
        if self.with_altitude:
            Ctp = Rotation.from_rotvec(error[self.psi] +
                                       R @ error[self.dr]).as_matrix()
            lla = transform.perturb_lla(trajectory_point[['lat', 'lon', 'alt']],
                                        -error[self.dr])
            velocity_n = Ctp @ (trajectory_point[['VN', 'VE', 'VD']]
                                - error[self.dv])
        else:
            trajectory_point = trajectory_point.copy()
            trajectory_point.VD = 0
            error_dr = np.asarray([error[self.dr1], error[self.dr2], 0])
            Ctp = Rotation.from_rotvec(error[self.psi] +
                                       R @ error_dr).as_matrix()
            lla = transform.perturb_lla(trajectory_point[['lat', 'lon', 'alt']],
                  [-error[self.dr1], -error[self.dr2], 0])
            velocity_n = Ctp @ (trajectory_point[['VN', 'VE', 'VD']]
                                - [error[self.dv1], error[self.dv2], 0])
            velocity_n[2] = 0

        rph = transform.mat_to_rph(Ctp @ transform.mat_from_rph(
            trajectory_point[['roll', 'pitch', 'heading']]))

        return pd.Series(data=np.hstack((lla, velocity_n, rph)),
                         index=trajectory_point.index)

    def position_error_jacobian(self, trajectory_point):
        result = np.zeros((3, self.n_states))
        result[:, self.dr] = np.eye(3, len(self.dr))
        return result

    def ned_velocity_error_jacobian(self, trajectory_point):
        if self.with_altitude == False:
            trajectory_point = trajectory_point.copy()
            trajectory_point.VD = 0
        result = np.zeros((3, self.n_states))
        result[:, self.dv] = np.eye(3, len(self.dv))
        result[:, self.psi] = util.skew_matrix(
            trajectory_point[['VN', 'VE', 'VD']])
        return result

    def body_velocity_error_jacobian(self, trajectory_point):
        Cnb = transform.mat_from_rph(
            trajectory_point[['roll', 'pitch', 'heading']])
        result = np.zeros((3, self.n_states))
        result[:, self.dv] = Cnb.transpose()[:, self.dv_range]
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
    if error_model.with_altitude == False:
        delta_position_n = delta_position_n[:2]
        delta_velocity_n = delta_velocity_n[:2]
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
    x = np.empty((n_samples, error_model.n_states))
    x[0] = x0
    for i in range(n_samples - 1):
        x[i + 1] = Phi[i].dot(x[i]) + delta_sensor[i] * dt

    state = pd.DataFrame(data=x, index=trajectory.index,
                         columns=list(error_model.states.keys()))

    x_out = util.mv_prod(T, x)
    x_out[:, error_model.drph] = np.rad2deg(x_out[:, error_model.drph])
    if error_model.with_altitude:
        columns=['north', 'east', 'down', 'VN', 'VE', 'VD',
                 'roll', 'pitch', 'heading']
    else:
        columns=['north', 'east', 'VN', 'VE', 'roll', 'pitch', 'heading']
    error_out = pd.DataFrame(data=x_out, index=trajectory.index,
                             columns=columns)
    return error_out, state

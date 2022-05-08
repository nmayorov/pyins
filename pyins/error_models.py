"""Navigation error models to use in EKF-like estimation filters."""
from collections import OrderedDict
import numpy as np
import pandas as pd
from . import dcm, earth, util, transform


class ErrorModel:
    """Error model interface."""

    N_OUTPUT_STATES = 9
    DRE = 0
    DRN = 1
    DRU = 2
    DVE = 3
    DVN = 4
    DVU = 5
    DROLL = 6
    DPITCH = 7
    DHEADING = 8

    STATES = None
    N_STATES = None

    def system_matrix(self, trajectory):
        """Compute error ODE system matrix.

        Parameters
        ----------
        trajectory : pd.DataFrame
            Trajectory.

        Returns
        -------
        system_matrix : ndarray, shape (n_points, n_states, n_states)
        """
        raise NotImplementedError

    def transform_to_output(self, trajectory):
        """Compute matrix which transform internal states into output states.

        Parameters
        ----------
        trajectory : pd.DataFrame
            Trajectory.

        Returns
        -------
        transform_matrix : ndarray, shape (n_points, n_states, n_states)
        """
        raise NotImplementedError

    def correct_state(self, trajectory_point, error):
        """Correct navigation state with estimated errors.

        Parameters
        ----------
        trajectory_point : pd.Series
            Point of trajectory.

        error : ndarray
            Estimates errors. First n_states components are assumed to
            contain error states.

        Returns
        -------
        corrected_trajectory_point : pd.Series
        """
        raise NotImplementedError

    def position_error_jacobian(self, trajectory_point):
        """Compute position error Jacobian matrix.

        The position error is assumed to be resolved in ENU frame.

        Parameters
        ----------
        trajectory_point : pd.Series
            Point of trajectory.

        Returns
        -------
        jacobian : ndarray, shape (3, n_states)
        """
        raise NotImplementedError

    def enu_velocity_error_jacobian(self, trajectory_point):
        """Compute ENU velocity error Jacobian matrix.

        Parameters
        ----------
        trajectory_point : pd.Series
            Point of trajectory.

        Returns
        -------
        jacobian : ndarray, shape (3, n_states)
        """

        raise NotImplementedError

    def body_velocity_error_jacobian(self, trajectory_point):
        """Compute body velocity error Jacobian matrix.

        Parameters
        ----------
        trajectory_point : pd.Series
            Point of trajectory.

        Returns
        -------
        jacobian : ndarray, shape (3, n_states)
        """
        raise NotImplementedError


class ModifiedPhiModel(ErrorModel):
    N_STATES = 9
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

        V_skew = dcm.skew_matrix(trajectory[['VE', 'VN', 'VU']])
        R = earth.curvature_matrix(trajectory.lat, trajectory.alt)
        Omega_n = earth.rate_n(trajectory.lat)
        rho_n = util.mv_prod(R, trajectory[['VE', 'VN', 'VU']])
        g_n = earth.gravity_n(trajectory.lat, trajectory.alt)
        Cnb = dcm.from_rph(trajectory[['roll', 'pitch', 'heading']])

        F = np.zeros((n_samples, self.N_STATES, self.N_STATES))
        samples = np.arange(n_samples)

        F[np.ix_(samples, self.DR, self.DV)] = np.eye(3)
        F[np.ix_(samples, self.DR, self.PHI)] = V_skew

        F[np.ix_(samples, self.DV, self.DV)] = -dcm.skew_matrix(
            2 * Omega_n + rho_n)
        F[np.ix_(samples, self.DV, self.PHI)] = -dcm.skew_matrix(g_n)
        F[:, self.DV3, self.DR3] = 2 * earth.gravity(trajectory.lat,
                                                     trajectory.alt) / earth.R0

        F[np.ix_(samples, self.PHI, self.DR)] = util.mm_prod(
            dcm.skew_matrix(Omega_n), R)
        F[np.ix_(samples, self.PHI, self.DV)] = R
        F[np.ix_(samples, self.PHI, self.PHI)] = \
            -dcm.skew_matrix(rho_n + Omega_n) + util.mm_prod(R, V_skew)

        B_gyro = np.zeros((n_samples, self.N_STATES, 3))
        B_gyro[np.ix_(samples, self.DV, [0, 1, 2])] = util.mm_prod(V_skew, Cnb)
        B_gyro[np.ix_(samples, self.PHI, [0, 1, 2])] = -Cnb

        B_accel = np.zeros((n_samples, self.N_STATES, 3))
        B_accel[np.ix_(samples, self.DV, [0, 1, 2])] = Cnb

        return F, B_gyro, B_accel

    def transform_to_output(self, trajectory):
        heading = np.deg2rad(trajectory.heading)
        pitch = np.deg2rad(trajectory.pitch)

        sh, ch = np.sin(heading), np.cos(heading)
        cp, tp = np.cos(pitch), np.tan(pitch)

        T = np.zeros((trajectory.shape[0], self.N_STATES, self.N_STATES))
        samples = np.arange(len(trajectory))
        T[np.ix_(samples, self.DR, self.DR)] = np.eye(3)
        T[np.ix_(samples, self.DV, self.DV)] = np.eye(3)
        T[np.ix_(samples, self.DV, self.PHI)] = dcm.skew_matrix(
            trajectory[['VE', 'VN', 'VU']])

        T[:, self.DHEADING, self.PHI1] = -sh * tp
        T[:, self.DHEADING, self.PHI2] = -ch * tp
        T[:, self.DHEADING, self.PHI3] = 1
        T[:, self.DPITCH, self.PHI1] = -ch
        T[:, self.DPITCH, self.PHI2] = sh
        T[:, self.DROLL, self.PHI1] = -sh / cp
        T[:, self.DROLL, self.PHI2] = -ch / cp

        return T

    def correct_state(self, trajectory_point, error):
        Ctp = dcm.from_rv(error[self.PHI])
        lla = transform.perturb_lla(trajectory_point[['lat', 'lon', 'alt']],
                                    -error[self.DR])
        velocity_n = Ctp @ (trajectory_point[['VE', 'VN', 'VU']]
                            - error[self.DV])
        rph = dcm.to_rph(
            Ctp @ dcm.from_rph(trajectory_point[['roll', 'pitch', 'heading']]))

        return pd.Series(data=np.hstack((lla, velocity_n, rph)),
                         index=trajectory_point.index)

    def position_error_jacobian(self, trajectory_point):
        result = np.zeros((3, self.N_STATES))
        result[:, self.DR] = np.eye(3)
        return result

    def enu_velocity_error_jacobian(self, trajectory_point):
        result = np.zeros((3, self.N_STATES))
        result[:, self.DV] = np.eye(3)
        result[:, self.PHI] = dcm.skew_matrix(
            trajectory_point[['VE', 'VN', 'VU']])
        return result

    def body_velocity_error_jacobian(self, trajectory_point):
        Cnb = dcm.from_rph(trajectory_point[['roll', 'pitch', 'heading']])
        result = np.zeros((3, self.N_STATES))
        result[:, self.DV] = Cnb.transpose()
        return result


def propagate_errors(dt, traj,
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
    traj : DataFrame
        Trajectory.
    delta_position_n : array_like, shape (3,)
        Initial position errors in meters resolved in ENU.
    delta_velocity_n : array_like, shape (3,)
        Initial velocity errors resolved in ENU.
    delta_rph : array_like, shape (3,)
        Initial heading, pitch and roll errors.
    delta_gyro, delta_accel : float or array_like
        Gyro and accelerometer errors (in SI units). Can be constant or
        specified for each time stamp in `traj`.
    error_model : ErrorModel
        Error model object to use for the propagation.

    Returns
    -------
    traj_err : DataFrame
        Trajectory errors.
    """
    Fi, Fig, Fia = error_model.system_matrix(traj)
    Phi = 0.5 * (Fi[1:] + Fi[:-1]) * dt
    Phi[:] += np.identity(Phi.shape[-1])

    delta_gyro = util.mv_prod(Fig, delta_gyro)
    delta_accel = util.mv_prod(Fia, delta_accel)
    delta_sensor = 0.5 * (delta_gyro[1:] + delta_gyro[:-1] +
                          delta_accel[1:] + delta_accel[:-1])

    T = error_model.transform_to_output(traj)
    x0 = np.hstack([delta_position_n, delta_velocity_n, np.deg2rad(delta_rph)])
    x0 = np.linalg.inv(T[0]).dot(x0)

    n_samples = Fi.shape[0]
    x = np.empty((n_samples, error_model.N_STATES))
    x[0] = x0
    for i in range(n_samples - 1):
        x[i + 1] = Phi[i].dot(x[i]) + delta_sensor[i] * dt

    x = util.mv_prod(T, x)
    error = pd.DataFrame(index=traj.index)
    error['east'] = x[:, error_model.DRE]
    error['north'] = x[:, error_model.DRN]
    error['up'] = x[:, error_model.DRU]
    error['VE'] = x[:, error_model.DVE]
    error['VN'] = x[:, error_model.DVN]
    error['VU'] = x[:, error_model.DVU]
    error['roll'] = np.rad2deg(x[:, error_model.DROLL])
    error['pitch'] = np.rad2deg(x[:, error_model.DPITCH])
    error['heading'] = np.rad2deg(x[:, error_model.DHEADING])

    return error

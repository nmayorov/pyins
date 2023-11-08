"""Navigation Kalman filters."""
from collections import OrderedDict
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from . import error_models, kalman, util, transform, strapdown
from .imu_model import InertialSensor
from .util import (LLA_COLS, VEL_COLS, RPH_COLS, THETA_COLS, DV_COLS,
                   TRAJECTORY_ERROR_COLS, XYZ_TO_INDEX)


FIRST_ORDER_TIMESTEP_MAX = 0.1


class Observation:
    """Base class for observation models.

    Documentation is given to explain how you can implement a new observation
    model. All you need to do is to implement `compute_obs` function. See Also
    section contains links to already implemented models.

    Parameters
    ----------
    data : DataFrame
        Observed values as a DataFrame. Index must contain time stamps.

    Attributes
    ----------
    data : DataFrame
        Data saved from the constructor.

    See Also
    --------
    LatLonObs
    VeVnObs
    """
    def __init__(self, data):
        self.data = data

    def compute_obs(self, stamp, trajectory_point, error_model):
        """Compute ingredients for a single linearized observation.

        It must compute the observation model (z, H, R) at a given time stamp.
        If the observation is not available at a given `stamp`, it must return
        None.

        Parameters
        ----------
        stamp : int
            Time stamp.
        trajectory_point : Series
            Point of INS trajectory at `stamp`.
        error_model : InsErrorModel
            Error model object.

        Returns
        -------
        z : ndarray, shape (n_obs,)
            Observation vector. A difference between an INS corresponding
            value and an observed value.
        H : ndarray, shape (n_obs, 7)
            Observation model matrix. It relates the vector `z` to the
            INS error states.
        R : ndarray, shape (n_obs, n_obs)
            Covariance matrix of the observation error.
        """
        raise NotImplementedError


class PositionObs(Observation):
    """Observation of latitude and longitude (from GPS or any other source).

    Parameters
    ----------
    data : DataFrame
        Must contain columns 'lat', 'lon' and `alt` columns for latitude,
        longitude and altitude. Index must contain time stamps.
    sd : float
        Measurement accuracy in meters.
    imu_to_antenna_b : array_like, shape (3,) or None, optional
        Vector from IMU to antenna (measurement point) expressed in body
        frame. If None, assumed to be zero.

    Attributes
    ----------
    data : DataFrame
        Data saved from the constructor.
    """
    def __init__(self, data, sd, imu_to_antenna_b=None):
        super(PositionObs, self).__init__(data)
        self.R = sd**2 * np.eye(3)
        self.imu_to_antenna_b = imu_to_antenna_b

    def compute_obs(self, stamp, trajectory_point, error_model):
        if stamp not in self.data.index:
            return None

        z = transform.difference_lla(trajectory_point[LLA_COLS],
                                     self.data.loc[stamp, LLA_COLS])
        if self.imu_to_antenna_b:
            mat_nb = transform.mat_from_rph(trajectory_point[RPH_COLS])
            z += mat_nb @ self.imu_to_antenna_b

        H = error_model.position_error_jacobian(trajectory_point,
                                                self.imu_to_antenna_b)

        return z, H, self.R


class NedVelocityObs(Observation):
    """Observation of velocity resolved in NED frame.

    Parameters
    ----------
    data : DataFrame
        Must contain columns 'VE', 'VN' and 'VU' columns.
        Index must contain time stamps.
    sd : float
        Measurement accuracy in m/s.

    Attributes
    ----------
    data : DataFrame
        Data saved from the constructor.
    """
    def __init__(self, data, sd):
        super(NedVelocityObs, self).__init__(data)
        self.R = sd**2 * np.eye(3)

    def compute_obs(self, stamp, trajectory_point, error_model):
        if stamp not in self.data.index:
            return None

        z = trajectory_point[VEL_COLS] - self.data.loc[stamp, VEL_COLS]
        H = error_model.ned_velocity_error_jacobian(trajectory_point)

        return z, H, self.R


class BodyVelocityObs(Observation):
    """Observation of velocity resolved in body frame.

    Parameters
    ----------
    data : DataFrame
        Must contain columns 'VX', 'VY' and 'VZ' columns.
        Index must contain time stamps.
    sd : float
        Measurement accuracy in m/s.

    Attributes
    ----------
    data : DataFrame
        Data saved from the constructor.
    """
    def __init__(self, data, sd):
        super(BodyVelocityObs, self).__init__(data)
        self.R = sd**2 * np.eye(3)

    def compute_obs(self, stamp, trajectory_point, error_model):
        if stamp not in self.data.index:
            return None

        Cnb = transform.mat_from_rph(trajectory_point[RPH_COLS])
        z = (Cnb.transpose() @ trajectory_point[VEL_COLS] -
             self.data.loc[stamp, ['VX', 'VY', 'VZ']])
        H = error_model.body_velocity_error_jacobian(trajectory_point)
        return z, H, self.R


def _refine_stamps(stamps, max_step):
    stamps = np.sort(np.unique(stamps))
    ds = np.diff(stamps)
    ds_new = []
    for d in ds:
        if d > max_step:
            repeat, left = divmod(d, max_step)
            ds_new.append([max_step] * repeat)
            if left > 0:
                ds_new.append(left)
        else:
            ds_new.append(d)
    ds_new = np.hstack(ds_new)
    stamps_new = stamps[0] + np.cumsum(ds_new)
    return np.hstack((stamps[0], stamps_new))


def _compute_output_errors(trajectory, x, P, output_stamps,
                           error_model, gyro_model, accel_model):
    T = error_model.transform_to_output(trajectory.loc[output_stamps])
    y = util.mv_prod(T, x[:, :error_model.N_STATES])
    Py = util.mm_prod(T, P[:, :error_model.N_STATES, :error_model.N_STATES])
    Py = util.mm_prod(Py, T, bt=True)
    sd_y = np.diagonal(Py, axis1=1, axis2=2) ** 0.5

    error = pd.DataFrame(index=output_stamps)
    error['north'] = y[:, error_model.DRN]
    error['east'] = y[:, error_model.DRE]
    error['down'] = y[:, error_model.DRD]
    error['VN'] = y[:, error_model.DVN]
    error['VE'] = y[:, error_model.DVE]
    error['VD'] = y[:, error_model.DVD]
    error['roll'] = np.rad2deg(y[:, error_model.DROLL])
    error['pitch'] = np.rad2deg(y[:, error_model.DPITCH])
    error['heading'] = np.rad2deg(y[:, error_model.DHEADING])

    sd = pd.DataFrame(index=output_stamps)
    sd['north'] = sd_y[:, error_model.DRN]
    sd['east'] = sd_y[:, error_model.DRE]
    sd['down'] = sd_y[:, error_model.DRD]
    sd['VN'] = sd_y[:, error_model.DVN]
    sd['VE'] = sd_y[:, error_model.DVE]
    sd['VD'] = sd_y[:, error_model.DVD]
    sd['roll'] = np.rad2deg(sd_y[:, error_model.DROLL])
    sd['pitch'] = np.rad2deg(sd_y[:, error_model.DPITCH])
    sd['heading'] = np.rad2deg(sd_y[:, error_model.DHEADING])

    gyro_estimates = pd.DataFrame(index=output_stamps)
    gyro_sd = pd.DataFrame(index=output_stamps)
    n = error_model.N_STATES
    for i, name in enumerate(gyro_model.states):
        gyro_estimates[name] = x[:, n + i]
        gyro_sd[name] = P[:, n + i, n + i] ** 0.5

    accel_estimates = pd.DataFrame(index=output_stamps)
    accel_sd = pd.DataFrame(index=output_stamps)
    ng = gyro_model.n_states
    for i, name in enumerate(accel_model.states):
        accel_estimates[name] = x[:, n + ng + i]
        accel_sd[name] = P[:, n + ng + i, n + ng + i] ** 0.5

    return error, sd, gyro_estimates, gyro_sd, accel_estimates, accel_sd


class FeedforwardFilter:
    """INS Kalman filter in a feedforward form.

    Parameters
    ----------
    dt : float
        Time step per stamp.
    traj_ref : DataFrame
        Trajectory which is used to propagate the error model. It should be
        reasonably accurate and must be recorded at each successive time stamp
        without skips.
    pos_sd : float
        Initial position uncertainty in meters.
    vel_sd : float
        Initial velocity uncertainty.
    azimuth_sd : float
        Initial azimuth (heading) uncertainty.
    level_sd : float
        Initial level (pitch and roll) uncertainty.
    gyro_model, accel_model : None or `InertialSensor`, optional
        Error models for gyros and accelerometers. If None (default), an empty
        model will be used.
    gyro, accel : array_like or None, optional
        Gyro and accelerometer readings, required only if a scale factor is
        modeled in `gyro_model` and `accel_model` respectively.

    Attributes
    ----------
    n_states : int
        Number of states.
    n_noises : int
        Number of noise sources.
    states : OrderedDict
        Dictionary mapping state names to their indices.
    """
    def __init__(self, dt, traj_ref, pos_sd, vel_sd, azimuth_sd, level_sd,
                 error_model=error_models.ModifiedPhiModel(),
                 gyro_model=None, accel_model=None, gyro=None, accel=None):
        if gyro_model is None:
            gyro_model = InertialSensor()
        if accel_model is None:
            accel_model = InertialSensor()

        if gyro_model.readings_required and gyro is None:
            raise ValueError(
                "`gyro_model` contains scale factor or misalignment errors, "
                "thus you must provide `gyro`.")
        if accel_model.readings_required and accel is None:
            raise ValueError(
                "`accel_model` contains scale factor or misalignment errors, "
                "thus you must provide `accel`.")

        self.traj_ref = traj_ref

        n_points = traj_ref.shape[0]
        n_states = error_model.N_STATES + gyro_model.n_states + \
                   accel_model.n_states
        n_noises = (gyro_model.n_noises + accel_model.n_noises +
                    3 * (gyro_model.noise is not None) +
                    3 * (accel_model.noise is not None))

        F = np.zeros((n_points, n_states, n_states))
        G = np.zeros((n_points, n_states, n_noises))
        P0 = np.zeros((n_states, n_states))

        n = error_model.N_STATES
        n1 = gyro_model.n_states
        n2 = accel_model.n_states

        states = error_model.STATES.copy()
        for name, state in gyro_model.states.items():
            states['GYRO_' + name] = n + state
        for name, state in accel_model.states.items():
            states['ACCEL_' + name] = n + n1 + state

        level_sd = np.deg2rad(level_sd)
        azimuth_sd = np.deg2rad(azimuth_sd)

        T = np.linalg.inv(error_model.transform_to_output(traj_ref.iloc[0]))
        P_nav = np.zeros((error_model.N_STATES, error_model.N_STATES))
        P_nav[error_model.DRN, error_model.DRN] = pos_sd ** 2
        P_nav[error_model.DRE, error_model.DRE] = pos_sd ** 2
        P_nav[error_model.DRD, error_model.DRD] = pos_sd ** 2
        P_nav[error_model.DVN, error_model.DVN] = vel_sd ** 2
        P_nav[error_model.DVE, error_model.DVE] = vel_sd ** 2
        P_nav[error_model.DVD, error_model.DVD] = vel_sd ** 2
        P_nav[error_model.DROLL, error_model.DROLL] = level_sd ** 2
        P_nav[error_model.DPITCH, error_model.DPITCH] = level_sd ** 2
        P_nav[error_model.DHEADING, error_model.DHEADING] = azimuth_sd ** 2

        P0[:n, :n] = T @ P_nav @ T.transpose()
        P0[n: n + n1, n: n + n1] = gyro_model.P
        P0[n + n1: n + n1 + n2, n + n1: n + n1 + n2] = accel_model.P

        self.P0 = P0

        Fi, Fig, Fia = error_model.system_matrix(traj_ref)
        F[:, :n, :n] = Fi
        F[:, n: n + n1, n: n + n1] = gyro_model.F
        F[:, n + n1:n + n1 + n2, n + n1: n + n1 + n2] = accel_model.F

        if gyro is not None:
            gyro = np.asarray(gyro)
            gyro = gyro / dt
            gyro = np.vstack((gyro, 2 * gyro[-1] - gyro[-2]))

        if accel is not None:
            accel = np.asarray(accel)
            accel = accel / dt
            accel = np.vstack((accel, 2 * accel[-1] - accel[-2]))

        H_gyro = gyro_model.output_matrix(gyro)
        H_accel = accel_model.output_matrix(accel)
        F[:, :n, n: n + n1] = util.mm_prod(Fig, H_gyro)
        F[:, :n, n + n1: n + n1 + n2] = util.mm_prod(Fia, H_accel)

        G[:, :n, :gyro_model.n_output_noises] = util.mm_prod(Fig, gyro_model.J)
        s = gyro_model.n_output_noises
        G[:, :n, s : s + accel_model.n_output_noises] = util.mm_prod(Fia, accel_model.J)
        s += accel_model.n_output_noises

        s1 = gyro_model.n_noises
        s2 = accel_model.n_noises

        G[:, n: n + n1, s: s + s1] = gyro_model.G
        G[:, n + n1: n + n1 + n2, s + s1: s + s1 + s2] = accel_model.G

        self.F = F
        self.q = np.hstack((gyro_model.v, accel_model.v, gyro_model.q, accel_model.q))
        self.G = G

        self.dt = dt
        self.n_points = n_points
        self.n_states = n_states
        self.n_noises = n_noises
        self.states = states

        self.error_model = error_model
        self.gyro_model = gyro_model
        self.accel_model = accel_model

    def _validate_parameters(self, trajectory, observations, max_step,
                             record_stamps):
        if trajectory is None:
            trajectory = self.traj_ref

        if not np.all(trajectory.index == self.traj_ref.index):
            raise ValueError("Time stamps of reference and computed "
                             "trajectories don't match.")

        if observations is None:
            observations = []

        stamps = pd.Index([])
        for obs in observations:
            stamps = stamps.union(obs.data.index)

        start, end = trajectory.index[0], trajectory.index[-1]
        stamps = stamps.union(pd.Index([start, end]))

        if record_stamps is not None:
            end = min(end, record_stamps[-1])
            record_stamps = record_stamps[(record_stamps >= start) &
                                          (record_stamps <= end)]
            stamps = stamps.union(pd.Index(record_stamps))

        stamps = stamps[(stamps >= start) & (stamps <= end)]

        max_step = max(1, int(np.floor(max_step / self.dt)))
        stamps = _refine_stamps(stamps, max_step)

        if record_stamps is None:
            record_stamps = stamps

        return trajectory, observations, stamps, record_stamps

    def _forward_pass(self, trajectory, observations, stamps, record_stamps):
        inds = stamps - stamps[0]

        n_stamps = record_stamps.shape[0]
        x = np.empty((n_stamps, self.n_states))
        P = np.empty((n_stamps, self.n_states, self.n_states))

        xc = np.zeros(self.n_states)
        Pc = self.P0.copy()
        i_save = 0

        H_max = np.zeros((10, self.n_states))

        obs_stamps = [[] for _ in range(len(observations))]
        obs_residuals = [[] for _ in range(len(observations))]

        for i in range(stamps.shape[0] - 1):
            stamp = stamps[i]
            ind = inds[i]
            next_ind = inds[i + 1]

            for i_obs, obs in enumerate(observations):
                ret = obs.compute_obs(stamp, trajectory.loc[stamp],
                                      self.error_model)
                if ret is not None:
                    z, H, R = ret
                    H_max[:H.shape[0], :self.error_model.N_STATES] = H
                    res = kalman.correct(xc, Pc, z, H_max[:H.shape[0]], R)
                    obs_stamps[i_obs].append(stamp)
                    obs_residuals[i_obs].append(res)

            if record_stamps[i_save] == stamp:
                x[i_save] = xc
                P[i_save] = Pc
                i_save += 1

            F = 0.5 * (self.F[ind] + self.F[next_ind])
            Q = 0.5 * (self.G[ind] + self.G[next_ind]) * self.q
            Q = np.dot(Q, Q.T)

            dt = self.dt * (next_ind - ind)
            Phi, Qd = kalman.compute_process_matrices(F, Q, dt,
                'expm' if dt > FIRST_ORDER_TIMESTEP_MAX else 'first-order')

            xc = Phi.dot(xc)
            Pc = Phi.dot(Pc).dot(Phi.T) + Qd

        x[-1] = xc
        P[-1] = Pc

        residuals = []
        for s, r in zip(obs_stamps, obs_residuals):
            residuals.append(pd.DataFrame(index=s, data=np.asarray(r)))

        return x, P, residuals

    def run(self, trajectory=None, observations=[], max_step=1,
            record_stamps=None):
        """Run the filter.

        Parameters
        ----------
        trajectory : DataFrame or None
            Trajectory computed by INS of which to estimate the errors.
            If None (default), use `traj_ref` from the constructor.
        observations : list of `Observation`
            Observations which will be processed. Empty by default.
        max_step : float, optional
            Maximum allowed time step in seconds for errors propagation.
            Default is 1 second. Set to 0 if you desire the smallest possible
            step.
        record_stamps : array_like or None
            Stamps at which record estimated errors. If None (default), errors
            will be saved at each stamp used internally in the filter.

        Returns
        -------
        Bunch object with the fields listed below. Note that all data frames
        contain stamps only presented in `record_stamps`.
        trajectory : DataFrame
            Trajectory corrected by estimated errors.
        error, sd : DataFrame
            Estimated trajectory errors and their standard deviations.
        gyro_estimates, gyro_sd : DataFrame
            Estimated gyro error states and their standard deviations.
        accel_estimates, accel_sd : DataFrame
            Estimated accelerometer error states and their standard deviations.
        x : ndarray, shape (n_points, n_states)
            History of the filter states.
        P : ndarray, shape (n_points, n_states, n_states)
            History of the filter covariance.
        residuals : list of DataFrame
            Each DataFrame corresponds to an observation from `observations`.
            Its index is observation time stamps and columns contain normalized
            observations residuals for each component of the observation
            vector `z`.
        """
        trajectory, observations, stamps, record_stamps = \
            self._validate_parameters(trajectory, observations, max_step,
                                      record_stamps)

        x, P, residuals = self._forward_pass(trajectory, observations, stamps,
                                             record_stamps)

        error, sd, gyro_estimates, gyro_sd, accel_estimates, accel_sd = \
            _compute_output_errors(self.traj_ref, x, P, record_stamps,
                                   self.error_model, self.gyro_model,
                                   self.accel_model)

        trajectory = transform.correct_trajectory(trajectory, error)

        return util.Bunch(trajectory=trajectory, error=error, sd=sd,
                          gyro_estimates=gyro_estimates, gyro_sd=gyro_sd,
                          accel_estimates=accel_estimates, accel_sd=accel_sd,
                          x=x, P=P, residuals=residuals)


def compute_average_pva(pva_1, pva_2):
    rot_1 = Rotation.from_euler('xyz', pva_1[RPH_COLS], True)
    rot_2 = Rotation.from_euler('xyz', pva_2[RPH_COLS], True)
    rph_average = pd.Series(
        Rotation.concatenate([rot_1, rot_2]).mean().as_euler('xyz', True), RPH_COLS)
    return pd.concat([
        0.5 * (pva_1[LLA_COLS] + pva_1[LLA_COLS]),
        0.5 * (pva_1[VEL_COLS] + pva_2[VEL_COLS]),
        rph_average
    ])


def concatenate_states(error_model, gyro_model, accel_model):
    return (list(error_model.STATES.keys()) +
            ['gyro_' + state for state in gyro_model.states] +
            ['accel_' + state for state in accel_model.states])


def initialize_covariance(pva,
                          pos_sd, vel_sd, level_sd, azimuth_sd,
                          error_model : error_models.InsErrorModel,
                          gyro_model : InertialSensor,
                          accel_model : InertialSensor):
    level_sd = np.deg2rad(level_sd)
    azimuth_sd = np.deg2rad(azimuth_sd)

    P_nav = np.zeros((error_model.N_STATES, error_model.N_STATES))
    P_nav[error_model.DRN, error_model.DRN] = pos_sd ** 2
    P_nav[error_model.DRE, error_model.DRE] = pos_sd ** 2
    P_nav[error_model.DRD, error_model.DRD] = pos_sd ** 2
    P_nav[error_model.DVN, error_model.DVN] = vel_sd ** 2
    P_nav[error_model.DVE, error_model.DVE] = vel_sd ** 2
    P_nav[error_model.DVD, error_model.DVD] = vel_sd ** 2
    P_nav[error_model.DROLL, error_model.DROLL] = level_sd ** 2
    P_nav[error_model.DPITCH, error_model.DPITCH] = level_sd ** 2
    P_nav[error_model.DHEADING, error_model.DHEADING] = azimuth_sd ** 2

    n_states = error_model.N_STATES + gyro_model.n_states + accel_model.n_states
    P = np.zeros((n_states, n_states))
    T = error_model.transform_to_output(pva)

    inertial_block = slice(error_model.N_STATES)
    gyro_block = slice(error_model.N_STATES, error_model.N_STATES + gyro_model.n_states)
    accel_block = slice(error_model.N_STATES + gyro_model.n_states, n_states)

    P[inertial_block, inertial_block] = T @ P_nav @ T.transpose()
    P[gyro_block, gyro_block] = gyro_model.P
    P[accel_block, accel_block] = accel_model.P

    return P


def compute_error_propagation_matrices(pva, gyro, accel, time_delta,
                                       error_model : error_models.InsErrorModel,
                                       gyro_model : InertialSensor,
                                       accel_model : InertialSensor):
    n_states = error_model.N_STATES + gyro_model.n_states + accel_model.n_states
    n_noises = (gyro_model.n_output_noises + gyro_model.n_noises +
                accel_model.n_output_noises + accel_model.n_noises)

    Fii, Fig, Fia = error_model.system_matrix(pva)

    Hg = gyro_model.output_matrix(gyro)
    Ha = accel_model.output_matrix(accel)

    inertial_block = slice(error_model.N_STATES)
    gyro_block = slice(error_model.N_STATES, error_model.N_STATES + gyro_model.n_states)
    accel_block = slice(error_model.N_STATES + gyro_model.n_states, n_states)

    gyro_out_noise_block = slice(gyro_model.n_output_noises)
    accel_out_noise_block = slice(
        gyro_model.n_output_noises,
        gyro_model.n_output_noises + accel_model.n_output_noises)
    gyro_noise_block = slice(
        gyro_model.n_output_noises + accel_model.n_output_noises,
        gyro_model.n_output_noises + accel_model.n_output_noises + gyro_model.n_noises)
    accel_noise_block = slice(
        gyro_model.n_output_noises + accel_model.n_output_noises + gyro_model.n_noises,
        n_noises)

    F = np.zeros((n_states, n_states))
    F[inertial_block, inertial_block] = Fii
    F[inertial_block, gyro_block] = Fig @ Hg
    F[inertial_block, accel_block] = Fia @ Ha
    F[gyro_block, gyro_block] = gyro_model.F
    F[accel_block, accel_block] = accel_model.F

    G = np.zeros((n_states, n_noises))
    G[inertial_block, gyro_out_noise_block] = Fig @ gyro_model.J
    G[inertial_block, accel_out_noise_block] = Fia @ accel_model.J
    G[gyro_block, gyro_noise_block] = gyro_model.G
    G[accel_block, accel_noise_block] = accel_model.G

    q = np.hstack((gyro_model.v, accel_model.v, gyro_model.q, accel_model.q))

    return kalman.compute_process_matrices(F, G @ np.diag(q**2) @ G.transpose(),
                                           time_delta, 'expm')


def compute_sd(P, trajectory, error_model, gyro_model, accel_model):
    inertial_block = slice(error_model.N_STATES)
    gyro_block = slice(error_model.N_STATES, error_model.N_STATES + gyro_model.n_states)
    accel_block = slice(
        error_model.N_STATES + gyro_model.n_states,
        error_model.N_STATES + gyro_model.n_states + accel_model.n_states)

    P_ins = P[:, inertial_block, inertial_block]
    P_gyro = P[:, gyro_block, gyro_block]
    P_accel = P[:, accel_block, accel_block]

    T = error_model.transform_to_output(trajectory)
    P_nav = util.mm_prod(T, P_ins)
    P_nav = util.mm_prod(P_nav, T, bt=True)
    trajectory_sd = np.diagonal(P_nav, axis1=1, axis2=2) ** 0.5
    gyro_sd = np.diagonal(P_gyro, axis1=1, axis2=2) ** 0.5
    accel_sd = np.diagonal(P_accel, axis1=1, axis2=2) ** 0.5

    trajectory_sd = pd.DataFrame(trajectory_sd, index=trajectory.index,
                                 columns=TRAJECTORY_ERROR_COLS)
    trajectory_sd[RPH_COLS] = np.rad2deg(trajectory_sd[RPH_COLS])
    gyro_sd = pd.DataFrame(gyro_sd, index=trajectory.index, columns=gyro_model.states)
    accel_sd = pd.DataFrame(accel_sd, index=trajectory.index,
                            columns=accel_model.states)
    return trajectory_sd, gyro_sd, accel_sd


def compute_feedforward_result(trajectory_nominal, trajectory, x, P,
                               error_model, gyro_model, accel_model):
    inertial_block = slice(error_model.N_STATES)
    gyro_block = slice(error_model.N_STATES, error_model.N_STATES + gyro_model.n_states)
    accel_block = slice(
        error_model.N_STATES + gyro_model.n_states,
        error_model.N_STATES + gyro_model.n_states + accel_model.n_states)

    P_ins = P[:, inertial_block, inertial_block]
    P_gyro = P[:, gyro_block, gyro_block]
    P_accel = P[:, accel_block, accel_block]

    x_ins = x[:, inertial_block]
    x_gyro = x[:, gyro_block]
    x_accel = x[:, accel_block]

    T = error_model.transform_to_output(trajectory_nominal)

    x_nav = util.mv_prod(T, x_ins)
    error_nav = pd.DataFrame(x_nav, index=trajectory.index,
                             columns=TRAJECTORY_ERROR_COLS)
    error_nav[RPH_COLS] *= transform.RAD_TO_DEG

    P_nav = util.mm_prod_symmetric(T, P_ins)
    trajectory = transform.correct_trajectory(trajectory, error_nav)
    trajectory_sd = pd.DataFrame(np.diagonal(P_nav, axis1=1, axis2=2) ** 0.5,
                                 index=trajectory.index, columns=TRAJECTORY_ERROR_COLS)
    trajectory_sd[RPH_COLS] *= transform.RAD_TO_DEG

    gyro = pd.DataFrame(x_gyro, index=trajectory.index, columns=gyro_model.states)
    gyro_sd = pd.DataFrame(np.diagonal(P_gyro, axis1=1, axis2=2) ** 0.5,
                           index=trajectory.index, columns=gyro_model.states)

    accel = pd.DataFrame(x_accel, index=trajectory.index,
                         columns=accel_model.states)
    accel_sd = pd.DataFrame(np.diagonal(P_accel, axis1=1, axis2=2) ** 0.5,
                            index=trajectory.index, columns=accel_model.states)

    return trajectory, trajectory_sd, gyro, gyro_sd, accel, accel_sd


def correct_increments(increments, gyro_estimates, accel_estimates):
    result = increments.copy()
    result[THETA_COLS] = gyro_estimates.correct_increments(increments.dt,
                                                           increments[THETA_COLS])
    result[DV_COLS] = accel_estimates.correct_increments(increments.dt,
                                                         increments[DV_COLS])
    return result


def run_feedback_filter(initial_pva,
                        pos_sd, vel_sd, level_sd, azimuth_sd,
                        increments, gyro_model=None, accel_model=None,
                        observations=None, time_step=0.1):
    if gyro_model is None:
        gyro_model = InertialSensor()
    if accel_model is None:
        accel_model = InertialSensor()
    if observations is None:
        observations = []

    observation_times = np.hstack([
        np.asarray(observation.data.index) for observation in observations])
    observation_times = np.sort(np.unique(observation_times))

    start_time = initial_pva.name
    end_time = increments.index[-1]
    observation_times = observation_times[(observation_times >= start_time) &
                                          (observation_times <= end_time)]
    observation_times = np.append(observation_times, np.inf)
    observation_times_index = 0

    error_model = error_models.ModifiedPhiModel()
    P = initialize_covariance(initial_pva, pos_sd, vel_sd, level_sd, azimuth_sd,
                              error_model, gyro_model, accel_model)
    states = concatenate_states(error_model, gyro_model, accel_model)

    inertial_block = slice(error_model.N_STATES)
    gyro_block = slice(error_model.N_STATES, error_model.N_STATES + gyro_model.n_states)
    accel_block = slice(error_model.N_STATES + gyro_model.n_states, None)

    n_states = len(states)
    trajectory_result = []
    gyro_result = []
    accel_result = []
    P_all = []

    integrator = strapdown.Integrator(initial_pva, True)
    gyro_model.reset_estimates()
    accel_model.reset_estimates()

    innovations = {}
    innovations_times = {}
    for observation in observations:
        name = observation.__class__.__name__
        innovations[name] = []
        innovations_times[name] = []

    while integrator.get_time() < end_time:
        time = integrator.get_time()
        if observation_times[observation_times_index] == time:
            x = np.zeros(n_states)
            ins_state = integrator.get_state()
            for observation in observations:
                ret = observation.compute_obs(time, ins_state, error_model)
                if ret is not None:
                    z, H, R = ret
                    H_full = np.zeros((len(z), n_states))
                    H_full[:, inertial_block] = H
                    innovation = kalman.correct(x, P, z, H_full, R)
                    name = observation.__class__.__name__
                    innovations[name].append(innovation)
                    innovations_times[name].append(time)

            integrator.set_state(
                error_model.correct_state(ins_state, x[inertial_block]))
            gyro_model.update_estimates(x[gyro_block])
            accel_model.update_estimates(x[accel_block])
            observation_times_index += 1

        trajectory_result.append(integrator.get_state())
        gyro_result.append(gyro_model.get_estimates())
        accel_result.append(accel_model.get_estimates())
        P_all.append(P)

        next_time = min(time + time_step,
                        observation_times[observation_times_index])
        time_delta = next_time - time
        increments_batch = correct_increments(
            increments.loc[np.nextafter(time, next_time) : next_time],
            gyro_model, accel_model)

        pva_old = integrator.get_state()
        integrator.integrate(increments_batch)
        pva_new = integrator.get_state()
        pva_average = compute_average_pva(pva_old, pva_new)
        gyro_average = increments_batch[THETA_COLS].sum(axis=0) / time_delta
        accel_average = increments_batch[DV_COLS].sum(axis=0) / time_delta

        Phi, Qd = compute_error_propagation_matrices(pva_average, gyro_average,
                                                     accel_average, time_delta,
                                                     error_model, gyro_model,
                                                     accel_model)
        P = Phi @ P @ Phi.transpose() + Qd

    P_all = np.asarray(P_all)
    trajectory_result = pd.DataFrame(trajectory_result)
    trajectory_sd, gyro_sd, accel_sd = compute_sd(
        P_all, trajectory_result, error_model, gyro_model, accel_model)

    for observation in observations:
        name = observation.__class__.__name__
        innovations[name] = pd.DataFrame(innovations[name], innovations_times[name],
                                         columns=observation.data.columns)

    return util.Bunch(
        trajectory=trajectory_result,
        trajectory_sd=trajectory_sd,
        gyro=pd.DataFrame(gyro_result, index=trajectory_result.index),
        gyro_sd=gyro_sd,
        accel=pd.DataFrame(accel_result, index=trajectory_result.index),
        accel_sd=accel_sd,
        innovations=innovations)


def run_feedforward_filter(trajectory_nominal, trajectory, pos_sd, vel_sd, level_sd,
                           azimuth_sd, gyro_model=None, accel_model=None,
                           observations=None, increments=None, time_step=0.1):
    if (trajectory_nominal.index != trajectory.index).any():
        raise ValueError(
            "`trajectory_nominal` and `trajectory` must have the same time index")
    times = trajectory_nominal.index

    if gyro_model is None:
        gyro_model = InertialSensor()
    if accel_model is None:
        accel_model = InertialSensor()
    if observations is None:
        observations = []

    if increments is None and (gyro_model.readings_required or
                               accel_model.readings_required):
        raise ValueError("When scale or misalignments errors are modelled `increments` "
                         "must be provided")

    observation_times = np.hstack([
        np.asarray(observation.data.index) for observation in observations])
    observation_times = np.sort(np.unique(observation_times))

    start_time = times[0]
    end_time = times[-1]
    observation_times = observation_times[(observation_times >= start_time) &
                                          (observation_times <= end_time)]
    observation_times = np.append(observation_times, np.inf)
    observation_times_index = 0

    error_model = error_models.ModifiedPhiModel()
    P = initialize_covariance(trajectory_nominal.iloc[0], pos_sd, vel_sd, level_sd,
                              azimuth_sd, error_model, gyro_model, accel_model)
    x = np.zeros(len(P))
    states = concatenate_states(error_model, gyro_model, accel_model)

    inertial_block = slice(error_model.N_STATES)

    n_states = len(states)
    x_all = []
    P_all = []
    times_all = []

    innovations = {}
    innovations_times = {}
    for observation in observations:
        name = observation.__class__.__name__
        innovations[name] = []
        innovations_times[name] = []

    time = start_time
    while time < end_time:
        pva = trajectory.loc[time]
        if observation_times[observation_times_index] == time:
            for observation in observations:
                ret = observation.compute_obs(time, pva, error_model)
                if ret is not None:
                    z, H, R = ret
                    H_full = np.zeros((len(z), n_states))
                    H_full[:, inertial_block] = H
                    innovation = kalman.correct(x, P, z, H_full, R)
                    name = observation.__class__.__name__
                    innovations[name].append(innovation)
                    innovations_times[name].append(time)

            observation_times_index += 1

        times_all.append(time)
        x_all.append(x)
        P_all.append(P)

        next_time = min(time + time_step,
                        observation_times[observation_times_index])
        next_time = times[np.searchsorted(times, next_time, side='right') - 1]
        time_delta = next_time - time

        pva_old = trajectory_nominal.loc[time]
        pva_new = trajectory_nominal.loc[next_time]
        pva_average = compute_average_pva(pva_old, pva_new)
        if increments is not None:
            increments_batch = increments.loc[np.nextafter(time, next_time) : next_time]
            gyro_average = increments_batch[THETA_COLS].sum(axis=0) / time_delta
            accel_average = increments_batch[DV_COLS].sum(axis=0) / time_delta
        else:
            gyro_average = None
            accel_average = None

        Phi, Qd = compute_error_propagation_matrices(pva_average, gyro_average,
                                                     accel_average, time_delta,
                                                     error_model, gyro_model,
                                                     accel_model)
        x = Phi @ x
        P = Phi @ P @ Phi.transpose() + Qd
        time = next_time

    times_all = np.asarray(times_all)
    x_all = np.asarray(x_all)
    P_all = np.asarray(P_all)
    trajectory_nominal = trajectory_nominal.loc[times_all]
    trajectory = trajectory.loc[times_all]

    trajectory, trajectory_sd, gyro, gyro_sd, accel, accel_sd = (
        compute_feedforward_result(trajectory_nominal, trajectory, x_all, P_all,
                                   error_model, gyro_model, accel_model))

    for observation in observations:
        name = observation.__class__.__name__
        innovations[name] = pd.DataFrame(innovations[name], innovations_times[name],
                                         columns=observation.data.columns)

    return util.Bunch(
        trajectory=trajectory,
        trajectory_sd=trajectory_sd,
        gyro=gyro,
        gyro_sd=gyro_sd,
        accel=accel,
        accel_sd=accel_sd,
        innovations=innovations)

"""Navigation Kalman filters."""
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from . import earth, error_models, kalman, util, transform, strapdown
from .imu_model import InertialSensor
from .util import (LLA_COLS, VEL_COLS, RPH_COLS, THETA_COLS, DV_COLS,
                   TRAJECTORY_ERROR_COLS)


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

    rn, _, rp = earth.principal_radii(trajectory_nominal.lat, trajectory_nominal.alt)
    trajectory = trajectory.copy()
    trajectory.lat -= error_nav.north / rn * transform.RAD_TO_DEG
    trajectory.lon -= error_nav.east / rp * transform.RAD_TO_DEG
    trajectory.alt += error_nav.down
    trajectory[VEL_COLS] -= error_nav[VEL_COLS]
    trajectory[RPH_COLS] -= error_nav[RPH_COLS]

    P_nav = util.mm_prod_symmetric(T, P_ins)
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

    inertial_block = slice(error_model.N_STATES)
    gyro_block = slice(error_model.N_STATES, error_model.N_STATES + gyro_model.n_states)
    accel_block = slice(error_model.N_STATES + gyro_model.n_states, None)

    n_states = len(P)
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

    increments_index = 0
    while integrator.get_time() < end_time:
        time = integrator.get_time()
        observation_time = observation_times[observation_times_index]
        increment = increments.iloc[increments_index]
        if observation_time < increment.name:
            x = np.zeros(n_states)
            pva = integrator.predict((observation_time - time) / increment['dt'] *
                                     increment)
            for observation in observations:
                ret = observation.compute_obs(observation_time, pva, error_model)
                if ret is not None:
                    z, H, R = ret
                    H_full = np.zeros((len(z), n_states))
                    H_full[:, inertial_block] = H
                    innovation = kalman.correct(x, P, z, H_full, R)
                    name = observation.__class__.__name__
                    innovations[name].append(innovation)
                    innovations_times[name].append(observation_time)

            integrator.set_state(
                error_model.correct_state(integrator.get_state(), x[inertial_block]))
            gyro_model.update_estimates(x[gyro_block])
            accel_model.update_estimates(x[accel_block])
            observation_times_index += 1

        trajectory_result.append(integrator.get_state())
        gyro_result.append(gyro_model.get_estimates())
        accel_result.append(accel_model.get_estimates())
        P_all.append(P)

        next_time = min(time + time_step,
                        observation_times[observation_times_index])
        next_increment_index = np.searchsorted(increments.index, next_time,
                                               side='right')
        increments_batch = correct_increments(
            increments.iloc[increments_index : next_increment_index],
            gyro_model, accel_model)
        increments_index = next_increment_index

        pva_old = integrator.get_state()
        integrator.integrate(increments_batch)
        pva_new = integrator.get_state()
        time_delta = integrator.get_time() - time
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

    inertial_block = slice(error_model.N_STATES)

    n_states = len(P)
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

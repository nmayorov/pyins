"""Navigation Kalman filters."""
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from . import earth, error_models, imu_model, kalman, util, transform, strapdown
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
        Observed values as a DataFrame indexed by time.

    Attributes
    ----------
    data : DataFrame
        Data saved from the constructor.

    See Also
    --------
    PositionObs
    NedVelocityObs
    BodyVelocityObs
    """
    def __init__(self, data):
        self.data = data

    def compute_obs(self, time, pva, error_model):
        """Compute matrices for a single linearized observation.

        It must compute the observation model (z, H, R) at a given time stamp.
        If the observation is not available at the given `time`, it must return
        None.

        Parameters
        ----------
        time : float
            Time.
        pva : Pva
            Position-velocity-attitude estimates from INS at `time`.
        error_model : InsErrorModel
            Error model object.

        Returns
        -------
        z : ndarray, shape (n_obs,)
            Observation vector. A difference between the value derived from `pva`
            and an observed value.
        H : ndarray, shape (n_obs, 9)
            Observation model matrix. It relates the vector `z` to the INS error states.
        R : ndarray, shape (n_obs, n_obs)
            Covariance matrix of the observation error.
        """
        raise NotImplementedError


class PositionObs(Observation):
    """Observation of latitude, longitude and altitude (from GNSS or any other source).

    Parameters
    ----------
    data : DataFrame
        Must be indexed by time and contain columns 'lat', 'lon' and `alt` columns for
        latitude, longitude and altitude.
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

    def compute_obs(self, time, pva, error_model):
        if time not in self.data.index:
            return None

        z = transform.difference_lla(pva[LLA_COLS], self.data.loc[time, LLA_COLS])
        if self.imu_to_antenna_b is not None:
            mat_nb = transform.mat_from_rph(pva[RPH_COLS])
            z += mat_nb @ self.imu_to_antenna_b

        H = error_model.position_error_jacobian(pva, self.imu_to_antenna_b)

        return z, H, self.R


class NedVelocityObs(Observation):
    """Observation of velocity resolved in NED frame (typically from GNSS).

    Parameters
    ----------
    data : DataFrame
        Must be indexed by time and contain 'VN', 'VE' and 'VD' columns.
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

    def compute_obs(self, time, pva, error_model):
        if time not in self.data.index:
            return None

        z = pva[VEL_COLS] - self.data.loc[time, VEL_COLS]
        H = error_model.ned_velocity_error_jacobian(pva)

        return z, H, self.R


class BodyVelocityObs(Observation):
    """Observation of velocity resolved in body frame.

    Parameters
    ----------
    data : DataFrame
        Must be indexed by `time` and contain columns 'VX', 'VY' and 'VZ'.
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

    def compute_obs(self, time, pva, error_model):
        if time not in self.data.index:
            return None

        mat_nb = transform.mat_from_rph(pva[RPH_COLS])
        z = mat_nb.T @ pva[VEL_COLS] - self.data.loc[time, ['VX', 'VY', 'VZ']]
        H = error_model.body_velocity_error_jacobian(pva)
        return z, H, self.R


def _interpolate_pva(first, second, alpha):
    rot_1 = Rotation.from_euler('xyz', first[RPH_COLS], True)
    rot_2 = Rotation.from_euler('xyz', second[RPH_COLS], True)
    rph_average = pd.Series(Rotation.concatenate([rot_1, rot_2])
                            .mean([1 - alpha, alpha]).as_euler('xyz', True), RPH_COLS)
    return pd.concat([
        (1 - alpha) * first[LLA_COLS] + alpha * second[LLA_COLS],
        (1 - alpha) * first[VEL_COLS] + alpha * second[VEL_COLS],
        rph_average
    ])


def _initialize_covariance(pva, pos_sd, vel_sd, level_sd, azimuth_sd,
                           error_model, gyro_model, accel_model):
    P_pva = np.zeros((error_model.N_STATES, error_model.N_STATES))
    P_pva[error_model.DRN, error_model.DRN] = pos_sd ** 2
    P_pva[error_model.DRE, error_model.DRE] = pos_sd ** 2
    P_pva[error_model.DRD, error_model.DRD] = pos_sd ** 2
    P_pva[error_model.DVN, error_model.DVN] = vel_sd ** 2
    P_pva[error_model.DVE, error_model.DVE] = vel_sd ** 2
    P_pva[error_model.DVD, error_model.DVD] = vel_sd ** 2
    P_pva[error_model.DROLL, error_model.DROLL] = np.deg2rad(level_sd) ** 2
    P_pva[error_model.DPITCH, error_model.DPITCH] = np.deg2rad(level_sd) ** 2
    P_pva[error_model.DHEADING, error_model.DHEADING] = np.deg2rad(azimuth_sd) ** 2

    n_states = error_model.N_STATES + gyro_model.n_states + accel_model.n_states

    ins_block = slice(error_model.N_STATES)
    gyro_block = slice(error_model.N_STATES, error_model.N_STATES + gyro_model.n_states)
    accel_block = slice(error_model.N_STATES + gyro_model.n_states, n_states)

    P = np.zeros((n_states, n_states))
    T = np.linalg.inv(error_model.transform_to_output(pva))

    P[ins_block, ins_block] = T @ P_pva @ T.transpose()
    P[gyro_block, gyro_block] = gyro_model.P
    P[accel_block, accel_block] = accel_model.P

    return P


def _compute_error_propagation_matrices(pva, gyro, accel, time_delta,
                                        error_model, gyro_model, accel_model):
    n_states = error_model.N_STATES + gyro_model.n_states + accel_model.n_states
    n_noises = (gyro_model.n_output_noises + gyro_model.n_noises +
                accel_model.n_output_noises + accel_model.n_noises)

    Fii, Fig, Fia = error_model.system_matrix(pva)

    Hg = gyro_model.output_matrix(gyro)
    Ha = accel_model.output_matrix(accel)

    ins_block = slice(error_model.N_STATES)
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
    F[ins_block, ins_block] = Fii
    F[ins_block, gyro_block] = Fig @ Hg
    F[ins_block, accel_block] = Fia @ Ha
    F[gyro_block, gyro_block] = gyro_model.F
    F[accel_block, accel_block] = accel_model.F

    G = np.zeros((n_states, n_noises))
    G[ins_block, gyro_out_noise_block] = Fig @ gyro_model.J
    G[ins_block, accel_out_noise_block] = Fia @ accel_model.J
    G[gyro_block, gyro_noise_block] = gyro_model.G
    G[accel_block, accel_noise_block] = accel_model.G

    q = np.hstack((gyro_model.v, accel_model.v, gyro_model.q, accel_model.q))

    return kalman.compute_process_matrices(F, G @ np.diag(q**2) @ G.transpose(),
                                           time_delta, 'expm')


def _compute_sd(P, trajectory, error_model, gyro_model, accel_model):
    ins_block = slice(error_model.N_STATES)
    gyro_block = slice(error_model.N_STATES, error_model.N_STATES + gyro_model.n_states)
    accel_block = slice(
        error_model.N_STATES + gyro_model.n_states,
        error_model.N_STATES + gyro_model.n_states + accel_model.n_states)

    P_ins = P[:, ins_block, ins_block]
    P_gyro = P[:, gyro_block, gyro_block]
    P_accel = P[:, accel_block, accel_block]

    T = error_model.transform_to_output(trajectory)
    trajectory_sd = np.diagonal(
        util.mm_prod_symmetric(T, P_ins), axis1=1, axis2=2) ** 0.5
    gyro_sd = np.diagonal(P_gyro, axis1=1, axis2=2) ** 0.5
    accel_sd = np.diagonal(P_accel, axis1=1, axis2=2) ** 0.5

    trajectory_sd = pd.DataFrame(trajectory_sd, index=trajectory.index,
                                 columns=TRAJECTORY_ERROR_COLS)
    trajectory_sd[RPH_COLS] *= transform.RAD_TO_DEG
    gyro_sd = pd.DataFrame(gyro_sd, index=trajectory.index, columns=gyro_model.states)
    accel_sd = pd.DataFrame(accel_sd, index=trajectory.index,
                            columns=accel_model.states)
    return trajectory_sd, gyro_sd, accel_sd


def _compute_feedforward_result(x, P, trajectory_nominal, trajectory,
                                error_model, gyro_model, accel_model):
    ins_block = slice(error_model.N_STATES)
    gyro_block = slice(error_model.N_STATES, error_model.N_STATES + gyro_model.n_states)
    accel_block = slice(
        error_model.N_STATES + gyro_model.n_states,
        error_model.N_STATES + gyro_model.n_states + accel_model.n_states)

    P_ins = P[:, ins_block, ins_block]
    P_gyro = P[:, gyro_block, gyro_block]
    P_accel = P[:, accel_block, accel_block]

    x_ins = x[:, ins_block]
    x_gyro = x[:, gyro_block]
    x_accel = x[:, accel_block]

    T = error_model.transform_to_output(trajectory_nominal)
    error_nav = pd.DataFrame(util.mv_prod(T, x_ins), index=trajectory.index,
                             columns=TRAJECTORY_ERROR_COLS)
    error_nav[RPH_COLS] *= transform.RAD_TO_DEG

    rn, _, rp = earth.principal_radii(trajectory_nominal.lat, trajectory_nominal.alt)
    trajectory = trajectory.copy()
    trajectory.lat -= error_nav.north / rn * transform.RAD_TO_DEG
    trajectory.lon -= error_nav.east / rp * transform.RAD_TO_DEG
    trajectory.alt += error_nav.down
    trajectory[VEL_COLS] -= error_nav[VEL_COLS]
    trajectory[RPH_COLS] -= error_nav[RPH_COLS]

    trajectory_sd = pd.DataFrame(
        np.diagonal(util.mm_prod_symmetric(T, P_ins), axis1=1, axis2=2) ** 0.5,
        index=trajectory.index, columns=TRAJECTORY_ERROR_COLS)
    trajectory_sd[RPH_COLS] *= transform.RAD_TO_DEG

    gyro = pd.DataFrame(x_gyro, index=trajectory.index, columns=gyro_model.states)
    gyro_sd = pd.DataFrame(np.diagonal(P_gyro, axis1=1, axis2=2) ** 0.5,
                           index=trajectory.index, columns=gyro_model.states)

    accel = pd.DataFrame(x_accel, index=trajectory.index, columns=accel_model.states)
    accel_sd = pd.DataFrame(np.diagonal(P_accel, axis1=1, axis2=2) ** 0.5,
                            index=trajectory.index, columns=accel_model.states)

    return trajectory, trajectory_sd, gyro, gyro_sd, accel, accel_sd


def _correct_increments(increments, gyro_model, accel_model):
    result = increments.copy()
    result[THETA_COLS] = gyro_model.correct_increments(increments.dt,
                                                       increments[THETA_COLS])
    result[DV_COLS] = accel_model.correct_increments(increments.dt,
                                                     increments[DV_COLS])
    return result


def run_feedback_filter(initial_pva, position_sd, velocity_sd, level_sd, azimuth_sd,
                        increments, gyro_model=None, accel_model=None,
                        observations=None, time_step=0.1):
    """Run INS filter with feedback corrections.

    Also known as Extended Kalman Filter.

    Parameters
    ----------
    initial_pva : Pva
        Initial position-velocity-attitude.
    position_sd : float
        Initial assumed position standard deviation in meters.
    velocity_sd : float
        Initial assumed velocity standard deviation in m/s.
    level_sd : float
        Initial assumed roll and pitch standard deviation in degrees.
    azimuth_sd : float
        Initial assumed heading standard deviation in degrees.
    increments : Increments
        IMU increments.
    gyro_model : InertiaSensor, optional
        Sensor model for gyros. If None (default), a default model will be used.
    accel_model : InertialSensor, optional
        Sensor model for accelerometers.
        If None (default), a default model will be used.
    observations : list of Observation. optional
        Observations, empty by default.
    time_step : float, optional
        Time step for error state propagation.
        The value typically should not exceed 1 second. Default is 0.1 second.

    Returns
    -------
    Bunch with the following fields:

        trajectory, trajectory_sd : DataFrame
            Estimated trajectory and error standard deviations.
        gyro, gyro_sd : DataFrame
            Estimated gyro model parameters and its standard deviations.
        accel, accel_sd : DataFrame
            Estimated accelerometer model parameters and its standard deviations.
        innovations : dict of DataFrame
            For each observation class name contains DataFrame with measurement
            innovations.
    """
    if gyro_model is None:
        gyro_model = imu_model.InertialSensor()
    if accel_model is None:
        accel_model = imu_model.InertialSensor()
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
    P = _initialize_covariance(initial_pva, position_sd, velocity_sd, level_sd,
                               azimuth_sd, error_model, gyro_model, accel_model)

    ins_block = slice(error_model.N_STATES)
    gyro_block = slice(error_model.N_STATES, error_model.N_STATES + gyro_model.n_states)
    accel_block = slice(error_model.N_STATES + gyro_model.n_states, None)

    n_states = len(P)
    trajectory_result = []
    gyro_result = []
    accel_result = []
    P_result = []

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
                    H_full[:, ins_block] = H
                    innovation = kalman.correct(x, P, z, H_full, R)
                    name = observation.__class__.__name__
                    innovations[name].append(innovation)
                    innovations_times[name].append(observation_time)

            integrator.set_pva(
                error_model.correct_pva(integrator.get_pva(), x[ins_block]))
            gyro_model.update_estimates(x[gyro_block])
            accel_model.update_estimates(x[accel_block])
            observation_times_index += 1

        trajectory_result.append(integrator.get_pva())
        gyro_result.append(gyro_model.get_estimates())
        accel_result.append(accel_model.get_estimates())
        P_result.append(P)

        next_time = min(time + time_step,
                        observation_times[observation_times_index])
        next_increment_index = np.searchsorted(increments.index, next_time,
                                               side='right')
        increments_batch = _correct_increments(
            increments.iloc[increments_index : next_increment_index],
            gyro_model, accel_model)
        increments_index = next_increment_index

        pva_old = integrator.get_pva()
        integrator.integrate(increments_batch)
        pva_new = integrator.get_pva()
        time_delta = integrator.get_time() - time
        pva_average = _interpolate_pva(pva_old, pva_new, 0.5)
        gyro_average = increments_batch[THETA_COLS].sum(axis=0) / time_delta
        accel_average = increments_batch[DV_COLS].sum(axis=0) / time_delta

        Phi, Qd = _compute_error_propagation_matrices(pva_average, gyro_average,
                                                      accel_average, time_delta,
                                                      error_model, gyro_model,
                                                      accel_model)
        P = Phi @ P @ Phi.transpose() + Qd

    P_result = np.asarray(P_result)
    trajectory_result = pd.DataFrame(trajectory_result)
    trajectory_sd, gyro_sd, accel_sd = _compute_sd(
        P_result, trajectory_result, error_model, gyro_model, accel_model)

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


def run_feedforward_filter(trajectory_nominal, trajectory, position_sd, velocity_sd,
                           level_sd, azimuth_sd, gyro_model=None, accel_model=None,
                           observations=None, increments=None, time_step=0.1):
    if (trajectory_nominal.index != trajectory.index).any():
        raise ValueError(
            "`trajectory_nominal` and `trajectory` must have the same time index")
    times = trajectory_nominal.index

    if gyro_model is None:
        gyro_model = imu_model.InertialSensor()
    if accel_model is None:
        accel_model = imu_model.InertialSensor()
    if observations is None:
        observations = []

    if increments is None and (gyro_model.readings_required or
                               accel_model.readings_required):
        raise ValueError("When scale or misalignments errors are modelled, "
                         "`increments` must be provided")

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
    P = _initialize_covariance(trajectory_nominal.iloc[0], position_sd, velocity_sd,
                               level_sd, azimuth_sd, error_model,
                               gyro_model, accel_model)
    x = np.zeros(len(P))

    inertial_block = slice(error_model.N_STATES)

    n_states = len(P)
    x_result = []
    P_result = []
    times_result = []

    innovations = {}
    innovations_times = {}
    for observation in observations:
        name = observation.__class__.__name__
        innovations[name] = []
        innovations_times[name] = []

    index = 0
    while index + 1 < len(trajectory):
        time = times[index]
        next_time = times[index + 1]
        observation_time = observation_times[observation_times_index]
        if observation_time < next_time:
            pva = _interpolate_pva(trajectory.iloc[index], trajectory.iloc[index + 1],
                                   (observation_time - time) / (next_time - time))
            for observation in observations:
                ret = observation.compute_obs(observation_time, pva, error_model)
                if ret is not None:
                    z, H, R = ret
                    H_full = np.zeros((len(z), n_states))
                    H_full[:, inertial_block] = H
                    innovation = kalman.correct(x, P, z, H_full, R)
                    name = observation.__class__.__name__
                    innovations[name].append(innovation)
                    innovations_times[name].append(time)

            observation_times_index += 1

        times_result.append(time)
        x_result.append(x)
        P_result.append(P)

        next_time = min(time + time_step,
                        observation_times[observation_times_index])
        next_index = np.searchsorted(times, next_time, side='right') - 1
        next_time = times[next_index]
        time_delta = next_time - time

        pva_old = trajectory_nominal.iloc[index]
        pva_new = trajectory_nominal.iloc[next_index]
        pva_average = _interpolate_pva(pva_old, pva_new, 0.5)
        if increments is None:
            gyro_average = None
            accel_average = None
        else:
            increments_batch = increments.loc[np.nextafter(time, next_time) : next_time]
            gyro_average = increments_batch[THETA_COLS].sum(axis=0) / time_delta
            accel_average = increments_batch[DV_COLS].sum(axis=0) / time_delta

        Phi, Qd = _compute_error_propagation_matrices(pva_average, gyro_average,
                                                      accel_average, time_delta,
                                                      error_model, gyro_model,
                                                      accel_model)
        x = Phi @ x
        P = Phi @ P @ Phi.transpose() + Qd
        index = next_index

    times_result = np.asarray(times_result)
    x_result = np.asarray(x_result)
    P_result = np.asarray(P_result)
    trajectory_nominal = trajectory_nominal.loc[times_result]
    trajectory = trajectory.loc[times_result]

    trajectory, trajectory_sd, gyro, gyro_sd, accel, accel_sd = (
        _compute_feedforward_result(x_result, P_result, trajectory_nominal, trajectory,
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

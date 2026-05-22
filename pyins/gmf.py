from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.stats

from . import inertial_sensor, kalman, util, strapdown
from .error_model import InsErrorModel
from .util import (THETA_COLS, DV_COLS)
from ._filter_common import (interpolate_pva, initialize_covariance,
                             compute_error_propagation_matrices, compute_sd,
                             correct_increments)


@dataclass
class FilterState:
    weight: float
    integrator: strapdown.Integrator
    gyro_model: inertial_sensor.EstimationModel
    accel_model: inertial_sensor.EstimationModel
    P: np.ndarray


def run_gmf(initial_pva, position_sd, velocity_sd, level_sd,
            increments, n_components=12, critical_sigma=3.0, gyro_model=None,
            accel_model=None,
            measurements=None, time_step=0.1, with_altitude=True):
    """Run navigation filter with feedback corrections.

    Also known as extended Kalman filter (EKF).

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
    n_components : int
        Number of components in Gaussian mixture for initial heading estimation.
    gyro_model, accel_model : `pyins.inertial_sensor.EstimationModel`, optional
        Sensor models for gyros and accelerometers.
        If None (default), default models will be used.
    measurements : list of `pyins.measurements.Measurement` or None, optional
        List of measurements. If None (default), will be set to an empty list.
    time_step : float, optional
        Time step for covariance propagation.
        The value typically should not exceed 1 second. Default is 0.1 second.
    with_altitude : bool, optional
        Whether to estimate altitude or vertical velocity. Default is True.

    Returns
    -------
    Bunch with the following fields:

        trajectory, trajectory_sd : DataFrame
            Estimated trajectory and its error standard deviations.
            Note that `trajectory` will contain rows for each time moment, whereas
            `trajectory_sd` will contain rows only for some time moments (controlled
            by `time_step` parameter).
        gyro, gyro_sd : DataFrame
            Estimated gyro model parameters and its standard deviations.
        accel, accel_sd : DataFrame
            Estimated accelerometer model parameters and its standard deviations.
        innovations : dict of DataFrame
            For each measurement class name contains DataFrame with normalized
            measurement innovations.
    """
    critical_p = 2 * scipy.stats.norm.cdf(critical_sigma) - 1

    error_model = InsErrorModel(with_altitude)
    if gyro_model is None:
        gyro_model = inertial_sensor.EstimationModel()
    if accel_model is None:
        accel_model = inertial_sensor.EstimationModel()
    if measurements is None:
        measurements = []

    measurement_times = np.hstack([
        np.asarray(measurement.data.index) for measurement in measurements])
    measurement_times = np.sort(np.unique(measurement_times))

    start_time = initial_pva.name
    end_time = increments.index[-1]
    measurement_times = measurement_times[(measurement_times >= start_time) &
                                          (measurement_times <= end_time)]
    measurement_times = np.append(measurement_times, np.inf)
    measurement_time_index = 0

    heading_section_width = 360 / n_components
    azimuth_sd = heading_section_width / 12 ** 0.5

    gyro_model.reset_estimates()
    accel_model.reset_estimates()

    time = initial_pva.name
    states = []
    for start in np.linspace(0, 360, n_components, endpoint=False):
        initial_pva = initial_pva.copy()
        initial_pva.heading = start + 0.5 * heading_section_width

        states.append(FilterState(
            weight=1 / n_components,
            integrator=strapdown.Integrator(initial_pva, with_altitude),
            gyro_model=deepcopy(gyro_model),
            accel_model=deepcopy(accel_model),
            P=initialize_covariance(initial_pva, position_sd, velocity_sd, level_sd,
                                    azimuth_sd, error_model, gyro_model,
                                    accel_model)
        ))

    ins_block = slice(error_model.n_states)
    gyro_block = slice(error_model.n_states, error_model.n_states + gyro_model.n_states)
    accel_block = slice(error_model.n_states + gyro_model.n_states, None)

    del gyro_model
    del accel_model

    n_states = len(states[0].P)
    times_result = []
    weights = [[] for _ in range(n_components)]
    gyro_result = [[] for _ in range(n_components)]
    accel_result = [[] for _ in range(n_components)]
    P_result = [[] for _ in range(n_components)]

    innovations = {}
    innovations_times = {}
    for measurement in measurements:
        name = measurement.__class__.__name__
        innovations[name] = []
        innovations_times[name] = []

    increments_index = 0
    while time < end_time:
        measurement_time = measurement_times[measurement_time_index]
        increment = increments.iloc[increments_index]
        if measurement_time < increment.name:
            for state in states:
                increment_corrected = correct_increments(increment,
                                                         state.gyro_model,
                                                         state.accel_model)
                x = np.zeros(n_states)
                P = state.P
                pva = pd.concat([
                    state.integrator.predict(
                        (measurement_time - time) / increment_corrected['dt'] *
                        increment_corrected),
                    pd.Series(increment[THETA_COLS].values / increment_corrected['dt'],
                              index=['rate_x', 'rate_y', 'rate_z'])
                ])
                for measurement in measurements:
                    ret = measurement.compute_matrices(measurement_time, pva,
                                                       error_model)
                    if ret is not None:
                        z, H, R = ret
                        H_full = np.zeros((len(z), n_states))
                        H_full[:, ins_block] = H
                        x_new, P_new, innovation = kalman.correct(x, P, z, H_full, R)
                        chi2 = np.sum(innovation ** 2)
                        p = scipy.stats.chi2.cdf(chi2, len(z))
                        if p < critical_p:
                            x = x_new
                            P = P_new
                        state.weight *= np.exp(-0.5 * chi2)

                state.integrator.set_pva(
                    error_model.correct_pva(state.integrator.get_pva(), x[ins_block]))
                state.gyro_model.update_estimates(x[gyro_block])
                state.accel_model.update_estimates(x[accel_block])
                state.P = P

            weight_sum = sum(state.weight for state in states)
            for state in states:
                state.weight /= weight_sum

            measurement_time_index += 1

        times_result.append(time)
        for i, state in enumerate(states):
            weights[i].append(state.weight)
            gyro_result[i].append(state.gyro_model.get_estimates())
            accel_result[i].append(state.accel_model.get_estimates())
            P_result[i].append(state.P)

        next_time = min(time + time_step,
                        measurement_times[measurement_time_index])
        next_increment_index = np.searchsorted(increments.index, next_time,
                                               side='right')
        if next_increment_index == increments_index:
            next_increment_index += 1
        increments_batch = increments.iloc[increments_index: next_increment_index]
        increments_index = next_increment_index

        for state in states:
            increments_batch_corrected = correct_increments(increments_batch,
                                                            state.gyro_model,
                                                            state.accel_model)
            pva_old = state.integrator.get_pva()
            state.integrator.integrate(increments_batch_corrected)
            pva_new = state.integrator.get_pva()
            time_delta = state.integrator.get_time() - time
            pva_average = interpolate_pva(pva_old, pva_new, 0.5)
            gyro_average = increments_batch_corrected[THETA_COLS].sum(
                axis=0) / time_delta
            accel_average = increments_batch_corrected[DV_COLS].sum(axis=0) / time_delta

            Phi, Qd = compute_error_propagation_matrices(
                pva_average, gyro_average, accel_average, time_delta,
                error_model, state.gyro_model, state.accel_model)
            state.P = Phi @ state.P @ Phi.transpose() + Qd

        time = states[0].integrator.get_time()

    result = []
    for i in range(n_components):
        state = states[i]
        trajectory_sd, gyro_sd, accel_sd = compute_sd(
            np.asarray(P_result[i]), state.integrator.trajectory.loc[times_result],
            error_model, state.gyro_model, state.accel_model)
        result.append(util.Bunch(
            weights=pd.Series(weights[i], times_result),
            trajectory=state.integrator.trajectory,
            trajectory_sd=trajectory_sd,
            gyro=pd.DataFrame(gyro_result[i], index=times_result),
            gyro_sd=gyro_sd,
            accel=pd.DataFrame(accel_result[i], index=times_result),
            accel_sd=accel_sd,
            innovations=innovations))

    return result

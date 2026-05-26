from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.stats

from . import inertial_sensor, kalman, util, strapdown, transform
from .error_model import InsErrorModel
from .util import THETA_COLS, DV_COLS
from ._filter_common import (
    interpolate_pva,
    initialize_covariance,
    compute_error_propagation_matrices,
    compute_sd,
    correct_increments,
)


@dataclass
class FilterState:
    weight: float
    integrator: strapdown.Integrator
    gyro_model: inertial_sensor.EstimationModel
    accel_model: inertial_sensor.EstimationModel
    P: np.ndarray


@dataclass
class GmfSettings:
    """Settings for Gaussian Mixture Filter for heading estimation.

    The initial complete heading uncertainty of 360 degrees is split into even segments
    such that each component has small expected heading error and can be propagated
    with small error assumption. The component probabilities (weights) are adjusted
    based on how measurements evidence support a component's hypothesis. Components
    with very small weights are removed such that eventually a single component
    remains and the filter operates as a standard EKF.

    Attributes
    ----------
    n_components : int
        Number of mixture components.
    critical_sigma: float
        Measurement gating threshold expressed as a z-score of normal distribution.
        A component's state correction is skipped if the normalized innovation exceeds
        this value, indicating the measurement contradicts the component hypothesis.
        Prevents erroneous EKF corrections which can cause unpredictable results due
        to violation of inherent EKF assumptions.
    minimal_relative_weight:
        Weight threshold for component removal. If component weight becomes less
        that `minimal_relative_weight / n_components` the component is removed
        from the mixture.
    heading_accuracy_for_collapse:
        If total heading accuracy becomes less than this value, all the components
        are collapsed into one. It prevents the filter from maintaining several
        very similar components unnecessarily.
    """

    n_components = 12
    critical_sigma = 3.0
    minimal_relative_weight = 0.01
    heading_accuracy_for_collapse = 5.0


def run_gaussian_mixture_filter(
    initial_pva,
    position_sd,
    velocity_sd,
    level_sd,
    increments,
    gyro_model=None,
    accel_model=None,
    measurements=None,
    time_step=0.1,
    with_altitude=True,
    settings=GmfSettings(),
):
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
    increments : Increments
        IMU increments.
    settings : GmfSettings, optional
        Settings of Gaussian Mixture Filter algorithm.
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
    critical_p = 2 * scipy.stats.norm.cdf(settings.critical_sigma) - 1

    error_model = InsErrorModel(with_altitude)
    if gyro_model is None:
        gyro_model = inertial_sensor.EstimationModel()
    if accel_model is None:
        accel_model = inertial_sensor.EstimationModel()
    if measurements is None:
        measurements = []

    measurement_times = np.hstack(
        [np.asarray(measurement.data.index) for measurement in measurements]
    )
    measurement_times = np.sort(np.unique(measurement_times))

    start_time = initial_pva.name
    end_time = increments.index[-1]
    measurement_times = measurement_times[
        (measurement_times >= start_time) & (measurement_times <= end_time)
    ]
    measurement_times = np.append(measurement_times, np.inf)
    measurement_time_index = 0

    heading_section_width = 360 / settings.n_components
    azimuth_sd = heading_section_width / 12**0.5

    gyro_model.reset_estimates()
    accel_model.reset_estimates()

    time = initial_pva.name
    states = []
    for start in np.linspace(0, 360, settings.n_components, endpoint=False):
        initial_pva = initial_pva.copy()
        initial_pva.heading = start + 0.5 * heading_section_width

        states.append(
            FilterState(
                weight=1 / settings.n_components,
                integrator=strapdown.Integrator(initial_pva, with_altitude),
                gyro_model=deepcopy(gyro_model),
                accel_model=deepcopy(accel_model),
                P=initialize_covariance(
                    initial_pva,
                    position_sd,
                    velocity_sd,
                    level_sd,
                    azimuth_sd,
                    error_model,
                    gyro_model,
                    accel_model,
                ),
            )
        )

    ins_block = slice(error_model.n_states)
    gyro_block = slice(error_model.n_states, error_model.n_states + gyro_model.n_states)
    accel_block = slice(error_model.n_states + gyro_model.n_states, None)

    n_states = len(states[0].P)
    times_result = []
    weights_result = []
    pva_result = []
    gyro_result = []
    accel_result = []
    P_result = []

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
                increment_corrected = correct_increments(
                    increment, state.gyro_model, state.accel_model
                )
                x = np.zeros(n_states)
                P = state.P
                pva = pd.concat(
                    [
                        state.integrator.predict(
                            (measurement_time - time)
                            / increment_corrected["dt"]
                            * increment_corrected
                        ),
                        pd.Series(
                            increment[THETA_COLS].values / increment_corrected["dt"],
                            index=["rate_x", "rate_y", "rate_z"],
                        ),
                    ]
                )
                for measurement in measurements:
                    ret = measurement.compute_matrices(
                        measurement_time, pva, error_model
                    )
                    if ret is None:
                        continue

                    z, H, R = ret
                    H_full = np.zeros((len(z), n_states))
                    H_full[:, ins_block] = H
                    x_new, P_new, innovation, evidence = kalman.correct(
                        x, P, z, H_full, R
                    )
                    chi2 = np.sum(innovation**2)
                    if scipy.stats.chi2.cdf(chi2, len(z)) < critical_p:
                        x = x_new
                        P = P_new
                    state.weight *= evidence

                    if len(states) == 1:
                        name = measurement.__class__.__name__
                        innovations[name].append(innovation)
                        innovations_times[name].append(measurement_time)

                state.integrator.set_pva(
                    error_model.correct_pva(state.integrator.get_pva(), x[ins_block])
                )
                state.gyro_model.update_estimates(x[gyro_block])
                state.accel_model.update_estimates(x[accel_block])
                state.P = P

            weight_sum = sum(state.weight for state in states)
            for state in states:
                state.weight /= weight_sum

            required_weight = settings.minimal_relative_weight / len(states)
            states = [state for state in states if state.weight > required_weight]
            measurement_time_index += 1

        weights = [state.weight for state in states]
        if len(states) == 1:
            state = states[0]
            mean_pva = state.integrator.get_pva()
            mean_gyro_state = state.gyro_model.get_estimates()
            mean_accel_state = state.accel_model.get_estimates()
            P_total = state.P
        else:
            pvas = pd.DataFrame([state.integrator.get_pva() for state in states])
            mean_pva = transform.compute_mean_state(pvas, weights)
            gyro_states = pd.DataFrame(
                [state.gyro_model.get_estimates() for state in states]
            )
            mean_gyro_state = transform.compute_mean_state(gyro_states, weights)
            accel_states = pd.DataFrame(
                [state.accel_model.get_estimates() for state in states]
            )
            mean_accel_state = transform.compute_mean_state(accel_states, weights)

            Ps = [state.P for state in states]

            P_total = np.zeros((n_states, n_states))
            for weight, P, (_, pva), (_, gyro_state), (_, accel_state) in zip(
                weights,
                Ps,
                pvas.iterrows(),
                gyro_states.iterrows(),
                accel_states.iterrows(),
            ):
                diff_vector = np.hstack(
                    [
                        error_model.subtract_pva(pva, mean_pva),
                        gyro_state - mean_gyro_state,
                        accel_state - mean_accel_state,
                    ]
                )
                P_total += weight * (P + np.outer(diff_vector, diff_vector))

            if (
                P_total[error_model.PHI3, error_model.PHI3]
                < (settings.heading_accuracy_for_collapse * transform.DEG_TO_RAD) ** 2
            ):
                state = states[0]
                state.weight = 1.0
                state.integrator.set_pva(mean_pva)
                state.gyro_model.set_estimates(mean_gyro_state)
                state.accel_model.set_estimates(mean_accel_state)
                state.P = P_total
                states = [state]

        times_result.append(time)
        weights_result.append(weights)
        pva_result.append(mean_pva)
        gyro_result.append(mean_gyro_state)
        accel_result.append(mean_accel_state)
        P_result.append(P_total)

        next_time = min(time + time_step, measurement_times[measurement_time_index])
        next_increment_index = np.searchsorted(
            increments.index, next_time, side="right"
        )
        if next_increment_index == increments_index:
            next_increment_index += 1
        increments_batch = increments.iloc[increments_index:next_increment_index]
        increments_index = next_increment_index

        for state in states:
            increments_batch_corrected = correct_increments(
                increments_batch, state.gyro_model, state.accel_model
            )
            pva_old = state.integrator.get_pva()
            state.integrator.integrate(increments_batch_corrected)
            pva_new = state.integrator.get_pva()
            time_delta = state.integrator.get_time() - time
            pva_average = interpolate_pva(pva_old, pva_new, 0.5)
            gyro_average = (
                increments_batch_corrected[THETA_COLS].sum(axis=0) / time_delta
            )
            accel_average = increments_batch_corrected[DV_COLS].sum(axis=0) / time_delta

            Phi, Qd = compute_error_propagation_matrices(
                pva_average,
                gyro_average,
                accel_average,
                time_delta,
                error_model,
                state.gyro_model,
                state.accel_model,
            )
            state.P = Phi @ state.P @ Phi.transpose() + Qd

        time = states[0].integrator.get_time()

    trajectory = pd.DataFrame(pva_result, times_result)
    trajectory_sd, gyro_sd, accel_sd = compute_sd(
        np.asarray(P_result), trajectory, error_model, gyro_model, accel_model
    )

    for measurement in measurements:
        name = measurement.__class__.__name__
        innovation = innovations[name]
        if innovation:
            columns = measurement.data.columns[: len(innovation[0])]
        else:
            columns = measurement.data.columns
        innovations[name] = pd.DataFrame(
            innovations[name], innovations_times[name], columns=columns
        )

    return util.Bunch(
        weights=pd.Series(weights_result, times_result),
        trajectory=trajectory,
        trajectory_sd=trajectory_sd,
        gyro=pd.DataFrame(gyro_result, index=times_result),
        gyro_sd=gyro_sd,
        accel=pd.DataFrame(accel_result, index=times_result),
        accel_sd=accel_sd,
        innovations=innovations,
    )

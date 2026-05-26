"""Navigation Kalman filters.

Module provides functions to run a navigation Kalman filter in feedback
(extended Kalman filter) and feedforward (linearized Kalman filter) forms. It relies on
functionality provided by `pyins.inertial_sensor`, `pyins.error_model` and
`pyins.measurements` modules.

Refer to [1]_ for the discussion of Kalman filtering in context of inertial navigation.

Classes
-------
.. autosummary::
    :toctree: generated/

    GmfSettings

Functions
---------
.. autosummary::
    :toctree: generated/

    run_feedback_filter
    run_feedforward_filter

References
----------
.. [1] P. D. Groves, "Principles of GNSS, Inertial, and Multisensor Integrated
       Navigation Systems", 2nd edition
"""

import numpy as np
import pandas as pd
from . import inertial_sensor, kalman, util, strapdown
from ._gmf import GmfSettings, run_gaussian_mixture_filter
from .error_model import InsErrorModel
from .util import THETA_COLS, DV_COLS
from ._filter_common import (
    interpolate_pva,
    initialize_covariance,
    compute_error_propagation_matrices,
    compute_sd,
    compute_feedforward_result,
    correct_increments,
)


def run_feedback_filter(
    initial_pva,
    position_sd,
    velocity_sd,
    level_sd,
    azimuth_sd,
    increments,
    gyro_model=None,
    accel_model=None,
    measurements=None,
    time_step=0.1,
    with_altitude=True,
    gmf_settings=GmfSettings(),
):
    """Run navigation filter with feedback corrections.

    Also known as extended Kalman filter (EKF). It can be run as Gaussian Mixture Filter
    with complete heading uncertainty as well.

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
    azimuth_sd : float or None
        Initial assumed heading standard deviation in degrees.
        If None, complete 360 degrees heading uncertainty is assumed and
        Gaussian Mixture Filter (GMF) is run. See `GmfSettings` for a short description
        of the algorithm.
    increments : Increments
        IMU increments.
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
    gmf_settings : `GmfSettings`, optional
        Settings for GMF to be used when heading is estimated from total uncertainty.
        Contains sensible parameters by default.

    Returns
    -------
    Bunch with the following fields:

        trajectory, trajectory_sd : DataFrame
            Estimated trajectory and its error standard deviations.
            When run in normal mode, `trajectory` contains rows for each time
            moment and when run in GMF mode -- only for the time moments where
            the covariance was computed (controlled by `time_step`).
            And `trajectory_sd` always contains rows only for time moments where
            the covariance was computed.
        gyro, gyro_sd : DataFrame
            Estimated gyro model parameters and its standard deviations.
        accel, accel_sd : DataFrame
            Estimated accelerometer model parameters and its standard deviations.
        innovations : dict of DataFrame
            For each measurement class name contains DataFrame with normalized
            measurement innovations.
    """
    if azimuth_sd is None:
        return run_gaussian_mixture_filter(
            initial_pva,
            position_sd,
            velocity_sd,
            level_sd,
            increments,
            gyro_model,
            accel_model,
            measurements,
            time_step,
            with_altitude,
            gmf_settings,
        )

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

    P = initialize_covariance(
        initial_pva,
        position_sd,
        velocity_sd,
        level_sd,
        azimuth_sd,
        error_model,
        gyro_model,
        accel_model,
    )

    ins_block = slice(error_model.n_states)
    gyro_block = slice(error_model.n_states, error_model.n_states + gyro_model.n_states)
    accel_block = slice(error_model.n_states + gyro_model.n_states, None)

    n_states = len(P)
    times_result = []
    gyro_result = []
    accel_result = []
    P_result = []

    integrator = strapdown.Integrator(initial_pva, with_altitude)
    gyro_model.reset_estimates()
    accel_model.reset_estimates()

    innovations = {}
    innovations_times = {}
    for measurement in measurements:
        name = measurement.__class__.__name__
        innovations[name] = []
        innovations_times[name] = []

    increments_index = 0
    while integrator.get_time() < end_time:
        time = integrator.get_time()
        measurement_time = measurement_times[measurement_time_index]
        increment = correct_increments(
            increments.iloc[increments_index], gyro_model, accel_model
        )
        if measurement_time < increment.name:
            x = np.zeros(n_states)
            pva = pd.concat(
                [
                    integrator.predict(
                        (measurement_time - time) / increment["dt"] * increment
                    ),
                    pd.Series(
                        increment[THETA_COLS].values / increment["dt"],
                        index=["rate_x", "rate_y", "rate_z"],
                    ),
                ]
            )
            for measurement in measurements:
                ret = measurement.compute_matrices(measurement_time, pva, error_model)
                if ret is not None:
                    z, H, R = ret
                    H_full = np.zeros((len(z), n_states))
                    H_full[:, ins_block] = H
                    x, P, innovation, _ = kalman.correct(x, P, z, H_full, R)
                    name = measurement.__class__.__name__
                    innovations[name].append(innovation)
                    innovations_times[name].append(measurement_time)

            integrator.set_pva(
                error_model.correct_pva(integrator.get_pva(), x[ins_block])
            )
            gyro_model.update_estimates(x[gyro_block])
            accel_model.update_estimates(x[accel_block])
            measurement_time_index += 1

        times_result.append(time)
        gyro_result.append(gyro_model.get_estimates())
        accel_result.append(accel_model.get_estimates())
        P_result.append(P)

        next_time = min(time + time_step, measurement_times[measurement_time_index])
        next_increment_index = np.searchsorted(
            increments.index, next_time, side="right"
        )
        if next_increment_index == increments_index:
            next_increment_index += 1
        increments_batch = correct_increments(
            increments.iloc[increments_index:next_increment_index],
            gyro_model,
            accel_model,
        )
        increments_index = next_increment_index

        pva_old = integrator.get_pva()
        integrator.integrate(increments_batch)
        pva_new = integrator.get_pva()
        time_delta = integrator.get_time() - time
        pva_average = interpolate_pva(pva_old, pva_new, 0.5)
        gyro_average = increments_batch[THETA_COLS].sum(axis=0) / time_delta
        accel_average = increments_batch[DV_COLS].sum(axis=0) / time_delta

        Phi, Qd = compute_error_propagation_matrices(
            pva_average,
            gyro_average,
            accel_average,
            time_delta,
            error_model,
            gyro_model,
            accel_model,
        )
        P = Phi @ P @ Phi.transpose() + Qd

    P_result = np.asarray(P_result)
    trajectory_sd, gyro_sd, accel_sd = compute_sd(
        P_result,
        integrator.trajectory.loc[times_result],
        error_model,
        gyro_model,
        accel_model,
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
        trajectory=integrator.trajectory,
        trajectory_sd=trajectory_sd,
        gyro=pd.DataFrame(gyro_result, index=times_result),
        gyro_sd=gyro_sd,
        accel=pd.DataFrame(accel_result, index=times_result),
        accel_sd=accel_sd,
        innovations=innovations,
    )


def run_feedforward_filter(
    trajectory_nominal,
    trajectory,
    position_sd,
    velocity_sd,
    level_sd,
    azimuth_sd,
    gyro_model=None,
    accel_model=None,
    measurements=None,
    increments=None,
    time_step=0.1,
    with_altitude=True,
):
    """Run navigation filter with feedforward output compensation.

    Also known as a linearized Kalman filter.
    The approach is applicable in practice only for high-end precise INS systems.
    It can also be used for covariance modelling given an accurate reference trajectory
    (called `trajectory_nominal` in the function).

    Parameters
    ----------
    trajectory_nominal, trajectory : Trajectory
        Nominal and actually computed trajectories. The nominal trajectory can be
        set to the computed trajectory (standard practical situation) or to an accurate
        reference trajectory if available in a modelling scenario ("covariance
        analysis"). Both trajectories must have exactly the same time index.
    position_sd : float
        Initial assumed position standard deviation in meters.
    velocity_sd : float
        Initial assumed velocity standard deviation in m/s.
    level_sd : float
        Initial assumed roll and pitch standard deviation in degrees.
    azimuth_sd : float
        Initial assumed heading standard deviation in degrees.
    gyro_model, accel_model : pyins.inertial_sensor.EstimationModel, optional
        Sensor models for gyros and accelerometers.
         If None (default), default models will be used.
    measurements : list of `pyins.measurements.Measurement` or None, optional
        List of measurements. If None (default), will be set to an empty list.
    increments : Increments or None, optional
        IMU increments to be used when gyro or accelerometers scale factor errors or
        misalignment are modelled. Not necessary otherwise.
    time_step : float, optional
        Time step for covariance and error state propagation.
        The value typically should not exceed 1 second. Default is 0.1 second.
    with_altitude : bool, optional
        Whether to estimate altitude or vertical velocity. Default is True.

    Returns
    -------
    Bunch with the following fields:

        trajectory, trajectory_sd : DataFrame
            Estimated trajectory and its error standard deviations.
            Unlike `run_feedback_filter` both will contain rows only for time moments
            when the error vector and covariance are computed (controlled
            by `time_step` parameter).
        gyro, gyro_sd : DataFrame
            Estimated gyro model parameters and its standard deviations.
        accel, accel_sd : DataFrame
            Estimated accelerometer model parameters and its standard deviations.
        innovations : dict of DataFrame
            For each measurement class name contains DataFrame with normalized
            measurement innovations.
    """
    if (trajectory_nominal.index != trajectory.index).any():
        raise ValueError(
            "`trajectory_nominal` and `trajectory` must have the same time index"
        )
    times = trajectory_nominal.index

    error_model = InsErrorModel(with_altitude)
    if gyro_model is None:
        gyro_model = inertial_sensor.EstimationModel()
    gyro_model.reset_estimates()

    if accel_model is None:
        accel_model = inertial_sensor.EstimationModel()
    accel_model.reset_estimates()

    if measurements is None:
        measurements = []

    if increments is None and (
        gyro_model.scale_misal_modelled or accel_model.scale_misal_modelled
    ):
        raise ValueError(
            "When scale or misalignments errors are modelled, "
            "`increments` must be provided"
        )

    measurement_times = np.hstack(
        [np.asarray(measurement.data.index) for measurement in measurements]
    )
    measurement_times = np.sort(np.unique(measurement_times))

    start_time = times[0]
    end_time = times[-1]
    measurement_times = measurement_times[
        (measurement_times >= start_time) & (measurement_times <= end_time)
    ]
    measurement_times = np.append(measurement_times, np.inf)
    measurement_time_index = 0

    P = initialize_covariance(
        trajectory_nominal.iloc[0],
        position_sd,
        velocity_sd,
        level_sd,
        azimuth_sd,
        error_model,
        gyro_model,
        accel_model,
    )
    x = np.zeros(len(P))

    inertial_block = slice(error_model.n_states)

    n_states = len(P)
    x_result = []
    P_result = []
    times_result = []

    innovations = {}
    innovations_times = {}
    for measurement in measurements:
        name = measurement.__class__.__name__
        innovations[name] = []
        innovations_times[name] = []

    index = 0
    while index + 1 < len(trajectory):
        time = times[index]
        next_time = times[index + 1]
        measurement_time = measurement_times[measurement_time_index]
        if measurement_time < next_time:
            pva = interpolate_pva(
                trajectory.iloc[index],
                trajectory.iloc[index + 1],
                (measurement_time - time) / (next_time - time),
            )
            for measurement in measurements:
                ret = measurement.compute_matrices(measurement_time, pva, error_model)
                if ret is not None:
                    z, H, R = ret
                    H_full = np.zeros((len(z), n_states))
                    H_full[:, inertial_block] = H
                    x, P, innovation, _ = kalman.correct(x, P, z, H_full, R)
                    name = measurement.__class__.__name__
                    innovations[name].append(innovation)
                    innovations_times[name].append(time)

            measurement_time_index += 1

        times_result.append(time)
        x_result.append(x)
        P_result.append(P)

        next_time = min(time + time_step, measurement_times[measurement_time_index])
        next_index = np.searchsorted(times, next_time, side="right") - 1
        next_time = times[next_index]
        time_delta = next_time - time

        pva_old = trajectory_nominal.iloc[index]
        pva_new = trajectory_nominal.iloc[next_index]
        pva_average = interpolate_pva(pva_old, pva_new, 0.5)
        if increments is None:
            gyro_average = None
            accel_average = None
        else:
            increments_batch = increments.loc[np.nextafter(time, next_time) : next_time]
            gyro_average = increments_batch[THETA_COLS].sum(axis=0) / time_delta
            accel_average = increments_batch[DV_COLS].sum(axis=0) / time_delta

        Phi, Qd = compute_error_propagation_matrices(
            pva_average,
            gyro_average,
            accel_average,
            time_delta,
            error_model,
            gyro_model,
            accel_model,
        )
        x = Phi @ x
        P = Phi @ P @ Phi.transpose() + Qd
        index = next_index

    times_result = np.asarray(times_result)
    x_result = np.asarray(x_result)
    P_result = np.asarray(P_result)
    trajectory_nominal = trajectory_nominal.loc[times_result]
    trajectory = trajectory.loc[times_result]

    trajectory, trajectory_sd, gyro, gyro_sd, accel, accel_sd = (
        compute_feedforward_result(
            x_result,
            P_result,
            trajectory_nominal,
            trajectory,
            error_model,
            gyro_model,
            accel_model,
        )
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
        trajectory=trajectory,
        trajectory_sd=trajectory_sd,
        gyro=gyro,
        gyro_sd=gyro_sd,
        accel=accel,
        accel_sd=accel_sd,
        innovations=innovations,
    )

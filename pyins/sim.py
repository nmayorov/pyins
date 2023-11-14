"""Simulation of sensors.

The main functionality is synthesis of IMU readings from the given trajectory.
It also contains some utility functions useful for aides INS simulation.

Functions
---------
.. autosummary::
    :toctree: generated/

    generate_imu
    sinusoid_velocity_motion
    generate_position_measurements
    generate_ned_velocity_measurements
    generate_body_velocity_measurements
    generate_pva_error
    perturb_pva
"""
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, CubicHermiteSpline
from scipy.spatial.transform import Rotation, RotationSpline
from scipy._lib._util import check_random_state
from . import earth, transform, util
from .util import (LLA_COLS, VEL_COLS, RPH_COLS, NED_COLS, GYRO_COLS, ACCEL_COLS,
                   TRAJECTORY_COLS, TRAJECTORY_ERROR_COLS)


def _compute_increment_readings(dt, a, b, c, d, e):
    ab = np.cross(a, b)
    ac = np.cross(a, c)
    bc = np.cross(b, c)

    omega = np.empty((8,) + a.shape)
    omega[0] = a
    omega[1] = 2 * b
    omega[2] = 3 * c - 0.5 * ab
    omega[3] = -ac + np.cross(a, ab) / 6
    omega[4] = -0.5 * bc + np.cross(a, ac) / 3 + np.cross(b, ab) / 6
    omega[5] = np.cross(a, bc) / 6 + np.cross(b, ac) / 3 + np.cross(c, ab) / 6
    omega[6] = np.cross(b, bc) / 6 + np.cross(c, ac) / 3
    omega[7] = np.cross(c, bc) / 6

    gyros = 0
    for k in reversed(range(8)):
        gyros += omega[k] / (k + 1)
        gyros *= dt

    ad = np.cross(a, d)
    ae = np.cross(a, e)
    bd = np.cross(b, d)
    be = np.cross(b, e)
    cd = np.cross(c, d)
    ce = np.cross(c, e)

    f = np.empty((8,) + d.shape)
    f[0] = d
    f[1] = e - ad
    f[2] = -ae - bd + 0.5 * np.cross(a, ad)
    f[3] = -be - cd + 0.5 * (np.cross(a, ae + bd) + np.cross(b, ad))
    f[4] = -ce + 0.5 * (np.cross(a, be + cd) + np.cross(b, ae + bd) +
                        np.cross(c, ad))
    f[5] = 0.5 * (np.cross(a, ce) + np.cross(b, be + cd) +
                  np.cross(c, ae + bd))
    f[6] = 0.5 * (np.cross(b, ce) + np.cross(c, be + cd))
    f[7] = 0.5 * np.cross(c, ce)

    accels = 0
    for k in reversed(range(8)):
        accels += f[k] / (k + 1)
        accels *= dt

    return gyros, accels


def generate_imu(time, lla, rph, velocity_n=None, sensor_type='rate'):
    """Generate IMU readings from the trajectory.

    Attitude angles (`rph`) must be always given and there are 3 options for
    position and velocity:

        - Both position and velocity are given
        - Only position is given
        - Initial position and velocity are given

    Parameters
    ----------
    time : array_like, shape (n_points,)
        Time points for which the trajectory is provided.
    lla : array_like, shape (n_points, 3) or (3,)
        Either time series of latitude, longitude and altitude or initial
        values of those.
    rph : array_like, shape (n_points, 3)
        Time series of roll, pitch and heading angles.
    velocity_n : array_like, shape (n_points, 3) or None
        Time series of velocity expressed in NED frame.
    sensor_type: 'rate' or 'increment', optional
        Type of sensor to generate. If 'rate' (default), then instantaneous
        rate values are generated (in rad/s and m/s/s). If 'increment', then
        integrals over sampling intervals are generated (in rad and m/s).

    Returns
    -------
    trajectory : Trajectory
        Trajectory dataframe with n_points rows.
    imu : Imu
        IMU dataframe with n_points rows. When `sensor_type` is 'increment' the first
        sample is duplicated for more convenient future processing.
    """
    MAX_ITER = 3
    ACCURACY = 0.01

    if sensor_type not in ['rate', 'increment']:
        raise ValueError("`sensor_type` must be 'rate' or 'increment'.")

    lla = np.asarray(lla)
    if lla.ndim == 1 and velocity_n is None:
        raise ValueError("`velocity_n` must be provided when `lla` contains only "
                         "initial values")

    n_points = len(time)
    if lla.ndim == 1:
        lat0, lon0, alt0 = lla
        velocity_n = np.asarray(velocity_n, dtype=float)
        VU_spline = CubicSpline(time, -velocity_n[:, 2])
        alt_spline = VU_spline.antiderivative()
        alt = alt0 + alt_spline(time)

        lat = lat0 = np.deg2rad(lat0)
        for iteration in range(MAX_ITER):
            rn, _, _ = earth.principal_radii(np.rad2deg(lat), alt)
            dlat_spline = CubicSpline(time, velocity_n[:, 0] / rn)
            lat_spline = dlat_spline.antiderivative()
            lat_new = lat0 + lat_spline(time)
            delta = (lat - lat_new) * rn
            lat = lat_new
            if np.all(np.abs(delta) < ACCURACY):
                break

        lat = np.rad2deg(lat)
        _, _, rp = earth.principal_radii(lat, alt)
        dlon_spline = CubicSpline(time, velocity_n[:, 1] / rp)
        lon_spline = dlon_spline.antiderivative()

        lla = np.empty((n_points, 3))
        lla[:, 0] = lat
        lla[:, 1] = lon0 + np.rad2deg(lon_spline(time))
        lla[:, 2] = alt

    lla_inertial = lla.copy()
    lla_inertial[:, 1] += np.rad2deg(earth.RATE) * time
    mat_in = transform.mat_en_from_ll(lla_inertial[:, 0], lla_inertial[:, 1])

    r_i = transform.lla_to_ecef(lla_inertial)
    earth_rate_i = [0, 0, earth.RATE]
    if velocity_n is None:
        v_i_spline = CubicSpline(time, r_i).derivative()
        velocity_n = util.mv_prod(
            mat_in, v_i_spline(time) - np.cross(earth_rate_i, r_i), True)
    else:
        v_i = util.mv_prod(mat_in, velocity_n) + np.cross(earth_rate_i, r_i)
        v_i_spline = CubicHermiteSpline(time, r_i, v_i).derivative()

    mat_ib = util.mm_prod(mat_in, transform.mat_from_rph(rph))
    rot_ib_spline = RotationSpline(time, Rotation.from_matrix(mat_ib))
    g_i = earth.gravitation_ecef(lla_inertial)

    if sensor_type == 'rate':
        gyro = rot_ib_spline(time, 1)
        accel = util.mv_prod(mat_ib, v_i_spline(time, 1) - g_i, at=True)
    elif sensor_type == 'increment':
        dt = np.diff(time)[:, None]

        a = rot_ib_spline.interpolator.c[2]
        b = rot_ib_spline.interpolator.c[1]
        c = rot_ib_spline.interpolator.c[0]

        a_s = v_i_spline.derivative()
        d = a_s.c[1] - g_i[:-1]
        e = a_s.c[0] - np.diff(g_i, axis=0) / dt

        d = util.mv_prod(mat_ib[:-1], d, at=True)
        e = util.mv_prod(mat_ib[:-1], e, at=True)

        gyro, accel = _compute_increment_readings(dt, a, b, c, d, e)
        gyro = np.insert(gyro, 0, gyro[0], axis=0)
        accel = np.insert(accel, 0, accel[0], axis=0)
    else:
        assert False

    index = pd.Index(time, name='time')
    return (pd.DataFrame(np.hstack([lla, velocity_n, rph]),
                         index=index, columns=TRAJECTORY_COLS),
            pd.DataFrame(data=np.hstack((gyro, accel)), index=index,
                         columns=GYRO_COLS + ACCEL_COLS))


def sinusoid_velocity_motion(dt, total_time, lla0, velocity_mean,
                             velocity_change_amplitude=0,
                             velocity_change_period=60,
                             velocity_change_phase_offset=[0, 90, 0],
                             sensor_type='rate'):
    """Generate trajectory with NED velocity changing as sine.

    The NED velocity changes as::

        V = V_mean + V_ampl * sin(2 * pi * t / period + phase_offset)

    Roll is set to zero, pitch and heading angles are computed with zero
    lateral and vertical velocity assumptions.

    Parameters
    ----------
    dt : float
        Time step.
    total_time : float
        Total motion time.
    lla0 : array_like, shape (3,)
        Initial latitude, longitude and altitude.
    velocity_mean : array_like, shape (3,)
        Mean velocity resolved in NED.
    velocity_change_amplitude : array_like, optional
        Velocity change amplitude. Default is 0.
    velocity_change_period : float, optional
        Period of sinusoidal velocity change in seconds. Default is 60.
    velocity_change_phase_offset : array_like, shape (3,), optional
        Phase offset for sinusoid part in degrees. Default is [0, 90, 0]
        which will create an ellipse for latitude-longitude trajectory when
        the mean velocity is zero.
    sensor_type: 'rate' or 'increment', optional
        Type of sensor to generate. If 'rate' (default), then instantaneous
        rate values are generated (in rad/s and m/s/s). If 'increment', then
        integrals over sampling intervals are generated (in rad and m/s).

    Returns
    -------
    trajectory : Trajectory
        Trajectory dataframe with n_points rows.
    imu : Imu
        IMU dataframe with n_points rows. When `sensor_type` is 'increment' the first
        sample is duplicated for more convenient future processing.
    """
    time = np.arange(0, total_time, dt)
    phase = (2 * np.pi * time[:, None] / velocity_change_period +
             np.deg2rad(velocity_change_phase_offset))
    velocity_n = (np.atleast_2d(velocity_mean) +
                  np.atleast_2d(velocity_change_amplitude) * np.sin(phase))
    rph = np.zeros_like(velocity_n)
    rph[:, 1] = np.rad2deg(np.arctan2(
        velocity_n[:, 2], np.hypot(velocity_n[:, 0], velocity_n[:, 1])))
    rph[:, 2] = np.rad2deg(np.arctan2(velocity_n[:, 1], velocity_n[:, 0]))
    return generate_imu(time, lla0, rph, velocity_n, sensor_type)


def generate_position_measurements(trajectory, error_sd, rng=None):
    """Generate data with latitude-longitude-altitude position measurements.

    The measurements are computed as given (true) values perturbed by normal random
    errors.

    Parameters
    ----------
    trajectory : Trajectory
        Trajectory dataframe, only 'lat', 'lon' and 'alt' columns are used.
    error_sd : float
        Standard deviation of errors in meters.
    rng : None, int or numpy.RandomState
        Random seed.

    Returns
    -------
    DataFrame
        Contains 'lat', 'lon' and 'alt' columns with perturbed coordinates.
    """
    rng = check_random_state(rng)
    error = error_sd * rng.randn(len(trajectory), 3)
    lla = transform.perturb_lla(trajectory[LLA_COLS], error)
    return pd.DataFrame(data=lla, index=trajectory.index, columns=LLA_COLS)


def generate_ned_velocity_measurements(trajectory, error_sd, rng=None):
    """Generate data with NED velocity measurements.

    The measurements are computed as given (true) values perturbed by normal random
    errors.

    Parameters
    ----------
    trajectory : Trajectory
        Trajectory dataframe, only 'VN', 'VE' and 'VD' columns are used.
    error_sd : float
        Standard deviation of errors in m/s.
    rng : None, int or numpy.RandomState
        Random seed.

    Returns
    -------
    DataFrame
        Contains 'VN', 'VE' and 'VD' columns with perturbed velocities.
    """
    rng = check_random_state(rng)
    error = error_sd * rng.randn(len(trajectory), 3)
    velocity_n = trajectory[VEL_COLS] + error
    return pd.DataFrame(data=velocity_n, index=trajectory.index, columns=VEL_COLS)


def generate_body_velocity_measurements(trajectory, error_sd, rng=None):
    """Generate data with velocity measurements expressed in body frame.

    The measurements are computed by first projecting NED velocity into the body frame
    and then adding normal random errors to it.

    Parameters
    ----------
    trajectory : Trajectory
        Trajectory dataframe, only 'VN', VE', 'VD', 'roll', 'pitch' and 'heading'
        columns are used.
    error_sd : float
        Standard deviation of errors in m/s.
    rng : None, int or numpy.RandomState
        Random seed.

    Returns
    -------
    DataFrame
        Contains 'VX', 'VY' and 'VZ' columns with perturbed velocities.
    """
    rng = check_random_state(rng)
    error = error_sd * rng.randn(len(trajectory), 3)
    mat_nb = transform.mat_from_rph(trajectory[RPH_COLS])
    velocity_b = util.mv_prod(mat_nb, trajectory[VEL_COLS], at=True) + error
    return pd.DataFrame(data=velocity_b, index=trajectory.index,
                        columns=['VX', 'VY', 'VZ'])


def generate_pva_error(position_sd, velocity_sd, level_sd, azimuth_sd, rng=None):
    """Generate random position-velocity-attitude error.

    All errors are generated as independent and normally distributed.

    Parameters
    ----------
    position_sd : float
        Position error standard deviation in meters.
    velocity_sd : float
        Velocity error standard deviation in m/s.
    level_sd : float
        Roll and pitch standard deviation in degrees.
    azimuth_sd : float
        Heading standard deviation in degrees.
    rng : None, int or RandomState
        Random seed.

    Returns
    -------
    PvaError
        Series containing 9 elements with position-velocity-attitude errors.
    """
    rng = check_random_state(rng)
    result = pd.Series(index=TRAJECTORY_ERROR_COLS)
    result[NED_COLS] = position_sd * rng.randn(3)
    result[VEL_COLS] = velocity_sd * rng.randn(3)
    result[RPH_COLS] = [level_sd, level_sd, azimuth_sd] * rng.randn(3)
    return result


def perturb_pva(pva, pva_error):
    """Apply errors to position-velocity-attitude.

    Parameters
    ----------
    pva : Pva
        Position-velocity-attitude.
    pva_error : PvaError
        Errors of position-velocity-attitude.

    Returns
    -------
    Pva
        Position-velocity-attitude with applied errors.
    """
    result = pva.copy()
    result[LLA_COLS] = transform.perturb_lla(result[LLA_COLS], pva_error[NED_COLS])
    result[VEL_COLS] += pva_error[VEL_COLS]
    result[RPH_COLS] += pva_error[RPH_COLS]
    return result

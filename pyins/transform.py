"""Coordinate and attitude transformations.

Constants
----------
.. autosummary::
    :toctree: generated

    DEG_TO_RAD
    RAD_TO_DEG
    DH_TO_RS
    RS_TO_DH
    DRH_TO_RRS

Functions
---------
.. autosummary::
    :toctree: generated

    lla_to_ecef
    lla_to_ned
    perturb_lla
    translate_trajectory
    compute_lla_difference
    resample_state
    compute_state_difference
    smooth_rotations
    smooth_state
    mat_en_from_ll
    mat_from_rph
    mat_to_rph
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy import signal
from scipy.spatial.transform import Rotation, Slerp
from .util import LLA_COLS, VEL_COLS, RPH_COLS, NED_COLS, RATE_COLS
from . import earth, util

#: Degrees to radians.
DEG_TO_RAD = np.pi / 180
#: Radians to degrees.
RAD_TO_DEG = 1 / DEG_TO_RAD
#: Degrees per hour to radians per second.
DH_TO_RS = DEG_TO_RAD / 3600
#: Radians per second to degrees per hour.
RS_TO_DH = 1 / DH_TO_RS
#: Degrees per root-hour to radians per root-second.
DRH_TO_RRS = DEG_TO_RAD / 60


def lla_to_ecef(lla):
    """Convert latitude, longitude, altitude into ECEF Cartesian coordinates.

    Parameters
    ----------
    lla : array_like, shape (3,) or (n, 3)
        Latitude, longitude and altitude values.

    Returns
    -------
    r_e : ndarray, shape (3,) or (n, 3)
        Cartesian coordinates in ECEF frame.
    """
    lat, lon, alt = np.asarray(lla).T

    sin_lat = np.sin(np.deg2rad(lat))
    cos_lat = np.cos(np.deg2rad(lat))
    sin_lon = np.sin(np.deg2rad(lon))
    cos_lon = np.cos(np.deg2rad(lon))

    _, re, _ = earth.principal_radii(lat, 0)
    r_e = np.empty((3,) + lat.shape)
    r_e[0] = (re + alt) * cos_lat * cos_lon
    r_e[1] = (re + alt) * cos_lat * sin_lon
    r_e[2] = ((1 - earth.E2) * re + alt) * sin_lat

    return r_e.transpose()


def lla_to_ned(lla, lla_origin=None):
    """Convert lla into NED Cartesian coordinates.

    Parameters
    ----------
    lla : array_like, shape (n, 3)
        Latitude, longitude and altitude values. If DataFrame (with columns 'lat',
        'lon', 'alt) the result will be DataFrame with columns 'north', 'east', 'down'.
    lla_origin : array_like with shape (3,) or None, optional
        Values of latitude, longitude and latitude of the origin point.
        If None (default), the first row in `lla` will be used.

    Returns
    -------
    ndarray of DataFrame
        NED coordinates.
    """
    is_dataframe = isinstance(lla, pd.DataFrame)
    if is_dataframe:
        time = lla.index
        lla = lla[LLA_COLS].values
    else:
        lla = np.asarray(lla)
    if lla_origin is None:
        lla_origin = lla[0]
    r_e = lla_to_ecef(lla) - lla_to_ecef(lla_origin)
    mat_en = mat_en_from_ll(lla_origin[0], lla_origin[1])
    r_n = util.mv_prod(mat_en, r_e, True)
    return pd.DataFrame(r_n, index=time, columns=NED_COLS) if is_dataframe else r_n


def perturb_lla(lla, dr_n):
    """Perturb latitude, longitude and altitude.

    This function recomputes linear displacements in meters to changes in a
    latitude and longitude considering Earth curvature.

    Note that this computation is approximate in nature and makes a good
    sense only if displacements are significantly less than Earth radius.

    Parameters
    ----------
    lla : array_like, shape (3,) or (n, 3)
        Latitude, longitude and altitude.
    dr_n : array_like, shape (3,) or (n, 3)
        Perturbation values in meters resolved in NED frame.

    Returns
    -------
    lla_new : ndarray, shape (3,) or (n, 3)
        Perturbed values of latitude, longitude and altitude.
    """
    lla = np.asarray(lla, dtype=float)
    dr_n = np.asarray(dr_n)
    return_single = lla.ndim == 1 and dr_n.ndim == 1

    lla = np.atleast_2d(lla).copy()
    dr_n = np.atleast_2d(dr_n)

    rn, _, rp = earth.principal_radii(lla[:, 0], lla[:, 2])

    lla[:, 0] += np.rad2deg(dr_n[:, 0] / rn)
    lla[:, 1] += np.rad2deg(dr_n[:, 1] / rp)
    lla[:, 2] -= dr_n[:, 2]

    return lla[0] if return_single else lla


def translate_trajectory(trajectory, translation_b):
    """Translate trajectory by a vector expressed in body frame.

    Parameters
    ----------
    trajectory : Trajectory or Pva
        Either trajectory or position-velocity-attitude.
        If has columns 'rate_x', 'rate_y', 'rate_z', velocity will be adjusted by
        rotation effect.
    translation_b : array_like, shape (3,)
        Translation vector expressed in body frame.

    Returns
    -------
    Trajectory or Pva
        Translated trajectory or position-velocity-attitude.
    """
    mat_nb = mat_from_rph(trajectory[RPH_COLS])
    result = trajectory.copy()
    result[LLA_COLS] = perturb_lla(result[LLA_COLS],
                                   util.mv_prod(mat_nb, translation_b))
    if all(col in trajectory for col in RATE_COLS):
        result[VEL_COLS] += util.mv_prod(mat_nb,
            np.cross(trajectory[RATE_COLS], translation_b))
    return result


def compute_lla_difference(lla1, lla2):
    """Compute difference between lla points resolved in NED in meters.

    Parameters
    ----------
    lla1, lla2 : array_like
        Points with latitude, longitude and altitude.

    Returns
    -------
    dr_n : ndarray
        Difference in meters resolved in NED.
    """
    lla1 = np.asarray(lla1)
    lla2 = np.asarray(lla2)
    single = lla1.ndim == 1 and lla2.ndim == 1
    lla1 = np.atleast_2d(lla1)
    lla2 = np.atleast_2d(lla2)
    rn, _, rp = earth.principal_radii(0.5 * (lla1[:, 0] + lla2[:, 0]),
                                      0.5 * (lla1[:, 2] + lla2[:, 2]))
    diff = lla1 - lla2
    result = np.empty_like(diff)
    result[:, 0] = np.deg2rad(diff[:, 0]) * rn
    result[:, 1] = np.deg2rad(diff[:, 1]) * rp
    result[:, 2] = -diff[:, 2]
    return result[0] if single else result


def _has_rph(data):
    return all(col in data for col in RPH_COLS)


def resample_state(state, times):
    """Compute state values at new set of time values.

    Piecewise linear interpolation is used with special care for rotation
    ('roll', 'pitch', 'heading' columns), for which SLERP is used.

    Parameters
    ----------
    state : DataFrame
        State data indexed by time.
    times : array_like
        Values of time at which compute new state values. Values outside the original
        time span of `state` will be discarded.

    Returns
    -------
    DataFrame
        Resampled state values.
    """
    times = np.sort(times)
    times = times[(times >= state.index[0]) & (times <= state.index[-1])]

    result = pd.DataFrame(index=times)
    if _has_rph(state):
        slerp = Slerp(state.index, Rotation.from_euler('xyz', state[RPH_COLS], True))
        result[RPH_COLS] = slerp(times).as_euler('xyz', True)

    other_columns = state.columns.difference(RPH_COLS)
    interpolator = interp1d(state.index, state[other_columns].values, axis=0)
    result[other_columns] = interpolator(times)
    return result[state.columns]


def compute_state_difference(first, second):
    """Compute difference between two state data frames indexed by time.

    If both inputs are DataFrame, the function synchronizes data to the common time
    index using `resample_state`. The interpolation is done for the dataframe with more
    frequent data (to reduce average interpolation period).

    For columns 'lat', 'lon', 'alt', the difference is computed in meters
    resolved in NED frame. For columns 'roll', 'pitch' and 'heading' the interpolation
    is done in the rotation space using SLERP algorithm.

    Parameters
    ----------
    first, second : DataFrame or Series
        State data to compute the difference between.

    Returns
    -------
    DataFrame
        Computed difference.
    """
    def has_lla(data):
        return all(col in data for col in LLA_COLS)

    if isinstance(first, pd.DataFrame) and isinstance(second, pd.DataFrame):
        if np.median(np.diff(first.index)) < np.median(np.diff(second.index)):
            result_sign = -1.0
            first, second = second, first
        else:
            result_sign = 1.0

        index = first.index
        index = index[(index >= second.index[0]) & (index <= second.index[-1])]
        columns = first.columns.intersection(second.columns)

        first = first.loc[index, columns]
        second = resample_state(second[columns], index)
    elif isinstance(first, pd.Series) and isinstance(second, pd.Series):
        result_sign = 1.0
    else:
        raise ValueError("Both inputs must be either DataFrame or Series")

    difference = first - second
    if has_lla(difference):
        rn, _, rp = earth.principal_radii(0.5 * (first.lat + second.lat),
                                          0.5 * (first.alt + second.alt))
        difference.lat *= rn * DEG_TO_RAD
        difference.lon *= rp * DEG_TO_RAD
        difference.alt *= -1
        difference.rename({'lat': 'north', 'lon': 'east', 'alt': 'down'},
                          axis=1 if isinstance(difference, pd.DataFrame) else 0,
                          inplace=True)

    if _has_rph(difference):
        difference[RPH_COLS] = util.to_180_range(difference[RPH_COLS])

    return result_sign * difference


def _apply_smoothing(data, T, T_smooth):
    num_taps = 2 * round(T_smooth / T) + 1
    h = signal.firwin(num_taps, 1 / T_smooth, fs=1 / T)
    return signal.lfilter(h, 1, data, axis=0), num_taps


def smooth_rotations(rotations, dt, smoothing_time):
    """Smooth rotations.

    The function applies a FIR filter to the elements of outer products of quaternion
    representation of the rotations. The smoothed quaternion is constructed as the
    eigenvector of the matrix with the smoothed coefficients.

    A FIR filter is constructed using `scipy.signal.firwin` with the number of
    coefficients selected as::

        num_taps = 2 * round(smoothing_time / dt) + 1

    First ``num_taps - 1`` samples will be influenced by the edge effect.
    The filtered rotations are delayed by ``num_taps // 2`` samples.

    Parameters
    ----------
    rotations : `scipy.spatial.transform.Rotation`
        Rotation objects with multiple rotations. All rotations are supposed to be
        equispaced in time.
    dt : float
        Time interval between rotations.
    smoothing_time : float
        Smoothing time.

    Returns
    -------
    `scipy.spatial.transform.Rotation`
        Smoothed rotations.
    num_taps : int
        Length of the filter window.
    """
    if rotations.single:
        raise ValueError("`rotations` must contain multiple rotations.")
    q = rotations.as_quat()
    coefficients = np.einsum('...i,...j->...ij', q, q).reshape(-1, 16)
    coefficients, num_taps = _apply_smoothing(coefficients, dt, smoothing_time)
    _, v = np.linalg.eigh(coefficients.reshape(-1, 4, 4))
    return Rotation.from_quat(v[:, :, -1]), num_taps


def smooth_state(state, smoothing_time):
    """Smooth state data.

    Smoothing is done in 3 steps: first the state is resampled to the constant time
    step using `resample_state`, then the data is smoothed with FIR filter and then the
    smoothed data is resampled back to the original time.

    The smoothing is done by applying FIR filter constructed using
    `scipy.signal.firwin` with the number of coefficients computed as::

        num_taps = 2 * round(smoothing_time / dt) + 1

    The first ``num_taps - 1`` filtered samples are removed as they cannot be computed
    using actual data and the edge effects are hard to eliminate. The group delay is
    compensated by appropriately adjusting time index.

    Rotations ('roll', 'pitch', 'heading' columns) are smoothed by `smooth_rotations`.

    Parameters
    ----------
    state : DataFrame
        State date indexed by time.
    smoothing_time : float
        Smoothing time.

    Returns
    -------
    DataFrame
        Smoothed data.
    """
    dt = np.min(np.diff(state.index))
    resampled_state = resample_state(state,
                                     np.arange(state.index[0], state.index[-1], dt))
    smoothed = pd.DataFrame(index=resampled_state.index)
    if _has_rph(state):
        rotations = Rotation.from_euler('xyz', resampled_state[RPH_COLS], True)
        smoothed[RPH_COLS] = smooth_rotations(
            rotations, dt, smoothing_time)[0].as_euler('xyz', True)
    other_columns = state.columns.difference(RPH_COLS)
    smoothed[other_columns], num_taps = _apply_smoothing(
        resampled_state[other_columns].values, dt, smoothing_time)
    smoothed = smoothed.iloc[num_taps - 1:]
    smoothed.index -= dt * (num_taps // 2)
    return resample_state(smoothed[state.columns], state.index)


def mat_en_from_ll(lat, lon):
    """Create a rotation matrix projecting from ECEF to NED frame.

    Parameters
    ----------
    lat, lon : float or array_like, shape (n,)
        Latitude and longitude.

    Returns
    -------
    ndarray, shape (3, 3) or (n, 3, 3)
        Rotation matrices.
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)

    if lat.ndim == 0 and lon.ndim == 0:
        return Rotation.from_euler('ZY', [lon, -90 - lat], degrees=True).as_matrix()

    lat = np.atleast_1d(lat)
    lon = np.atleast_1d(lon)

    n = max(len(lat), len(lon))
    angles = np.empty((n, 2))
    angles[:, 0] = lon
    angles[:, 1] = -90 - lat
    return Rotation.from_euler('ZY', angles, degrees=True).as_matrix()


def mat_from_rph(rph):
    """Create a rotation matrix from roll, pitch and heading.

    Parameters
    ----------
    rph : array_like, shape (3,) or (n, 3)
        Heading, pitch and roll.

    Returns
    -------
    ndarray, shape (3, 3) or (n, 3, 3)
        Rotation matrices.
    """
    return Rotation.from_euler('xyz', rph, degrees=True).as_matrix()


def mat_to_rph(mat):
    """Convert a rotation matrix to roll, pitch and heading angles.

    Parameters
    ----------
    mat : array_like, shape (3, 3) or (n, 3, 3)
        Rotation matrices.

    Returns
    -------
    ndarray, with shape (3,) or (n, 3)
        Roll, pitch and heading angles.
    """
    return Rotation.from_matrix(mat).as_euler('xyz', degrees=True)

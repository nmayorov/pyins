"""Coordinate transformations."""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from .util import LLA_COLS, VEL_COLS, RPH_COLS, NED_COLS, TRAJECTORY_COLS
from . import earth

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
    """Convert latitude, longitude, altitude to ECEF Cartesian coordinates.

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


def perturb_lla(lla, delta_ned):
    """Perturb latitude, longitude and altitude.

    This function recomputes linear displacements in meters to changes in a
    latitude and longitude considering Earth curvature.

    Note that this computation is approximate in nature and makes a good
    sense only if displacements are significantly less than Earth radius.

    Parameters
    ----------
    lla : array_like, shape (3,) or (n, 3)
        Latitude, longitude and altitude.
    delta_ned : array_like, shape (3,) or (n, 3)
        Perturbation values in meters resolved in NED frame.

    Returns
    -------
    lla_new : ndarray, shape (3,) or (n, 3)
        Perturbed values of latitude, longitude and altitude.
    """
    lla = np.asarray(lla, dtype=float)
    delta_ned = np.asarray(delta_ned)
    return_single = lla.ndim == 1 and delta_ned.ndim == 1

    lla = np.atleast_2d(lla).copy()
    delta_ned = np.atleast_2d(delta_ned)

    rn, _, rp = earth.principal_radii(lla[:, 0], lla[:, 2])

    lla[:, 0] += np.rad2deg(delta_ned[:, 0] / rn)
    lla[:, 1] += np.rad2deg(delta_ned[:, 1] / rp)
    lla[:, 2] -= delta_ned[:, 2]

    return lla[0] if return_single else lla


def difference_lla(lla1, lla2):
    """Compute difference between lla points resolved in NED in meters.

    Parameters
    ----------
    lla1, lla2 : array_like
        Points with latitude, longitude and altitude.

    Returns
    -------
    difference_ned : ndarray
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


def to_180_degrees_range(angle):
    angle = angle % 360.0
    if isinstance(angle, (pd.Series, pd.DataFrame)):
        angle = angle.copy()
    else:
        angle = np.asarray(angle)
    angle[angle < -180.0] += 360.0
    angle[angle > 180.0] -= 360.0
    return angle


def compute_state_difference(first, second):
    def has_lla(data):
        return all(col in data for col in LLA_COLS)

    def has_velocity(data):
        return all(col in data for col in VEL_COLS)

    def has_rph(data):
        return all(col in data for col in RPH_COLS)

    if np.median(np.diff(first.index)) < np.median(np.diff(second.index)):
        result_sign = -1.0
        first, second = second, first
    else:
        result_sign = 1.0

    first = first[~first.index.duplicated()]

    index = first.index
    index = index[(index >= second.index[0]) & (index <= second.index[-1])]

    results = []

    if has_lla(first) and has_lla(second):
        lla_second_interpolator = interp1d(second.index, second[LLA_COLS], axis=0)
        lla_second = pd.DataFrame(lla_second_interpolator(index),
                                  index=index, columns=LLA_COLS)

        rn, _, rp = earth.principal_radii(0.5 * (first.lat + lla_second.lat),
                                          0.5 * (first.alt + lla_second.alt))
        difference = pd.DataFrame(index=index, columns=NED_COLS)
        difference['north'] = (first.lat - lla_second.lat) * rn * DEG_TO_RAD
        difference['east'] = (first.lon - lla_second.lon) * rp * DEG_TO_RAD
        difference['down'] = -(first.alt - lla_second.alt)
        results.append(difference)

    if has_velocity(first) and has_velocity(second):
        velocity_second_interpolator = interp1d(second.index, second[VEL_COLS], axis=0)
        velocity_second_at_first = velocity_second_interpolator(index)
        difference = pd.DataFrame(
            first.loc[index, VEL_COLS].values - velocity_second_at_first,
            index=index, columns=VEL_COLS)
        results.append(difference)

    if has_rph(first) and has_rph(second):
        rpy_second_interpolator = Slerp(
            second.index, Rotation.from_euler('xyz', second[RPH_COLS], True))
        rpy_second_at_first = rpy_second_interpolator(index).as_euler('xyz', True)
        difference = pd.DataFrame(
            to_180_degrees_range(first.loc[index, RPH_COLS].values -
                                 rpy_second_at_first), index=index, columns=RPH_COLS)
        results.append(difference)
    else:
        if 'heading' in first and 'heading' in second:
            heading_second_interpolator = Slerp(
                second.index, Rotation.from_euler('z', second.heading, True))
            heading_second_at_first = heading_second_interpolator(index).as_euler(
                'xyz', True)[:, 2]
            difference = pd.DataFrame(
                to_180_degrees_range(first.loc[index, 'heading'].values -
                                     heading_second_at_first),
                index=index, columns=['heading'])
            results.append(difference)
        if 'pitch' in first and 'pitch' in second:
            pitch_second_interpolator = Slerp(
                second.index, Rotation.from_euler('y', second.pitch, True))
            pitch_second_at_first = pitch_second_interpolator(index).as_euler(
                'xyz', True)[:, 1]
            difference = pd.DataFrame(first.loc[index, 'pitch'].values -
                                      pitch_second_at_first,
                                      index=index, columns=['pitch'])
            results.append(difference)

    other_columns = first.columns.intersection(second.columns).difference(
        TRAJECTORY_COLS)
    if not other_columns.empty:
        second_interpolator = interp1d(second.index, second[other_columns], axis=0)
        difference = pd.DataFrame(
            first.loc[index, other_columns].values - second_interpolator(index),
            index=index, columns=other_columns)
        results.append(difference)

    return (result_sign * pd.concat(results, axis=1) if results
            else pd.DataFrame(index=index))


def correct_trajectory(trajectory, error):
    """Correct trajectory by estimated errors.

    Note that it means subtracting errors from the trajectory.

    Parameters
    ----------
    trajectory : DataFrame
        Trajectory.
    error : DataFrame
        Estimated errors.

    Returns
    -------
    traj_corr : DataFrame
        Corrected trajectory.
    """
    rn, _, rp = earth.principal_radii(trajectory.lat, trajectory.alt)

    result = trajectory.copy()
    result['lat'] -= np.rad2deg(error.north / rn)
    result['lon'] -= np.rad2deg(error.east / rp)
    result['alt'] += error.down
    result[VEL_COLS] -= error[VEL_COLS]
    result[RPH_COLS] -= error[RPH_COLS]
    return result.dropna()


def phi_to_delta_rph(rph):
    """Compute transformation matrix relating small phi angle and rph error.

    This function computes matrix `T` which relates perturbation of roll,
    pitch and heading angles to the perturbation of the matrix
    `exp(-phi) @ C` as::

        delta_rph = T @ phi

    Parameters
    ----------
    rph : array_like, shape (3,) or (n, 3)
        Roll, pitch and heading.

    Returns
    -------
    ndarray, shape (3, 3) or (n, 3, 3)
        Transformation matrix.
    """

    rph = np.asarray(rph)
    single = rph.ndim == 1
    rph = np.atleast_2d(rph)
    result = np.zeros((len(rph), 3, 3))

    sin = np.sin(np.deg2rad(rph))
    cos = np.cos(np.deg2rad(rph))

    result[:, 0, 0] = -cos[:, 2] / cos[:, 1]
    result[:, 0, 1] = -sin[:, 2] / cos[:, 1]
    result[:, 1, 0] = sin[:, 2]
    result[:, 1, 1] = -cos[:, 2]
    result[:, 2, 0] = -cos[:, 2] * sin[:, 1] / cos[:, 1]
    result[:, 2, 1] = -sin[:, 2] * sin[:, 1] / cos[:, 1]
    result[:, 2, 2] = -1

    return result[0] if single else result


def mat_en_from_ll(lat, lon):
    """Create a direction cosine from ECEF to NED frame.

    The sequence of elemental rotations is as follows::

              lon      -90 - lat
        E ----------> ----------> N
               3           2

    Here E denotes the ECEF frame and N denotes the local level
    north-pointing frame. The resulting DCM projects from N frame to E frame.

    Parameters
    ----------
    lat, lon : float or array_like with shape (n,)
        Latitude and longitude.

    Returns
    -------
    dcm : ndarray, shape (3, 3) or (n, 3, 3)
        Direction Cosine Matrices.
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)

    if lat.ndim == 0 and lon.ndim == 0:
        return Rotation.from_euler('ZY', [lon, -90 - lat],
                                   degrees=True).as_matrix()

    lat = np.atleast_1d(lat)
    lon = np.atleast_1d(lon)

    n = max(len(lat), len(lon))
    angles = np.empty((n, 2))
    angles[:, 0] = lon
    angles[:, 1] = -90 - lat
    return Rotation.from_euler('ZY', angles, degrees=True).as_matrix()


def mat_from_rph(rph):
    """Create a rotation matrix from roll, pitch and heading.

    The sequence of elemental rotations is as follows::

           heading    pitch   roll
        N ---------> ------> -----> B
              3         2       1

    The resulting matrix projects from B frame to N frame.

    Parameters
    ----------
    rph : array_like, shape (3,) or (n, 3)
        Heading, pitch and roll.

    Returns
    -------
    dcm : ndarray, shape (3, 3) or (n, 3, 3)
        Direction cosine matrices.
    """
    return Rotation.from_euler('xyz', rph, degrees=True).as_matrix()


def mat_to_rph(mat):
    """Convert a rotation matrix to roll, pitch, heading angles.

    The returned heading is within [0, 360], the pitch is within [-90, 90]
    and the roll is within [-90, 90].

    Parameters
    ----------
    mat : array_like, shape (3, 3) or (n, 3, 3)
        Direction cosine matrices.

    Returns
    -------
    rph : ndarray, with shape (3,) or (n, 3)
        Heading, pitch and roll.
    """
    return Rotation.from_matrix(mat).as_euler('xyz', degrees=True)

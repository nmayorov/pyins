"""Coordinate transformations."""
import numpy as np
from . import earth


DEG_TO_RAD = np.pi / 180
RAD_TO_DEG = 1 / DEG_TO_RAD
DH_TO_RS = DEG_TO_RAD / 3600
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

    re, _, _ = earth.principal_radii(lat, 0)
    r_e = np.empty((3,) + lat.shape)
    r_e[0] = (re + alt) * cos_lat * cos_lon
    r_e[1] = (re + alt) * cos_lat * sin_lon
    r_e[2] = ((1 - earth.E2) * re + alt) * sin_lat

    return r_e.transpose()


def perturb_lla(lla, delta_enu):
    """Perturb latitude, longitude and altitude.

    This function recomputes linear displacements in meters to changes in a
    latitude and longitude considering Earth curvature.

    Note that this computation is approximate in nature and makes a good
    sense only if displacements are significantly less than Earth radius.

    Parameters
    ----------
    lla : array_like, shape (3,) or (n, 3)
        Latitude, longitude and altitude.
    delta_enu : array_like, shape (3,) or (n, 3)
        Perturbation values in meters resolved in ENU frame.

    Returns
    -------
    lla_new : ndarray, shape (3,) or (n, 3)
        Perturbed values of latitude, longitude and altitude.
    """
    lla = np.asarray(lla, dtype=float)
    delta_enu = np.asarray(delta_enu)
    return_single = lla.ndim == 1 and delta_enu.ndim == 1

    lla = np.atleast_2d(lla).copy()
    delta_enu = np.atleast_2d(delta_enu)

    _, rn, rp = earth.principal_radii(lla[:, 0], lla[:, 2])

    lla[:, 0] += np.rad2deg(delta_enu[:, 0] / rn)
    lla[:, 1] += np.rad2deg(delta_enu[:, 1] / rp)
    lla[:, 2] -= delta_enu[:, 2]

    return lla[0] if return_single else lla


def difference_lla(lla1, lla2):
    """Compute difference between lla points resolved in ENU in meters.

    Parameters
    ----------
    lla1, lla2 : array_like
        Points with latitude, longitude and altitude.

    Returns
    -------
    difference_enu : ndarray
        Difference in meters resolved in ENU.
    """
    lla1 = np.asarray(lla1)
    lla2 = np.asarray(lla2)
    single = lla1.ndim == 1 and lla2.ndim == 1
    lla1 = np.atleast_2d(lla1)
    lla2 = np.atleast_2d(lla2)
    _, rn, rp = earth.principal_radii(0.5 * (lla1[:, 0] + lla2[:, 0]),
                                      0.5 * (lla1[:, 2] + lla2[:, 2]))
    diff = lla1 - lla2
    result = np.empty_like(diff)
    result[:, 0] = np.deg2rad(diff[:, 0]) * rn
    result[:, 1] = np.deg2rad(diff[:, 1]) * rp
    result[:, 2] = -diff[:, 2]
    return result[0] if single else result


def difference_trajectories(t1, t2):
    """Compute trajectory difference.

    Parameters
    ----------
    t1, t2 : DataFrame
        Trajectories.

    Returns
    -------
    diff : DataFrame
        Trajectory difference. It can be interpreted as errors in `t1` relative
        to `t2`.
    """
    TRAJECTORY_ERROR_COLUMNS = ['north', 'east', 'down', 'VN', 'VE', 'VD',
                                'roll', 'pitch', 'heading']

    diff = t1 - t2
    _, rn, rp = earth.principal_radii(0.5 * (t1.lat + t2.lat),
                                      0.5 * (t1.alt + t2.alt))
    diff.lat *= np.deg2rad(rn)
    diff.lon *= np.deg2rad(rp)
    diff.alt = -diff.alt
    diff.heading %= 360
    diff.heading[diff.heading < -180] += 360
    diff.heading[diff.heading > 180] -= 360

    diff = diff.rename(columns={'lat': 'north', 'lon': 'east', 'alt': 'down'})

    other_columns = [column for column in diff.columns
                     if column not in TRAJECTORY_ERROR_COLUMNS]
    diff = diff[TRAJECTORY_ERROR_COLUMNS + other_columns]

    return diff.loc[t1.index.intersection(t2.index)]


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
    _, rn, rp = earth.principal_radii(trajectory.lat, trajectory.alt)

    result = trajectory.copy()
    result['lat'] -= np.rad2deg(error.north / rn)
    result['lon'] -= np.rad2deg(error.east / rp)
    result['alt'] += error.down
    result[['VN', 'VE', 'VD']] -= error[['VN', 'VE', 'VD']]
    result[['roll', 'pitch', 'heading']] -= error[['roll', 'pitch', 'heading']]
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

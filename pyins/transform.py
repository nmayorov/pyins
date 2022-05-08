"""Coordinate transformations."""
import numpy as np
from . import earth


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

    lla[:, 0] += np.rad2deg(delta_enu[:, 1] / rn)
    lla[:, 1] += np.rad2deg(delta_enu[:, 0] / rp)
    lla[:, 2] += delta_enu[:, 2]

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
    result[:, 0] = np.deg2rad(diff[:, 1]) * rp
    result[:, 1] = np.deg2rad(diff[:, 0]) * rn
    result[:, 2] = diff[:, 2]
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
    diff = t1 - t2
    _, rn, rp = earth.principal_radii(0.5 * (t1.lat + t2.lat),
                                      0.5 * (t1.alt + t2.alt))
    diff.lat *= np.deg2rad(rn)
    diff.lon *= np.deg2rad(rp)
    diff.heading %= 360
    diff.heading[diff.heading < -180] += 360
    diff.heading[diff.heading > 180] -= 360

    return diff.loc[t1.index.intersection(t2.index)]


def correct_traj(traj, error):
    """Correct trajectory by estimated errors.

    Note that it means subtracting errors from the trajectory.

    Parameters
    ----------
    traj : DataFrame
        Trajectory.
    error : DataFrame
        Estimated errors.

    Returns
    -------
    traj_corr : DataFrame
        Corrected trajectory.
    """
    traj_corr = traj.copy()
    traj_corr['lat'] -= np.rad2deg(error.lat / earth.R0)
    traj_corr['lon'] -= np.rad2deg(error.lon / (earth.R0 *
                                   np.cos(np.deg2rad(traj_corr['lat']))))
    traj_corr['alt'] -= error.alt
    traj_corr['VE'] -= error.VE
    traj_corr['VN'] -= error.VN
    traj_corr['VU'] -= error.VU
    traj_corr['roll'] -= error.roll
    traj_corr['pitch'] -= error.pitch
    traj_corr['heading'] -= error.heading

    return traj_corr.dropna()

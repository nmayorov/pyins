"""Earth geometry and gravity models.

This module defines constants and computation models for ellipsoidal Earth using
WGS84 parameters. All definitions and explanations for used models can be found in [1]_.


Constants
---------
.. autosummary::
    :toctree: generated

    RATE
    A
    E2
    GE
    GP

Functions
---------
.. autosummary::
    :toctree: generated/

    principal_radii
    gravity
    gravity_n
    gravitation_ecef
    curvature_matrix
    rate_n

References
----------
.. [1] P. D. Groves, "Principles of GNSS, Inertial, and Multisensor Integrated
       Navigation Systems", 2nd edition
"""
import numpy as np
from . import transform, util


#: Rotation rate of Earth in rad/s.
RATE = 7.292115e-5
#: Semi major axis of Earth ellipsoid.
A = 6378137.0
#: Squared eccentricity of Earth ellipsoid
E2 = 6.6943799901413e-3
#: Gravity at the equator.
GE = 9.7803253359
#: Gravity at the pole.
GP = 9.8321849378
F = (1 - E2) ** 0.5 * GP / GE - 1


def principal_radii(lat, alt):
    """Compute the principal radii of curvature of Earth ellipsoid.

    Parameters
    ----------
    lat, alt : array_like
        Latitude and altitude.

    Returns
    -------
    rn : float or ndarray
        Principle radius in North direction.
    re : float or ndarray
        Principle radius in East direction.
    rp : float or ndarray
        Radius of cross-section along the parallel.
    """
    sin_lat = np.sin(np.deg2rad(lat))
    cos_lat = np.sqrt(1 - sin_lat**2)

    x = 1 - E2 * sin_lat ** 2
    re = A / np.sqrt(x)
    rn = re * (1 - E2) / x

    return rn + alt, re + alt, (re + alt) * cos_lat


def gravity(lat, alt):
    """Compute gravity magnitude.

    Somigliana model used in WGS84 with linear vertical correction is implemented.

    Parameters
    ----------
    lat, alt : array_like
        Latitude and altitude.

    Returns
    -------
    gravity : float or ndarray
        Magnitude of the gravity.
    """
    sin_lat = np.sin(np.deg2rad(lat))
    alt = np.asarray(alt)
    return (GE * (1 + F * sin_lat**2) / (1 - E2 * sin_lat**2) ** 0.5
            * (1 - 2 * alt / A))


def gravity_n(lat, alt):
    """Compute gravity vector in NED frame.

    Parameters
    ----------
    lat, alt : array_like
        Latitude and altitude.

    Returns
    -------
    g_n : ndarray, shape (3,) or (n, 3)
        Vector of the gravity.
    """
    g = gravity(lat, alt)
    if g.ndim == 0:
        return np.array([0, 0, g])
    else:
        result = np.zeros((len(g), 3))
        result[:, 2] = g
        return result


def gravitation_ecef(lla):
    """Compute a vector of the gravitational force in ECEF frame.

    It accounts only for Earth mass attraction by eliminating the centrifugal force
    from `gravity` model.

    Parameters
    ----------
    lla : array_like, shape (3,) or (n, 3)
        Latitude, longitude and altitude.

    Returns
    -------
    g0_e: ndarray, shape (3,) or (n, 3)
        Vectors of the gravitational force expressed in ECEF frame.
    """
    lat, lon, alt = np.asarray(lla).T

    sin_lat = np.sin(np.deg2rad(lat))
    cos_lat = np.cos(np.deg2rad(lat))

    _, _, rp = principal_radii(lat, alt)

    g0_g = np.zeros((3,) + lat.shape)
    g0_g[0] = RATE**2 * rp * sin_lat
    g0_g[2] = gravity(lat, alt) + RATE ** 2 * rp * cos_lat
    g0_g = g0_g.T

    mat_eg = transform.mat_en_from_ll(lat, lon)

    return util.mv_prod(mat_eg, g0_g)


def curvature_matrix(lat, alt):
    """Compute Earth curvature matrix.

    Curvature matrix ``F`` links linear displacement and angular rotation of
    NED frame as::

        rotation_n = F @ translation_n

    Where ``translation_n`` is linear translation in NED and ``rotation_n`` is the
    corresponding small rotation vector of NED.

    For example ``transport_rate_n = F @ velocity_n``.

    Parameters
    ----------
    lat, alt : array_like
        Latitude and altitude.

    Returns
    -------
    F: ndarray, shape (3, 3) or (n, 3, 3)
        Curvature matrix.
    """
    rn, re, _ = principal_radii(lat, alt)
    n = 1 if re.ndim == 0 else len(re)

    result = np.zeros((n, 3, 3))
    result[:, 0, 1] = 1 / re
    result[:, 1, 0] = -1 / rn
    result[:, 2, 1] = -result[:, 0, 1] * np.tan(np.deg2rad(lat))

    return result[0] if re.ndim == 0 else result


def rate_n(lat):
    """Compute Earth rate resolved in NED frame.

    Parameters
    ----------
    lat : array_like
        Latitude.

    Returns
    -------
    earth_rate_n : ndarray, shape (3,) or (n, 3)
        NED components of Earth rate.
    """
    lat = np.asarray(lat)
    n = 1 if lat.ndim == 0 else len(lat)
    result = np.zeros((n, 3))
    result[:, 0] = RATE * np.cos(np.deg2rad(lat))
    result[:, 2] = -RATE * np.sin(np.deg2rad(lat))
    return result[0] if lat.ndim == 0 else result

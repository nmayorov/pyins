"""Earth geometry and gravity models."""
import numpy as np
from . import transform
from . util import mv_prod


#: Rotation rate of Earth in rad/s.
RATE = 7.2921157e-5
#: Approximate value of gravity.
G0 = 9.8
#: Semi major axis of Earth ellipsoid (or radius of Earth approximately).
R0 = 6378137.0
#: Squared eccentricity of Earth ellipsoid
E2 = 6.6943799901413e-3
#: Gravity at the equator
GE = 9.7803253359
#: Gravity at the pole.
GP = 9.8321849378


def principal_radii(lat, alt):
    """Compute the principal radii of curvature of Earth ellipsoid.

    See [1]_ (p. 41) for the definition and formulas.

    Parameters
    ----------
    lat : array_like
        Latitude.
    alt : array_like
        Altitude.

    Returns
    -------
    re : ndarray
        Principle radius in East direction.
    rn : ndarray
        Principle radius in North direction.
    rp : ndarray
        Radius of cross-section along the parallel.

    References
    ----------
    .. [1] P. D. Groves, "Principles of GNSS, Inertial, and Multisensor
           Integrated Navigation Systems".
    """
    sin_lat = np.sin(np.deg2rad(lat))
    cos_lat = np.sqrt(1 - sin_lat**2)

    x = 1 - E2 * sin_lat ** 2
    re = R0 / np.sqrt(x)
    rn = re * (1 - E2) / x

    return re + alt, rn + alt, (re + alt) * cos_lat


def gravity(lat, alt=0):
    """Compute gravity according to a theoretical model.

    See `set_model` for the explanation of the formula used.

    Parameters
    ----------
    lat : array_like
        Latitude.
    alt : array_like, optional
        Altitude. Default is 0.

    Returns
    -------
    g : ndarray
        Magnitude of the gravity.
    """
    sin_lat = np.sin(np.deg2rad(lat))
    alt = np.asarray(alt)
    F = (1 - E2) ** 0.5 * GP / GE - 1
    return (GE * (1 + F * sin_lat**2) / (1 - E2 * sin_lat**2)**0.5
            * (1 - 2 * alt / R0))


def gravity_n(lat, alt):
    """Compute gravity vector in NED frame.

    Parameters
    ----------
    lat : array_like
        Latitude.
    alt : array_like
        Altitude.

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

    It accounts only for Earth mass attraction. It is computed from
    `gravity` model by eliminating the centrifugal force.

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

    g0_g = np.zeros((3,) + sin_lat.shape)
    g0_g[0] = RATE**2 * rp * sin_lat
    g0_g[2] = gravity(lat, alt) + RATE ** 2 * rp * cos_lat
    g0_g = g0_g.T

    Ceg = transform.mat_en_from_ll(lat, lon)

    return mv_prod(Ceg, g0_g)


def curvature_matrix(lat, alt):
    """Compute Earth curvature matrix.

    Curvature matrix `F` links linear displacement and angular rotation of
    NED frame as `rotation_n = F @ translation_n`, where `translation_n` is
    linear translation in NED and `rotation_n` is the corresponding small
    rotation vector of NED.

    For example `transport_rate_n = F @ velocity_n`.

    Parameters
    ----------
    lat : array_like, shape (n,)
        Latitude.
    alt : array_like, shape (n,)
        Altitude.

    Returns
    -------
    F: ndarray, shape (n, 3, 3)
        Curvature matrix.
    """
    re, rn, _ = principal_radii(lat, alt)
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
    earth_rate_n : ndarray
    """
    lat = np.asarray(lat)
    n = 1 if lat.ndim == 0 else len(lat)
    result = np.zeros((n, 3))
    result[:, 0] = RATE * np.cos(np.deg2rad(lat))
    result[:, 2] = -RATE * np.sin(np.deg2rad(lat))
    return result[0] if lat.ndim == 0 else result

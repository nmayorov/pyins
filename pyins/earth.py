"""Earth geometry and gravity models."""
import numpy as np
from . import dcm
from . util import mv_prod


#: Rotation rate of Earth in rad/s.
RATE = 7.2921157e-5
#: Schuller frequency.
SF = 1.2383e-3
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

F = R0 * (1 - E2) ** 0.5 * GP / (GE * R0) - 1
#: Standard gravitational parameter for Earth
MU = 3.986004418e14


def principal_radii(lat, alt=0):
    """Compute the principal radii of curvature of Earth ellipsoid.

    See [1]_ (p. 41) for the definition and formulas.

    Parameters
    ----------
    lat : array_like
        Latitude.
    alt : array_like
        Altitude. Default is 0.

    Returns
    -------
    re, rn : ndarray
        Radii of curvature in East and North directions respectively.

    References
    ----------
    .. [1] P. D. Groves, "Principles of GNSS, Inertial, and Multisensor
           Integrated Navigation Systems".
    """
    sin_lat = np.sin(np.deg2rad(lat))

    x = 1 - E2 * sin_lat ** 2
    re = R0 / np.sqrt(x)
    rn = re * (1 - E2) / x

    return re + alt, rn + alt


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

    return GE * (1 + F * sin_lat**2) / (1 - E2 * sin_lat**2)**0.5 * (1 - 2 * alt / R0)


def gravitation_ecef(lat, lon, alt=0):
    """Compute a vector of the gravitational force in ECEF frame.

    It accounts only for Earth mass attraction. It is computed from
    `gravity` model by eliminating the centrifugal force.

    Parameters
    ----------
    lat, lon : array_like
        Latitude and longitude.
    alt : array_like, optional
        Altitude. Default is 0.

    Returns
    -------
    g0_e: ndarray, shape (3,) or (n, 3)
        Vectors of the gravitational force expressed in ECEF frame.
    """

    sin_lat = np.sin(np.deg2rad(lat))
    cos_lat = np.cos(np.deg2rad(lat))

    re, _ = principal_radii(lat)
    rp = (re + alt) * cos_lat

    g0_g = np.zeros((3,) + sin_lat.shape)
    g0_g[1] = RATE**2 * rp * sin_lat
    g0_g[2] = -gravity(lat, alt) - RATE ** 2 * rp * cos_lat
    g0_g = g0_g.T

    Ceg = dcm.from_llw(lat, lon)

    return mv_prod(Ceg, g0_g)

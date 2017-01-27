"""Earth geometry and gravity models."""
import numpy as np
from . import dcm
from . util import mv_prod


MODEL = None
#: Rotation rate of Earth in rad/s.
RATE = 7.2921157e-5
#: Schuller frequency.
SF = 1.2383e-3
#: Approximate value of gravity.
G0 = 9.8
#: Semi major axis of Earth ellipsoid (or radius of Earth approximately).
R0 = None
E2 = None
GE = None
F = None


def set_model(model='WGS84'):
    """Set Earth ellipsoid parameters and the gravity model.

    This function set global constants and will be called with the default
    argument when `earth` module is imported for the first time.

    The 2 implemented models are:

        1. WGS84 which is used in GPS.
        2. PZ90 which is used in GLONASS.

    All parameters are taken from official datums.

    Earth ellipsoid is defined by its major semi axis ``R0`` and the square of
    eccentricity ``E2``.

    The vector of gravity is assumed to be orthogonal to the surface of Earth
    ellipsoid, its magnitude depends on a latitude. The scaling with an
    altitude is computed as ``g0 * (1 - 2 * alt / R0)``, where ``g0`` is the
    gravity on the surface and  ``alt`` is the altitude.

    The value of ``g0`` depends on a latitude according to Somigliana formula
    [1]_ (p. 70). It is determined by semi major and semi minor axes and
    values of the gravity at the equator and the pole (which are given in the
    datums).

    Parameters
    ----------
    model : 'WGS84' or 'PZ90', optional
        Model to use. Default is 'WGS84'.

    References
    ----------
    .. [1] W. A. Heiskanen, H. Moritz, "Physical Geodesy".
    """
    global MODEL, RATE, R0, E2, GE, F
    if model == 'WGS84':
        R0 = 6378137.0
        E2 = 6.6943799901413e-3
        GE = 9.7803253359
        gp = 9.8321849378  # Gravity at the pole.
    elif model == 'PZ90':
        R0 = 6378136.0
        E2 = 6.69436619e-3
        GE = 9.7803284
        gp = 9.8321880  # Gravity at the pole.
    else:
        raise ValueError("`model` must be 'WGS84' or 'PZ90'.")

    b = R0 * (1 - E2) ** 0.5  # Semi minor axis.
    F = b * gp / (GE * R0) - 1

    MODEL = model


def principal_radii(slat):
    """Compute the principal radii of curvature of Earth ellipsoid.

    See [1]_ (p. 41) for the definition and formulas.

    Parameters
    ----------
    slat : array_like
        Sine of a latitude.

    Returns
    -------
    re, rn : ndarray
        Radii of curvature in East and North directions respectively.

    Notes
    -----
    The motivation to make this function accept the sine of a latitude as an
    argument is twofold:

        1. The radii depend only on the sine of a latitude (namely on its
           square).
        2. The sine of a latitude can be conveniently extracted from a DCM
           related ECEF and local navigation frames. So this implementation
           is more convenient when coordinates are computed in a DCM (including
           usage of a wander-azimuth frame), even though it is not currently
           presented in the package.

    References
    ----------
    .. [1] P. D. Groves, "Principles of GNSS, Inertial, and Multisensor
           Integrated Navigation Systems".
    """
    slat = np.asarray(slat)
    if np.any(np.abs(slat) > 1):
        raise ValueError("`sin_lat` must contain values between -1 and 1.")

    x = 1 - E2 * slat ** 2
    re = R0 / np.sqrt(x)
    rn = re * (1 - E2) / x

    return re, rn


def gravity(slat, alt=0):
    """Compute gravity according to a theoretical model.

    See `set_model` for the explanation of the formula used.

    Parameters
    ----------
    slat : array_like
        Sine of a latitude.
    alt : array_like, optional
        Altitude. Default is 0.

    Returns
    -------
    g : ndarray
        Magnitude of the gravity.

    Notes
    -----
    The motivation to make this function accept the sine of a latitude as an
    argument is twofold:

        1. The gravity depends only on the sine of a latitude (namely on its
           square).
        2. The sine of a latitude can be conveniently extracted from a DCM
           related ECEF and local navigation frames. So this implementation
           is more convenient for DCM coordinates mechanization (including
           usage of a wander-azimuth frame), even though it is not currently
           presented in the package.
    """
    slat = np.asarray(slat)
    if np.any(np.abs(slat) > 1):
        raise ValueError("`slat` must contain values between -1 and 1.")

    alt = np.asarray(alt)

    slat2 = slat ** 2

    return GE * (1 + F * slat2) / (1 - E2 * slat2)**0.5 * (1 - 2 * alt / R0)


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

    slat, clat = np.sin(np.deg2rad(lat)), np.cos(np.deg2rad(lat))
    re, _ = principal_radii(slat)
    rp = re * clat

    g0_g = np.zeros((3,) + slat.shape)
    g0_g[1] = RATE**2 * rp * slat
    g0_g[2] = -gravity(slat, alt) - RATE ** 2 * rp * clat
    g0_g = g0_g.T

    Ceg = dcm.from_llw(lat, lon)

    return mv_prod(Ceg, g0_g)


set_model()

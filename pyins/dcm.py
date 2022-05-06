"""Create and manipulate direction cosine matrices."""
import numpy as np
from scipy.spatial.transform import Rotation


def from_rv(rv):
    """Create a direction cosine matrix from a rotation vector.

    The direction of a rotation vector determines the axis of rotation and its
    magnitude determines the angle of rotation.

    The returned DCM projects a vector from the rotated frame to the original
    frame.

    Parameters
    ----------
    rv : array_like, shape (3,) or (n, 3)
        Rotation vectors.

    Returns
    -------
    dcm : ndarray, shape (3, 3) or (n, 3, 3)
        Direction cosine matrices.
    """
    return Rotation.from_rotvec(rv).as_matrix()


def from_hpr(h, p, r):
    """Create a direction cosine matrix from heading, pitch and roll angles.

    The sequence of elemental rotations is as follows::

           -heading   pitch   roll
        N ---------> ------> -----> B
              3         1       2

    Here N denotes the local level wander-azimuth frame and B denotes the body
    frame. The resulting DCM projects from B frame to N frame.

    Parameters
    ----------
    h, p, r : float or array_like with shape (n,)
        Heading, pitch and roll.

    Returns
    -------
    dcm : ndarray, shape (3, 3) or (n, 3, 3)
        Direction cosine matrices.
    """
    h = np.asarray(h)
    p = np.asarray(p)
    r = np.asarray(r)
    if h.ndim == 0 and p.ndim == 0 and r.ndim == 0:
        return Rotation.from_euler('yxz', [r, p, -h], degrees=True).as_matrix()

    h = np.atleast_1d(h)
    p = np.atleast_1d(p)
    r = np.atleast_1d(r)

    n = max(len(h), len(p), len(r))

    angles = np.empty((n, 3))
    angles[:, 0] = r
    angles[:, 1] = p
    angles[:, 2] = -h
    return Rotation.from_euler('yxz', angles, degrees=True).as_matrix()


def from_ll(lat, lon):
    """Create a direction cosine matrix from latitude and longitude.

    The sequence of elemental rotations is as follows::

           pi/2+lon    pi/2-lan
        E ----------> ----------> N
               3           1

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
        return Rotation.from_euler('ZX', [90 + lon, 90 - lat],
                                   degrees=True).as_matrix()

    lat = np.atleast_1d(lat)
    lon = np.atleast_1d(lon)

    n = max(len(lat), len(lon))
    angles = np.empty((n, 2))
    angles[:, 0] = 90 + lon
    angles[:, 1] = 90 - lat
    return Rotation.from_euler('ZX', angles, degrees=True).as_matrix()


def to_hpr(dcm):
    """Convert a direction cosine matrix to heading, pitch and roll angles.

    The returned heading is within [0, 360], the pitch is within [-90, 90]
    and the roll is within [-90, 90].

    If ``90 - abs(pitch) < np.rad2deg(1e-3)`` then the roll is set to 0 and
    the heading is computed accordingly.

    Parameters
    ----------
    dcm : array_like, shape (3, 3) or (n, 3, 3)
        Direction cosine matrices.

    Returns
    -------
    h, p, r : float or ndarray with shape (n,)
        Heading, pitch and roll.
    """
    h, p, r = Rotation.from_matrix(dcm).as_euler('ZXY', degrees=True).T
    h = -h
    if h.ndim == 0:
        if h < 0:
            h += 360
    else:
        h[h < 0] += 360

    return h, p, r

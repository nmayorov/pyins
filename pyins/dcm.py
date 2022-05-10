"""Create and manipulate direction cosine matrices."""
import numpy as np
from scipy.spatial.transform import Rotation


def skew_matrix(vec):
    """Compute a skew matrix corresponding to a vector.

    Parameters
    ----------
    vec : array_like, shape (3,) or (n, 3)
        Vector.

    Returns
    -------
    skew_matrix : ndarray, shape (3, 3) or (n, 3)
        Corresponding skew matrix.
    """
    vec = np.asarray(vec)
    single = vec.ndim == 1
    n = 1 if single else len(vec)
    vec = np.atleast_2d(vec)
    result = np.zeros((n, 3, 3))
    result[:, 0, 1] = -vec[:, 2]
    result[:, 0, 2] = vec[:, 1]
    result[:, 1, 0] = vec[:, 2]
    result[:, 1, 2] = -vec[:, 0]
    result[:, 2, 0] = -vec[:, 1]
    result[:, 2, 1] = vec[:, 0]
    return result[0] if single else result


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


def from_rph(rph):
    """Create a direction cosine matrix from roll, pitch and heading.

    The sequence of elemental rotations is as follows::

           -heading   pitch   roll
        N ---------> ------> -----> B
              3         1       2

    Here N denotes the local level wander-azimuth frame and B denotes the body
    frame. The resulting DCM projects from B frame to N frame.

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
        return Rotation.from_euler('ZY', [lon, -90 - lat],
                                   degrees=True).as_matrix()

    lat = np.atleast_1d(lat)
    lon = np.atleast_1d(lon)

    n = max(len(lat), len(lon))
    angles = np.empty((n, 2))
    angles[:, 0] = lon
    angles[:, 1] = -90 - lat
    return Rotation.from_euler('ZY', angles, degrees=True).as_matrix()


def to_rph(dcm):
    """Convert a direction cosine matrix to roll, pitch, heading angles.

    The returned heading is within [0, 360], the pitch is within [-90, 90]
    and the roll is within [-90, 90].

    Parameters
    ----------
    dcm : array_like, shape (3, 3) or (n, 3, 3)
        Direction cosine matrices.

    Returns
    -------
    rph : ndarray, with shape (3,) or (n, 3)
        Heading, pitch and roll.
    """
    return Rotation.from_matrix(dcm).as_euler('xyz', degrees=True)

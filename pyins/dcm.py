"""Create and manipulate direction cosine matrices."""
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

"""Create and manipulate direction cosine matrices."""
import numpy as np
from scipy.interpolate import PPoly
from scipy.linalg import svd, det
from scipy.spatial.transform import Rotation
from . import util
from ._dcm_spline_solver import solve_for_omega


def _skew_matrix_single(rv):
    return np.array([
        [0, -rv[2], rv[1]],
        [rv[2], 0, -rv[0]],
        [-rv[1], rv[0], 0]
    ])


def _skew_matrix_array(rv):
    skew = np.zeros((rv.shape[0], 3, 3))
    skew[:, 0, 1] = -rv[:, 2]
    skew[:, 0, 2] = rv[:, 1]
    skew[:, 1, 0] = rv[:, 2]
    skew[:, 1, 2] = -rv[:, 0]
    skew[:, 2, 0] = -rv[:, 1]
    skew[:, 2, 1] = rv[:, 0]
    return skew


def _from_rotvec_single(rv):
    norm2 = np.dot(rv, rv)
    if norm2 > 1e-6:
        norm = norm2 ** 0.5
        k1 = np.sin(norm) / norm
        k2 = (1 - np.cos(norm)) / norm2
    else:
        norm4 = norm2 * norm2
        k1 = 1 - norm2 / 6 + norm4 / 120
        k2 = 0.5 - norm2 / 24 + norm4 / 720

    skew = _skew_matrix_single(rv)
    return np.eye(3) + k1 * skew + k2 * np.dot(skew, skew)


def _from_rotvec_array(rv):
    norm = np.linalg.norm(rv, axis=1)
    norm2 = norm ** 2
    norm4 = norm2 ** 2

    k1 = np.empty_like(norm2)
    k2 = np.empty_like(norm2)

    small = norm2 < 1e-6
    k1[small] = 1 - norm2[small] / 6 + norm4[small] / 120
    k2[small] = 0.5 - norm2[small] / 24 + norm4[small] / 720

    big = ~small
    k1[big] = np.sin(norm[big]) / norm[big]
    k2[big] = (1 - np.cos(norm[big])) / norm2[big]

    skew = _skew_matrix_array(rv)
    skew_squared = np.einsum('...ij,...jk->...ik', skew, skew)
    identity = np.empty_like(skew)
    identity[:] = np.identity(3)
    return (identity +
            k1[:, np.newaxis, np.newaxis] * skew +
            k2[:, np.newaxis, np.newaxis] * skew_squared)


def from_basic(axis, angle):
    """Create a direction cosine matrix corresponding to a basic rotation.

    This is a DCM corresponding to the rotation around one of the axes of a
    frame (i.e. 1, 2 or 3).

    Parameters
    ----------
    axis : 1, 2 or 3
        Axis to rotate around.
    angle : float or array_like, shape (n_points,)
        Angle of rotation in degrees.

    Returns
    -------
    dcm : array_like, shape (3, 3) or (n_points, 3, 3)
        Direction cosine matrix.
    """
    if axis not in [1, 2, 3]:
        raise ValueError("`axis` must be 1, 2 or 3.")

    angle = np.deg2rad(angle)
    dcm = np.zeros((3, 3) + angle.shape)
    sa = np.sin(angle)
    ca = np.cos(angle)
    if axis == 1:
        dcm[0, 0] = 1
        dcm[1, 1] = ca
        dcm[1, 2] = -sa
        dcm[2, 1] = sa
        dcm[2, 2] = ca
    elif axis == 2:
        dcm[0, 0] = ca
        dcm[0, 2] = sa
        dcm[1, 1] = 1
        dcm[2, 0] = -sa
        dcm[2, 2] = ca
    elif axis == 3:
        dcm[0, 0] = ca
        dcm[0, 1] = -sa
        dcm[1, 0] = sa
        dcm[1, 1] = ca
        dcm[2, 2] = 1

    if dcm.ndim == 3:
        dcm = np.rollaxis(dcm, -1)

    return dcm


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
    rv = np.asarray(rv)
    if rv.ndim not in [1, 2] or rv.shape[-1] != 3:
        raise ValueError("`rv` has a wrong shape.")

    if rv.ndim == 1:
        return _from_rotvec_single(rv)

    n = rv.shape[0]
    if n < 10:
        dcm = np.empty((n, 3, 3))
        for i in range(n):
            dcm[i] = _from_rotvec_single(rv[i])
        return dcm
    else:
        return _from_rotvec_array(rv)


def _to_rotvec_single(dcm):
    s = 0.5 * (
        (dcm[2, 1] - dcm[1, 2])**2 +
        (dcm[0, 2] - dcm[2, 0])**2 +
        (dcm[1, 0] - dcm[0, 1])**2) ** 0.5
    c = 0.5 * (dcm.trace() - 1)
    angle = np.arctan2(s, c)

    if c > 0:
        if angle < 1e-3:
            F = 1 + angle**2 / 6 + 7 * angle**4 / 360
        else:
            F = angle / s
        rv = 0.5 * F * np.array([dcm[2, 1] - dcm[1, 2],
                                 dcm[0, 2] - dcm[2, 0],
                                 dcm[1, 0] - dcm[0, 1]])
    else:
        axis_abs = (1 + (dcm.diagonal() - 1) / (1 - c)) ** 0.5
        max_dir = np.argmax(axis_abs)
        if max_dir == 0:
            u1 = axis_abs[0] * np.sign(dcm[2, 1] - dcm[1, 2])
            u2 = 0.5 * (dcm[0, 1] + dcm[1, 0]) / (u1 * (1 - c))
            u3 = 0.5 * (dcm[0, 2] + dcm[2, 0]) / (u1 * (1 - c))
        elif max_dir == 1:
            u2 = axis_abs[1] * np.sign(dcm[0, 2] - dcm[2, 0])
            u3 = 0.5 * (dcm[1, 2] + dcm[2, 1]) / (u2 * (1 - c))
            u1 = 0.5 * (dcm[0, 1] + dcm[1, 0]) / (u2 * (1 - c))
        else:
            u3 = axis_abs[2] * np.sign(dcm[1, 0] - dcm[0, 1])
            u1 = 0.5 * (dcm[0, 2] + dcm[2, 0]) / (u3 * (1 - c))
            u2 = 0.5 * (dcm[1, 2] + dcm[2, 1]) / (u3 * (1 - c))
        rv = angle * np.array([u1, u2, u3])

    return rv


def _to_rotvec_array(dcm):
    s = 0.5 * ((dcm[:, 2, 1] - dcm[:, 1, 2])**2 +
               (dcm[:, 0, 2] - dcm[:, 2, 0])**2 +
               (dcm[:, 1, 0] - dcm[:, 0, 1])**2) ** 0.5
    c = 0.5 * (dcm.trace(axis1=1, axis2=2) - 1)
    norm = np.arctan2(s, c)

    rv = np.empty((norm.size, 3))
    # Three cases need to be considered.

    # Case 1: non negative cosine.
    mask = c >= 0
    rv[mask, 0] = dcm[mask, 2, 1] - dcm[mask, 1, 2]
    rv[mask, 1] = dcm[mask, 0, 2] - dcm[mask, 2, 0]
    rv[mask, 2] = dcm[mask, 1, 0] - dcm[mask, 0, 1]

    F = np.empty_like(norm)
    # Sub case 1a: non small angle.
    big = mask & (norm >= 1e-3)
    F[big] = norm[big] / s[big]
    # Sub case 1b: small angles.
    small = mask & ~big
    norm2 = norm[small] ** 2
    F[small] = 1 + norm2 / 6 + 7 * norm2**2 / 360
    rv[mask] *= 0.5 * F[mask, np.newaxis]

    # Case 2: negative cosine.
    mask = c < 0
    dcm_mask = dcm[mask]
    axis = np.empty(shape=(np.sum(mask), 3))
    c_mask = c[mask]

    # Might become -eps.
    axis_abs = np.maximum((dcm_mask.diagonal(axis1=1, axis2=2) - 1) /
                          (1 - c_mask[:, np.newaxis]) + 1, 0) ** 0.5

    k = np.argmax(axis_abs, axis=1)
    # Case 2a: the first direction component has max abs value
    k_mask = k == 0
    omc = 1 - c_mask[k_mask]
    u1 = axis_abs[k_mask, 0] * np.sign(dcm_mask[k_mask, 2, 1] -
                                       dcm_mask[k_mask, 1, 2])
    u2 = 0.5 * (dcm_mask[k_mask, 0, 1] + dcm_mask[k_mask, 1, 0]) / (u1 * omc)
    u3 = 0.5 * (dcm_mask[k_mask, 0, 2] + dcm_mask[k_mask, 2, 0]) / (u1 * omc)
    axis[k_mask, 0] = u1
    axis[k_mask, 1] = u2
    axis[k_mask, 2] = u3

    # Case 2b: the first direction component has max abs value
    k_mask = k == 1
    omc = 1 - c_mask[k_mask]
    u2 = axis_abs[k_mask, 1] * np.sign(dcm_mask[k_mask, 0, 2] -
                                       dcm_mask[k_mask, 2, 0])
    u3 = 0.5 * (dcm_mask[k_mask, 1, 2] + dcm_mask[k_mask, 2, 1]) / (u2 * omc)
    u1 = 0.5 * (dcm_mask[k_mask, 0, 1] + dcm_mask[k_mask, 1, 0]) / (u2 * omc)
    axis[k_mask, 0] = u1
    axis[k_mask, 1] = u2
    axis[k_mask, 2] = u3

    # Case 2c: the first direction component has max abs value
    k_mask = k == 2
    omc = 1 - c_mask[k_mask]
    u3 = axis_abs[k_mask, 2] * np.sign(dcm_mask[k_mask, 1, 0] -
                                       dcm_mask[k_mask, 0, 1])
    u1 = 0.5 * (dcm_mask[k_mask, 0, 2] + dcm_mask[k_mask, 2, 0]) / (u3 * omc)
    u2 = 0.5 * (dcm_mask[k_mask, 1, 2] + dcm_mask[k_mask, 2, 1]) / (u3 * omc)
    axis[k_mask, 0] = u1
    axis[k_mask, 1] = u2
    axis[k_mask, 2] = u3

    rv[mask] = norm[mask, np.newaxis] * axis
    return rv


def to_rv(dcm):
    """Convert a direction cosine matrix to a rotation vector.

    Parameters
    ----------
    dcm : array_like, shape (3, 3) or (n, 3, 3)
        Direction cosine matrices.

    Returns
    -------
    rv : ndarray, shape (3,) or (n, 3)
        Rotation vectors.
    """
    dcm = np.asarray(dcm)
    if dcm.ndim not in [2, 3] or dcm.shape[-1] != 3 or dcm.shape[-2] != 3:
        raise ValueError("`dcm` has a wrong shape.")

    if dcm.ndim == 2:
        return _to_rotvec_single(dcm)

    n = dcm.shape[0]
    if n < 10:
        rv = np.empty((n, 3))
        for i in range(n):
            rv[i] = _to_rotvec_single(dcm[i])
        return rv
    else:
        return _to_rotvec_array(dcm)


def from_quat(quat):
    """Create a direction cosine matrix from a quaternion.

    First 3 elements of the quaternion form its vector part.

    Parameters
    ----------
    quat : array_like, shape (4,) or (n, 4)
        Quaternions.

    Returns
    -------
    dcm : ndarray, shape (3, 3) or (n, 3, 3)
        Direction cosine matrices.
    """
    q = np.asarray(quat)
    if q.ndim == 1:
        rho = q[:3]
        q4 = q[3]
        rho_skew = _skew_matrix_single(rho)
        dcm = 2 * (np.outer(rho, rho) + q4 * rho_skew)
        dcm[np.diag_indices_from(dcm)] += q4**2 - np.dot(rho, rho)
    else:
        rho = q[:, :3]
        q4 = q[:, 3]
        rho_skew = _skew_matrix_array(rho)
        dcm = 2 * (rho[:, None, :] * rho[:, :, None] +
                   q4[:, None, None] * rho_skew)
        diag = q4**2 - np.sum(rho**2, axis=1)
        dcm[:, np.arange(3), np.arange(3)] += diag[:, None]

    return dcm


def to_quat(dcm):
    """Convert a direction cosine matrix to a quaternion.

    First 3 elements of the quaternion form its vector part. The conversion
    algorithm is from [1]_.

    Parameters
    ----------
    dcm : array_like, shape (3, 3) or (n, 3, 3)
        Direction cosine matrices.

    Returns
    -------
    quat : ndarray, shape (4,) or (n, 4)
        Quaternions.

    References
    ----------
    .. [1] F. L. Markley, "Unit Quaternion from Rotation Matrix", Journal
       of Guidance Control and Dynamics, Vol. 31, No. 2, 2008
    """
    if dcm.ndim == 2:
        A = dcm
        v = np.hstack((np.diag(A), np.trace(A)))
        case = np.argmax(v)
        x = np.empty(4)
        if case == 0:
            x[0] = -1 - A[0, 0] + A[1, 1] + A[2, 2]
            x[1] = -A[0, 1] - A[1, 0]
            x[2] = -A[0, 2] - A[2, 0]
            x[3] = A[1, 2] - A[2, 1]
        elif case == 1:
            x[0] = -A[1, 0] - A[0, 1]
            x[1] = -1 - A[1, 1] + A[2, 2] + A[0, 0]
            x[2] = -A[1, 2] - A[2, 1]
            x[3] = A[2, 0] - A[0, 2]
        elif case == 2:
            x[0] = -A[2, 0] - A[0, 2]
            x[1] = -A[2, 1] - A[1, 2]
            x[2] = -1 - A[2, 2] + A[0, 0] + A[1, 1]
            x[3] = A[0, 1] - A[1, 0]
        else:
            x[0] = -A[1, 2] + A[2, 1]
            x[1] = -A[2, 0] + A[0, 2]
            x[2] = -A[0, 1] + A[1, 0]
            x[3] = 1 + A[0, 0] + A[1, 1] + A[2, 2]

        x /= np.linalg.norm(x)
    else:
        v = np.hstack((dcm[:, np.arange(3), np.arange(3)],
                       np.trace(dcm, axis1=1, axis2=2)[:, None]))
        case = np.argmax(v, axis=1)
        x = np.empty((dcm.shape[0], 4))
        dcm = np.transpose(dcm, axes=(1, 2, 0))

        mask = case == 0
        A = dcm[:, :, mask]
        x[mask, 0] = -1 - A[0, 0] + A[1, 1] + A[2, 2]
        x[mask, 1] = -A[0, 1] - A[1, 0]
        x[mask, 2] = -A[0, 2] - A[2, 0]
        x[mask, 3] = A[1, 2] - A[2, 1]

        mask = case == 1
        A = dcm[:, :, mask]
        x[mask, 0] = -A[1, 0] - A[0, 1]
        x[mask, 1] = -1 - A[1, 1] + A[2, 2] + A[0, 0]
        x[mask, 2] = -A[1, 2] - A[2, 1]
        x[mask, 3] = A[2, 0] - A[0, 2]

        mask = case == 2
        A = dcm[:, :, mask]
        x[mask, 0] = -A[2, 0] - A[0, 2]
        x[mask, 1] = -A[2, 1] - A[1, 2]
        x[mask, 2] = -1 - A[2, 2] + A[0, 0] + A[1, 1]
        x[mask, 3] = A[0, 1] - A[1, 0]

        mask = case == 3
        A = dcm[:, :, mask]
        x[mask, 0] = -A[1, 2] + A[2, 1]
        x[mask, 1] = -A[2, 0] + A[0, 2]
        x[mask, 2] = -A[0, 1] + A[1, 0]
        x[mask, 3] = 1 + A[0, 0] + A[1, 1] + A[2, 2]

        x /= np.linalg.norm(x, axis=1)[:, None]

    return x


def from_gibbs(gibbs):
    """Create a direction cosine matrix from Gibbs vectors.

    These parameters are related to quaternion components as follows::

        mrp = 2 * q_vec / q_scal

    It can be expressed from the angle and axis of rotation::

        mrp = 2 * tan(angle / 2) * axis

    Parameters
    ----------
    gibbs : array_like, shape (3,) or (n, 3)
        Gibbs vectors.

    Returns
    -------
    dcm : ndarray, shape (3, 3) or (n, 3, 3)
        Direction cosine matrices.
    """
    gibbs = np.asarray(gibbs, dtype=float).T
    norm2 = np.sum(gibbs ** 2, axis=0)

    q_scal = 2 / (4 + norm2) ** 0.5
    q_vec = 0.5 * q_scal * gibbs

    if gibbs.ndim == 1:
        q = np.hstack((q_vec, q_scal))
    else:
        q = np.vstack((q_vec, q_scal)).T

    return from_quat(q)


def to_gibbs(dcm):
    """Convert a direction cosine matrix to Gibbs vectors.

    See `from_gibbs`.

    Parameters
    ----------
    dcm : array_like, shape (3, 3) or (n, 3, 3)
        Direction cosine matrices.

    Returns
    -------
    gibbs : ndarray, shape (3,) or (n, 3)
        Gibbs vectors.
    """
    q = to_quat(dcm).T
    gibbs = 2 * q[:3] / q[3]

    return gibbs.T


def from_mrp(mrp):
    """Create a direction cosine matrix from modified Rodrigues parameters.

    These parameters are related to quaternion components as follows::

        mrp = 4 * q_vec / (1 + q_scal)

    It can be expressed from the angle and axis of rotation::

        mrp = 4 * tan(angle / 4) * axis

    Parameters
    ----------
    mrp : array_like, shape (3,) or (n, 3)
        Vectors of modified Rodigues parameters.

    Returns
    -------
    dcm : ndarray, shape (3, 3) or (n, 3, 3)
        Direction cosine matrices.
    """
    mrp = np.asarray(mrp, dtype=float).T
    norm2 = np.sum(mrp ** 2, axis=0)

    q_scal = (16 - norm2) / (16 + norm2)
    q_vec = (1 + q_scal) / 4 * mrp

    if mrp.ndim == 1:
        q = np.hstack((q_vec, q_scal))
    else:
        q = np.vstack((q_vec, q_scal)).T

    return from_quat(q)


def to_mrp(dcm):
    """Convert a direction cosine matrix to modified Rodrigues parameters.

    See `from_mrp`.

    Parameters
    ----------
    dcm : array_like, shape (3, 3) or (n, 3, 3)
        Direction cosine matrices.

    Returns
    -------
    mrp : ndarray, shape (3,) or (n, 3)
        Vectors of generalized Rodrigues parameters.
    """
    q = to_quat(dcm).T
    grp = 4 / (1 + q[3]) * q[:3]

    return grp.T


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


def from_llw(lat, lon):
    """Create a direction cosine matrix from latitude and longitude.

    The sequence of elemental rotations is as follows::

           pi/2+lon    pi/2-lan     wan
        E ----------> ----------> ------> N
               3           1         3

    Here E denotes the ECEF frame and N denotes the local level wander-angle
    frame. The resulting DCM projects from N frame to E frame.

    If ``wan=0`` then the 2nd axis of N frame points to North.

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
        return Rotation.from_euler('xz', [90 - lat, 90 + lon], degrees=True).as_matrix()

    lat = np.atleast_1d(lat)
    lon = np.atleast_1d(lon)

    n = max(len(lat), len(lon))

    angles = np.empty((n, 2))
    angles[:, 0] = 90 - lat
    angles[:, 1] = 90 + lon
    return Rotation.from_euler('xz', angles, degrees=True).as_matrix()


def _to_hpr_single(dcm):
    pitch = np.arcsin(dcm[2, 1])
    if np.abs(pitch) < 0.5 * np.pi - 1e-3:
        heading = np.arctan2(dcm[0, 1], dcm[1, 1])
        roll = np.arctan2(-dcm[2, 0], dcm[2, 2])
    elif pitch > 0:
        roll = 0
        heading = np.arctan2(-dcm[0, 2] - dcm[1, 0], dcm[0, 0] - dcm[1, 2])
    else:
        roll = 0
        heading = np.arctan2(dcm[0, 2] - dcm[1, 0], dcm[0, 0] + dcm[1, 2])

    if heading < 0:
        heading += 2 * np.pi

    if heading == 2 * np.pi:
        heading = 0

    return heading, pitch, roll


def _to_hpr_array(dcm):
    h = np.empty(dcm.shape[0])
    p = np.arcsin(dcm[:, 2, 1])
    r = np.empty(dcm.shape[0])

    mask = np.abs(p) < 0.5 * np.pi - 1e-3
    h[mask] = np.arctan2(dcm[mask, 0, 1], dcm[mask, 1, 1])
    r[mask] = np.arctan2(-dcm[mask, 2, 0], dcm[mask, 2, 2])

    mask = ~mask
    r[mask] = 0

    h_mask = mask & (p > 0)
    h[h_mask] = np.arctan2(-dcm[h_mask, 0, 2] - dcm[h_mask, 1, 0],
                            dcm[h_mask, 0, 0] - dcm[h_mask, 1, 2])

    h_mask = mask & (p < 0)
    h[h_mask] = np.arctan2(dcm[h_mask, 0, 2] - dcm[h_mask, 1, 0],
                           dcm[h_mask, 0, 0] + dcm[h_mask, 1, 2])

    h[h < 0] += 2 * np.pi

    # If h were very small and negative after adding 2 * pi it can become
    # 2 * pi exactly, then it's better to make it 0.
    h[h == 2 * np.pi] = 0

    return h, p, r


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
    dcm = np.asarray(dcm)
    if dcm.ndim not in [2, 3] or dcm.shape[-1] != 3 or dcm.shape[-2] != 3:
        raise ValueError('`dcm` has a wrong shape.')

    if dcm.ndim == 2:
        h, p, r = _to_hpr_single(dcm)
    else:
        h, p, r = _to_hpr_array(dcm)

    return np.rad2deg([h, p, r])


def from_llw(lat, lon, wan=0):
    """Create a direction cosine matrix from latitude, longitude and wander.

    The sequence of elemental rotations is as follows::

           pi/2+lon    pi/2-lan     wan
        E ----------> ----------> ------> N
               3           1         3

    Here E denotes the ECEF frame and N denotes the local level wander-angle
    frame. The resulting DCM projects from N frame to E frame.

    If ``wan=0`` then the 2nd axis of N frame points to North.

    Parameters
    ----------
    lat, lon : float or array_like with shape (n,)
        Latitude and longitude.
    wan : float or array_like with shape (n,), optional
        Wander angle. Default is 0.

    Returns
    -------
    dcm : ndarray, shape (3, 3) or (n, 3, 3)
        Direction Cosine Matrices.
    """
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    wan = np.deg2rad(wan)

    clat = np.cos(lat)
    slat = np.sin(lat)
    clon = np.cos(lon)
    slon = np.sin(lon)
    cwan = np.cos(wan)
    swan = np.sin(wan)

    dcm = np.empty((3, 3) + lat.shape)

    dcm[0, 0] = -clon * slat * swan - slon * cwan
    dcm[0, 1] = -clon * slat * cwan + slon * swan
    dcm[0, 2] = clon * clat
    dcm[1, 0] = -slon * slat * swan + clon * cwan
    dcm[1, 1] = -slon * slat * cwan - clon * swan
    dcm[1, 2] = slon * clat
    dcm[2, 0] = clat * swan
    dcm[2, 1] = clat * cwan
    dcm[2, 2] = slat

    if dcm.ndim == 3:
        dcm = np.rollaxis(dcm, -1)

    return dcm


def _to_llw_single(dcm):
    lat = np.arcsin(dcm[2, 2])
    if np.abs(lat) < 0.5 * np.pi - 1e-3:
        lon = np.arctan2(dcm[1, 2], dcm[0, 2])
        wan = np.arctan2(dcm[2, 0], dcm[2, 1])
    elif lat > 0:
        lon = 0
        wan = np.arctan2(-dcm[0, 0] - dcm[1, 1], dcm[1, 0] - dcm[0, 1])
    else:
        lon = 0
        wan = np.arctan2(dcm[0, 0] - dcm[1, 1], dcm[0, 1] + dcm[1, 0])

    return lat, lon, wan


def _to_llw_array(dcm):
    lat = np.arcsin(dcm[:, 2, 2])
    lon = np.empty(dcm.shape[0])
    wan = np.empty(dcm.shape[0])

    mask = np.abs(lat) < 0.5 * np.pi - 1e-3
    lon[mask] = np.arctan2(dcm[mask, 1, 2], dcm[mask, 0, 2])
    wan[mask] = np.arctan2(dcm[mask, 2, 0], dcm[mask, 2, 1])

    mask = ~mask
    lon[mask] = 0

    l_mask = mask & (lat > 0)
    wan[l_mask] = np.arctan2(-dcm[l_mask, 0, 0] - dcm[l_mask, 1, 1],
                             dcm[l_mask, 1, 0] - dcm[l_mask, 0, 1])

    l_mask = mask & (lat < 0)
    wan[l_mask] = np.arctan2(dcm[l_mask, 0, 0] - dcm[l_mask, 1, 1],
                             dcm[l_mask, 0, 1] + dcm[l_mask, 1, 0])

    return lat, lon, wan


def to_llw(dcm):
    """Convert a direction cosine matrix to latitude, longitude and wander.

    The latitude is within [-90, 90], the longitude and the wander are
    within [-180, 180].

    If ``90 - abs(latitude) < np.rad2deg(1e-3)`` then the longitude is set to
    0 and the wander is computed accordingly.

    Parameters
    ----------
    dcm : array_like, shape (3, 3) or (n, 3, 3)
        Direction cosine matrices.

    Returns
    -------
    lat, lon, wan : float or ndarray with shape (n,)
        Latitude, longitude and wander. Note that `wan` is always returned
        even though it can be known to be equal 0.
    """
    dcm = np.asarray(dcm)
    if dcm.ndim not in [2, 3] or dcm.shape[-1] != 3 or dcm.shape[-2] != 3:
        raise ValueError("`dcm` has a wrong shape.")

    if dcm.ndim == 2:
        lat, lon, wan = _to_llw_single(dcm)
    else:
        lat, lon, wan = _to_llw_array(dcm)

    return np.rad2deg([lat, lon, wan])


def _dtheta_from_omega_matrix(theta):
    norm = np.linalg.norm(theta, axis=1)
    k = np.empty_like(norm)

    mask = norm > 1e-4
    nm = norm[mask]
    k[mask] = (1 - 0.5 * nm / np.tan(0.5 * nm)) / nm**2
    mask = ~mask
    nm = norm[mask]
    k[mask] = 1/12 + 1/720 * nm**2

    A = np.empty((norm.shape[0], 3, 3))
    skew = _skew_matrix_array(theta)
    A[:] = np.identity(3)
    A[:] += 0.5 * skew
    A[:] += k[:, None, None] * util.mm_prod(skew, skew)

    return A


def _omega_from_dtheta_matrix(theta):
    norm = np.linalg.norm(theta, axis=1)
    k1 = np.empty_like(norm)
    k2 = np.empty_like(norm)

    mask = norm > 1e-4
    nm = norm[mask]
    k1[mask] = (1 - np.cos(nm)) / nm**2
    k2[mask] = (nm - np.sin(nm)) / nm**3

    mask = ~mask
    nm = norm[mask]
    k1[mask] = 0.5 - nm**2 / 24
    k2[mask] = 1/6 - nm**2 / 120

    A = np.empty((norm.shape[0], 3, 3))
    skew = _skew_matrix_array(theta)
    A[:] = np.identity(3)
    A[:] -= k1[:, None, None] * skew
    A[:] += k2[:, None, None] * util.mm_prod(skew, skew)

    return A


def _compute_delta_beta(theta, dtheta):
    norm = np.linalg.norm(theta, axis=1)
    dp = np.sum(theta * dtheta, axis=1)
    cp = np.cross(theta, dtheta)
    ccp = np.cross(theta, cp)
    dccp = np.cross(dtheta, cp)

    k1 = np.empty_like(norm)
    k2 = np.empty_like(norm)
    k3 = np.empty_like(norm)

    mask = norm > 1e-4
    nm = norm[mask]
    k1[mask] = (-nm * np.sin(nm) - 2 * (np.cos(nm) - 1)) / nm ** 4
    k2[mask] = (-2 * nm + 3 * np.sin(nm) - nm * np.cos(nm)) / nm ** 5
    k3[mask] = (nm - np.sin(nm)) / nm ** 3

    mask = ~mask
    nm = norm[mask]
    k1[mask] = 1/12 - nm ** 2 / 180
    k2[mask] = -1/60 + nm ** 2 / 12604
    k3[mask] = 1/6 - nm ** 2 / 120

    dp = dp[:, None]
    k1 = k1[:, None]
    k2 = k2[:, None]
    k3 = k3[:, None]

    return dp * (k1 * cp + k2 * ccp) + k3 * dccp


def _compute_omega(theta, dtheta):
    cp = np.cross(theta, dtheta)
    ccp = np.cross(theta, cp)

    k1 = np.empty(theta.shape[0])
    k2 = np.empty(theta.shape[0])

    norm = np.linalg.norm(theta, axis=1)

    mask = norm > 1e-4
    nm = norm[mask]
    k1[mask] = (1 - np.cos(nm)) / nm**2
    k2[mask] = (nm - np.sin(nm)) / nm**3

    mask = ~mask
    nm = norm[mask]
    k1[mask] = 0.5 - nm**2 / 24
    k2[mask] = 1/6 - nm**2 / 120

    return dtheta - k1[:, None] * cp + k2[:, None] * ccp


def _compute_beta(theta, dtheta, ddtheta):
    return _compute_omega(theta, ddtheta) + _compute_delta_beta(theta, dtheta)


def _solve_system(dt, A, B, Theta, dTheta, omega):
    delta_beta = _compute_delta_beta(Theta, dTheta)

    diag = 4 * (1 / dt[:-1] + 1 / dt[1:])
    diag = np.hstack((1, diag))
    rhs = (6 / dt[:-1, None]**2 * Theta[:-1] + 6 / dt[1:, None]**2 * Theta[1:]
           - delta_beta[:-1])
    rhs = np.vstack((omega[0], rhs))
    omega = omega.copy()
    solve_for_omega(dt, diag, A, B, rhs, omega)

    return omega


class Spline:
    """Spline interpolator of DCMs.

    The interpolation between the successive DCMs is done by rotation vectors
    which are cubic functions of time past from a previous time stamp.
    The coefficients are determined from the condition of continuity of the
    angular velocity and acceleration.

    This method can be considered as a cubic spline interpolation for the
    rotation vector, but now we want to establish the continuity of the angular
    velocity and acceleration, which are not equivalent to the first and second
    derivatives of the rotation vector.

    Parameters
    ----------
    t : array_like, shape (n,)
        Sequence of times. Must be strictly increasing.
    dcm : array_like, shape (n, 3, 3)
        Sequence of DCMs corresponding to `t`.

    Attributes
    ----------
    t : ndarray, shape (n,)
        Times given to the constructor.
    c : ndarray, (4, n, 3)
        Coefficients for the rotation vector cubic interpolants, ``coeff[0]``
        corresponds to the cubic term and ``coeff[3]`` corresponds to the
        constant term, which is zero identically.
    dcm : ndarray, shape (n, 3, 3)
        DCMs given to the constructor.
    rv : ndarray, shape (n - 1, 3)
        Full rotation vector on each interval.
    """
    MAX_ITER = 10

    def __init__(self, t, dcm):
        t = np.asarray(t, dtype=float)
        if t.ndim != 1:
            raise ValueError("`t` must be 1-dimensional.")

        if t.shape[0] < 2:
            raise ValueError("`t` must contain at least 2 elements.")

        dt = np.diff(t)
        if np.any(dt <= 0):
            raise ValueError("`t` must be strictly increasing.")

        dcm = np.asarray(dcm)
        if dcm.shape != (t.shape[0], 3, 3):
            raise ValueError("`C` is expected to have shape {}, but actually "
                             "has {}".format((t.shape[0], 3, 3), dcm.shape))

        Theta = to_rv(util.mm_prod(dcm[:-1], dcm[1:], at=True))
        omega = Theta / dt[:, None]
        omega = np.vstack((omega[0], omega))
        A = _dtheta_from_omega_matrix(Theta)
        B = _omega_from_dtheta_matrix(Theta)
        dTheta = util.mv_prod(A, omega[1:])

        if dt.shape[0] > 2:
            for iteration in range(self.MAX_ITER):
                omega_new = _solve_system(dt, A, B, Theta, dTheta, omega)
                delta = omega - omega_new
                omega = omega_new
                dTheta = util.mv_prod(A, omega[1:])

                if np.max(delta) < 1e-9:
                    break

        dt = dt[:, None]
        coeff = np.empty((4, t.shape[0] - 1, 3))
        coeff[3] = 0
        coeff[2] = omega[:-1]
        coeff[1] = (3 * Theta - 2 * dt * omega[:-1] - dt * dTheta) / dt**2
        coeff[0] = (-2 * Theta + dt * omega[:-1] + dt * dTheta) / dt**3

        self.theta = PPoly(coeff, t)
        self.C = dcm
        self.rv = Theta

    @property
    def c(self):
        return self.theta.c

    @property
    def t(self):
        return self.theta.x

    def __call__(self, t, order=0):
        """Compute interpolated values.

        Parameters
        ----------
        t : float or array_like, shape (n,)
            Times of interest.
        order : {0, 1, 2}, optional
            Order of differentiation:

                * 0 : return DCM
                * 1 : return the angular velocity
                * 2 : return the angular acceleration

        Returns
        -------
        ret : ndarray, shape (n, 3, 3) or (n, 3)
            DCM, angular velocity or acceleration depending on `order`.
        """
        if order not in [0, 1, 2]:
            raise ValueError("`order` must be 0, 1 or 2.")

        t = np.asarray(t, dtype=float)

        if t.ndim > 1:
            raise ValueError("`t` must be at most 1-dimensional.")

        if t.ndim == 0:
            t = np.array([t])
            single_t = True
        else:
            single_t = False

        theta = self.theta(t)
        if order == 0:
            order = np.argsort(t)
            reverse = np.empty_like(order)
            reverse[order] = np.arange(order.shape[0])

            t = t[order]
            theta = theta[order]
            index = np.searchsorted(self.t, t, side='right')
            index -= 1
            index[index < 0] = 0
            n_segments = self.t.shape[0] - 1
            index[index > n_segments - 1] = n_segments - 1

            dC = from_rv(theta)
            ret = util.mm_prod(self.C[index], dC)
            ret = ret[reverse]
        elif order == 1:
            dtheta = self.theta(t, 1)
            ret = _compute_omega(theta, dtheta)
        elif order == 2:
            dtheta = self.theta(t, 1)
            ddtheta = self.theta(t, 2)
            ret = _compute_beta(theta, dtheta, ddtheta)
        else:
            assert False

        if single_t:
            ret = np.squeeze(ret)

        return ret


def match_vectors(va, vb, w=None):
    """Compute a DCM for the closest match between the set of vectors.

    The matrix Cab is sought as the solution to the optimization problem::

        sum(w_i * ||Cab vb_i - va_i||^2, i=1, ..., n_points) -> min Cab,
        s. t. Cab being a right orthogonal matrix

    Vectors vb and va have unit norms. Moreover there should be at least 2
    noncollinear vectors in each set.

    Parameters
    ----------
    va, vb : array_like, shape (n_points, 3)
        Vectors measured in frames a and b. Normalized to the unit norm before
        processing.
    w : array_like, shape (n_points,)
        Weights for each observation. If None, assumed equal for each.

    Returns
    -------
    Cab : ndarray, shape (3, 3)
        Direction cosine matrix projecting from frame b to frame a.
    """
    va = np.asarray(va)
    vb = np.asarray(vb)

    va = va / np.linalg.norm(va, axis=1)[:, None]
    vb = vb / np.linalg.norm(vb, axis=1)[:, None]

    if w is None:
        B = va.T.dot(vb) / va.shape[0]
    else:
        w = np.asarray(w)
        B = va.T.dot(vb * w[:, None]) / np.sum(w)
    U, s, VT = svd(B, overwrite_a=True)

    return U.dot(np.diag([1, 1, det(U) * det(VT)])).dot(VT)

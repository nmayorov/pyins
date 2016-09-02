"""Create and manipulate direction cosine matrices."""
import numpy as np


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


def from_rv(rv):
    """Create a direction cosine matrix from a rotation vector.

    The direction of a rotation vector gives the axis of rotation and its
    magnitude gives the angle of rotation.

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


def from_hpr(h, p, r):
    """Create a direction dosine matrix from heading, pitch and roll angles.

    The sequence of elemental rotations is as follows::

           -heading   pitch   roll
        N ---------> ------> -----> B
              3         1       2

    Here N denotes the local level wander-azimuth frame and B denotes the body
    frame. The resulting DCM projects from B frame to N frame.

    Parameters
    ----------
    h, p, r : array_like, shape (3,) or (n, 3)
        Heading, pitch and roll.

    Returns
    -------
    dcm : ndarray, shape (3, 3) or (n, 3, 3)
        Direction cosine matrices.
    """
    h = np.deg2rad(h)
    p = np.deg2rad(p)
    r = np.deg2rad(r)

    ch = np.cos(h)
    sh = np.sin(h)
    cp = np.cos(p)
    sp = np.sin(p)
    cr = np.cos(r)
    sr = np.sin(r)

    dcm = np.empty((3, 3) + h.shape)
    dcm[0, 0] = ch * cr + sh * sp * sr
    dcm[0, 1] = sh * cp
    dcm[0, 2] = ch * sr - sh * sp * cr
    dcm[1, 0] = -sh * cr + ch * sp * sr
    dcm[1, 1] = ch * cp
    dcm[1, 2] = -sh * sr - ch * sp * cr
    dcm[2, 0] = -cp * sr
    dcm[2, 1] = sp
    dcm[2, 2] = cp * cr

    if dcm.ndim == 3:
        dcm = np.rollaxis(dcm, -1)

    return dcm


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
        Direction dosine matrices.

    Returns
    -------
    h, p, r : ndarray with shape (n,) or float
        Heading, pitch and roll.
    """
    dcm = np.asarray(dcm)
    if dcm.ndim not in [2, 3] or dcm.shape[-1] != 3 or dcm.shape[-2] != 3:
        raise ValueError('`dcm` has a wrong shape.')

    if dcm.ndim == 2:
        h, p, r = _to_hpr_single(dcm)
        return np.rad2deg(h), np.rad2deg(p), np.rad2deg(r)
    else:
        h, p, r = _to_hpr_array(dcm)

    return np.rad2deg(h), np.rad2deg(p), np.rad2deg(r)


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
    lat, lon : array_like
        Latitude and longitude.
    wan : array_like, optional
        Wander angle. Default is 0.

    Returns
    -------
    dcm : ndarray, shape (3,) or (n, 3)
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

    The latitude is withing [-90, 90], the longitude and the wander are
    within [-180, 180].

    If ``90 - abs(latitude) < np.rad2deg(1e-3)`` then the longitude is set to
    0 and the wander is computed accordingly.

    Parameters
    ----------
    dcm : array_like, shape (3, 3) or (n, 3, 3)
        Direction cosine matrices.

    Returns
    -------
    lat, lon, wan : ndarray with shape (n,) or float
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

    return np.rad2deg(lat), np.rad2deg(lon), np.rad2deg(wan)

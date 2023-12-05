"""Utility functions.

Functions
---------
.. autosummary::
    :toctree: generated

    mm_prod
    mm_prod_symmetric
    mv_prod
    skew_matrix
    compute_rms
    to_180_range
"""
import numpy as np
import pandas as pd


LLA_COLS = ['lat', 'lon', 'alt']
VEL_COLS = ['VN', 'VE', 'VD']
RPH_COLS = ['roll', 'pitch', 'heading']
RATE_COLS = ['rate_x', 'rate_y', 'rate_z']
GYRO_COLS = ['gyro_x', 'gyro_y', 'gyro_z']
ACCEL_COLS = ['accel_x', 'accel_y', 'accel_z']
THETA_COLS = ['theta_x', 'theta_y', 'theta_z']
DV_COLS = ['dv_x', 'dv_y', 'dv_z']
NED_COLS = ["north", "east", "down"]
TRAJECTORY_COLS = LLA_COLS + VEL_COLS + RPH_COLS
TRAJECTORY_ERROR_COLS = NED_COLS + VEL_COLS + RPH_COLS
XYZ_TO_INDEX = {'x': 0, 'y': 1, 'z': 2}
INDEX_TO_XYZ = {0: 'x', 1: 'y', 2: 'z'}


def mm_prod(a, b, at=False, bt=False):
    """Compute products of multiple matrices stored in a stack.

    Parameters
    ----------
    a, b : array_like with 2 or 3 dimensions
        Single matrix or stack of matrices. Matrices are stored in the two
        trailing dimensions. If one of the arrays is 2-D and another is
        3-D then broadcasting along the 0-th axis is applied.
    at, bt : bool, optional
        Whether to use transpose of `a` and `b` respectively.

    Returns
    -------
    ndarray
        Computed products.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    if a.ndim not in [2, 3]:
        raise ValueError("Wrong number of dimensions in `a`.")
    if b.ndim not in [2, 3]:
        raise ValueError("Wrong number of dimensions in `b`.")

    if at:
        if a.ndim == 3:
            a = np.transpose(a, (0, 2, 1))
        else:
            a = a.T
    if bt:
        if b.ndim == 3:
            b = np.transpose(b, (0, 2, 1))
        else:
            b = b.T

    return np.einsum("...ij,...jk->...ik", a, b)


def mm_prod_symmetric(a, b):
    """Compute symmetric product of stack of matrices.

    The result is ``a @ b @ a.T``.

    Parameters
    ----------
    a, b : array_like with 2 or 3 dimensions
        Single matrix or stack of matrices. Matrices are stored in the two
        trailing dimensions. If one of the arrays is 2-D and another is
        3-D then broadcasting along the 0-th axis is applied.

    Returns
    -------
    ndarray
        Computed products.
    """
    ab = mm_prod(a, b)
    return mm_prod(ab, a, bt=True)


def mv_prod(a, b, at=False):
    """Compute products of multiple matrices and vectors stored in a stack.

    Parameters
    ----------
    a : array_like with 2 or 3 dimensions
        Single matrix or stack of matrices. Matrices are stored in the two
        trailing dimensions.
    b : ndarray with 1 or 2 dimensions
        Single vector or stack of vectors. Vectors are stored in the trailing
        dimension.
    at : bool, optional
        Whether to use transpose of `a`.

    Returns
    -------
    ndarray
        Computed products.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    if a.ndim not in [2, 3]:
        raise ValueError("Wrong number of dimensions in `a`.")
    if b.ndim not in [1, 2]:
        raise ValueError("Wrong number of dimensions in `b`.")

    if at:
        if a.ndim == 3:
            a = np.transpose(a, (0, 2, 1))
        else:
            a = a.T

    return np.einsum("...ij,...j->...i", a, b)


def skew_matrix(vec):
    """Create a skew matrix corresponding to a vector.

    Parameters
    ----------
    vec : array_like, shape (3,) or (n, 3)
        Vector.

    Returns
    -------
    ndarray, shape (3, 3) or (n, 3)
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


def compute_rms(data):
    """Compute root-mean-square of data along 0 axis."""
    return np.mean(np.square(data), axis=0) ** 0.5


def to_180_range(angle):
    """Reduce angle in degrees to the range of [-180, 180]."""
    is_pandas = isinstance(angle, (pd.Series, pd.DataFrame))
    if not is_pandas:
        angle = np.asarray(angle)
    result = angle % 360
    if is_pandas or result.ndim > 0:
        result[result < -180] += 360
        result[result > 180] -= 360
    elif result < -180:
        result += 360
    elif result > 180:
        result -= 360
    return result


class Bunch(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join(['{}: {}'.format(k.rjust(m), type(v))
                              for k, v in self.items()])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())

"""Utility functions."""
import numpy as np


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
    ab : ndarray
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
        if a.ndim == 3:
            b = np.transpose(b, (0, 2, 1))
        else:
            b = b.T

    return np.einsum("...ij,...jk->...ik", a, b)


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
    ab : ndarray
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

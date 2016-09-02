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

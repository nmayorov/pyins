"""Alignment algorithms."""
from warnings import warn
import numpy as np
from scipy.linalg import svd, det
from . import earth
from . import dcm
from . import util


def align_wahba(dt, theta, dv, lat, VE=None, VN=None):
    """Estimate attitude matrix by solving Wahba's problem.

    This method is based on solving a least-squares problem for a direction
    cosine matrix A (originally formulated in [1]_)::

        L = sum(||A r_i - b_i||^2, i=1, ..., m) -> min A,
        s. t. A being a right orthogonal matrix.

    Here ``(r_i, b_i)`` are measurements of the same unit vectors in two
    frames.

    The application of this method to self alignment of INS is explained in
    [2]_. In this problem the vectors ``(r_i, b_i)`` are normalized velocity
    increments due to gravity. It is applicable to dynamic conditions as well,
    but in this case a full accuracy can be achieved only if velocity is
    provided.

    The optimization problem is solved using the most straightforward method
    based on SVD [3]_.

    Parameters
    ----------
    dt : double
        Sensors sampling period.
    theta, dv : array_like, shape (n_samples, 3)
        Rotation vectors and velocity increments computed from gyro and
        accelerometer readings after applying coning and sculling
        corrections.
    lat : float
        Latitude of the place.
    VE, VN : array_like with shape (n_samples + 1, 3) or None
        East and North velocity of the target. If None (default), it is
        assumed to be 0. See Notes for further details.

    Returns
    -------
    hpr : tuple of 3 floats
        Estimated heading, pitch and roll at the end of the alignment.
    P_align : ndarray, shape (3, 3)
        Covariance matrix of misalignment angles, commonly known as
        "phi-angle" in INS literature. Its values are measured in degrees
        squared. This matrix is estimated in a rather ad-hoc fashion, see
        Notes.

    Notes
    -----
    If the alignment takes place in dynamic conditions but velocities `VE`
    and `VN` are not provided, the alignment accuracy will be decreased (to
    some extent it will be reflected in `P_align`). Note that `VE` and `VN` are
    required with the same rate as inertial readings (and contain 1 more
    sample). It means that you usually have to do some sort of interpolation.
    In on-board implementation you just provide the last available velocity
    data from GPS and it will work fine.

    The paper [3]_ contains a recipe of computing the covariance matrix given
    that errors in measurements are independent, small and follow a statistical
    distribution with zero mean and known variance. In our case we estimate
    measurement error variance from the optimal value of the optimized function
    (see above). But as our errors are not independent and necessary small
    (nor they follow any reasonable distribution) we don't scale their
    variance by the number of observations (which is commonly done for the
    variance of an average value). Some experiments show that this approach
    gives reasonable values of `P_align`.

    Also note, that `P_align` accounts only for misalignment errors due
    to non-perfect alignment conditions. In addition to that, azimuth accuracy
    is always limited by gyro drifts and level accuracy is limited by the
    accelerometer biases. You should add these systematic uncertainties to the
    diagonal of `P_align`.

    References
    ----------
    .. [1] G. Wahba, "Problem 65–1: A Least Squares Estimate of Spacecraft
           Attitude", SIAM Review, 1965, 7(3), 409.
    .. [2] P. M. G. Silson, "Coarse Alignment of a Ship’s Strapdown Inertial
          Attitude Reference System Using Velocity Loci", IEEE Trans. Instrum.
          Meas., vol. 60, pp. 1930-1941, Jun. 2011.
    .. [3] F. L. Markley, "Attitude Determination using Vector Observations
           and the Singular Value Decomposition", The Journal of the
           Astronautical Sciences, Vol. 36, No. 3, pp. 245-258, Jul.-Sept.
           1988.
    """
    n_samples = theta.shape[0]
    Vg = np.zeros((n_samples + 1, 3))
    if VE is not None:
        Vg[:, 0] = VE
    if VN is not None:
        Vg[:, 1] = VN

    lat = np.deg2rad(lat)

    slat, clat = np.sin(lat), np.cos(lat)
    tlat = slat / clat
    re, rn = earth.principal_radii(lat)
    u = earth.RATE * np.array([0, clat, slat])
    g = np.array([0, 0, -earth.gravity(slat)])

    Cb0b = np.empty((n_samples + 1, 3, 3))
    Cg0g = np.empty((n_samples + 1, 3, 3))
    Cb0b[0] = np.identity(3)
    Cg0g[0] = np.identity(3)

    Vg_m = 0.5 * (Vg[1:] + Vg[:-1])

    rho = np.empty_like(Vg_m)
    rho[:, 0] = -Vg_m[:, 1] / rn
    rho[:, 1] = Vg_m[:, 0] / re
    rho[:, 2] = Vg_m[:, 0] / re * tlat

    for i in range(n_samples):
        Cg0g[i + 1] = Cg0g[i].dot(dcm.from_rv((rho[i] + u) * dt))
        Cb0b[i + 1] = Cb0b[i].dot(dcm.from_rv(theta[i]))

    f_g = np.cross(u, Vg) - g
    f_g0 = util.mv_prod(Cg0g, f_g)
    f_g0 = 0.5 * (f_g0[1:] + f_g0[:-1])
    f_g0 = np.vstack((np.zeros(3), f_g0))
    V_g0 = util.mv_prod(Cg0g, Vg) + dt * np.cumsum(f_g0, axis=0)

    V_b0 = np.cumsum(util.mv_prod(Cb0b[:-1], dv), axis=0)
    V_b0 = np.vstack((np.zeros(3), V_b0))

    k = n_samples // 2
    b = V_g0[k:2 * k] - V_g0[:k]
    b /= np.linalg.norm(b, axis=1)[:, None]

    r = V_b0[k:2 * k] - V_b0[:k]
    r /= np.linalg.norm(r, axis=1)[:, None]

    B = np.zeros((3, 3))
    for bi, ri in zip(b, r):
        B += np.outer(bi, ri)
    n_obs = b.shape[0]
    B /= n_obs

    U, s, VT = svd(B, overwrite_a=True)
    d = det(U) * det(VT)
    Cg0b0 = U.dot(np.diag([1, 1, d])).dot(VT)

    Cgb = Cg0g[-1].T.dot(Cg0b0).dot(Cb0b[-1])

    s[-1] *= d
    trace_s = np.sum(s)
    L = 1 - trace_s
    D = trace_s - s
    M = np.identity(3) - np.diag(s)
    if L < 0 or np.any(M < 0):
        L = max(L, 0)
        M[M < 0] = 0
        warn("Negative values encountered when estimating the covariance, "
             "they were set to zeros.")

    R = (L * M / n_obs) ** 0.5 / D
    R = U.dot(R)
    R = Cg0g[-1].T.dot(R)
    R = np.rad2deg(R)

    return dcm.to_hpr(Cgb), R.dot(R.T)

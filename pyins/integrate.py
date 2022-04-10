"""Compute a navigation solution by integration of inertial readings."""
import math
try:
    from numba import jit
except ImportError as error:
    print(error)
    print("Can't find numba.jit, execution may slow down dramatically.")
    def jit(fn):
        return fn
import numpy as np
import pandas as pd
from . import earth
from . import dcm
from ._integrate import integrate_fast, integrate_fast_stationary


def coning_sculling(gyro, accel, order=1, dt=None):
    """Apply coning and sculling corrections to inertial readings.

    The algorithm assumes a polynomial model for the angular velocity and the
    specific force, fitting coefficients by considering previous time
    intervals. The algorithm for a linear approximation is well known and
    described in [1]_ and [2]_.

    The accelerometer readings are also corrected for body frame rotation
    during a sampling period.

    Parameters
    ----------
    gyro : array_like, shape (n_readings, 3)
        Gyro readings.
    accel : array_like, shape (n_readings, 3)
        Accelerometer readings.
    order : {0, 1, 2}, optional
        Angular velocity and specific force polynomial model order.
        Note that 0 means not applying non-commutative corrections at all.
        For
        Default is 1.
    dt : float or None, optional
        If None (default), `gyro` and `accel` are assumed to contain increments.
        If float, it is interpreted as sampling rate of rate sensors.

    Returns
    -------
    theta : ndarray, shape (n_readings, 3)
        Estimated rotation vectors.
    dv : ndarray, shape (n_readings, 3)
        Estimated velocity increments.

    References
    ----------
    .. [1] P. G. Savage, "Strapdown Inertial Navigation Integration Algorithm
           Design Part 1: Attitude Algorithms", Journal of Guidance, Control,
           and Dynamics 1998, Vol. 21, no. 2.
    .. [2] P. G. Savage, "Strapdown Inertial Navigation Integration Algorithm
           Design Part 2: Velocity and Position Algorithms", Journal of
           Guidance, Control, and Dynamics 1998, Vol. 21, no. 2.
    """
    if order not in [0, 1, 2]:
        raise ValueError("`order` must be 1, 2 or 3.")

    gyro = np.asarray(gyro)
    accel = np.asarray(accel)

    if dt is not None:
        a_gyro = gyro[:-1]
        b_gyro = gyro[1:] - gyro[:-1]
        a_accel = accel[:-1]
        b_accel = accel[1:] - accel[:-1]
        alpha = (a_gyro + 0.5 * b_gyro) * dt
        dv = (a_accel + 0.5 * b_accel) * dt

        if order == 0:
            coning = 0
            sculling = 0
        else:
            coning = np.cross(a_gyro, b_gyro) * dt**2 / 12
            sculling = (np.cross(a_gyro, b_accel) +
                        np.cross(a_accel, b_gyro)) * dt**2/12

        return alpha + coning, dv + sculling + 0.5 * np.cross(alpha, dv)

    if order == 0:
        coning = 0
        sculling = 0
    elif order == 1:
        coning = np.vstack((np.zeros(3), np.cross(gyro[:-1], gyro[1:]) / 12))
        sculling = np.vstack((np.zeros(3),
                             (np.cross(gyro[:-1], accel[1:]) +
                              np.cross(accel[:-1], gyro[1:])) / 12))
    elif order == 2:
        coning = (-121 * np.cross(gyro[2:], gyro[1:-1]) +
                  31 * np.cross(gyro[2:], gyro[:-2]) -
                  np.cross(gyro[1:-1], gyro[:-2])) / 720
        sculling = (-121 * np.cross(gyro[2:], accel[1:-1]) +
                    31 * np.cross(gyro[2:], accel[:-2]) -
                    np.cross(gyro[1:-1], accel[:-2]) -
                    121 * np.cross(accel[2:], gyro[1:-1]) +
                    31 * np.cross(accel[2:], gyro[:-2]) -
                    np.cross(accel[1:-1], gyro[:-2])) / 720
        coning = np.vstack((np.zeros((2, 3)), coning))
        sculling = np.vstack((np.zeros((2, 3)), sculling))
    else:
        assert False

    rc = 0.5 * np.cross(gyro, accel)

    return gyro + coning, accel + sculling + rc


def integrate(dt, lat, lon, VE, VN, h, p, r, theta, dv, stamp=0):
    """Integrate inertial readings.

    The algorithm described in [1]_ and [2]_ is used with slight
    simplifications. The position is updated using the trapezoid rule.

    Parameters
    ----------
    dt : float
        Sensors sampling period.
    lat, lon : float
        Initial latitude and longitude.
    VE, VN : float
        Initial East and North velocity.
    h, p, r : float
        Initial heading, pitch and roll.
    theta, dv : array_like, shape (n_readings, 3)
        Rotation vectors and velocity increments computed from gyro and
        accelerometer readings after applying coning and sculling
        corrections.
    stamp : int, optional
        Stamp of the initial point.

    Returns
    -------
    traj : DataFrame
        Computed trajectory.

    See Also
    --------
    coning_sculling : Apply coning and sculling corrections.

    References
    ----------
    .. [1] P. G. Savage, "Strapdown Inertial Navigation Integration Algorithm
           Design Part 1: Attitude Algorithms", Journal of Guidance, Control,
           and Dynamics 1998, Vol. 21, no. 2.
    .. [2] P. G. Savage, "Strapdown Inertial Navigation Integration Algorithm
           Design Part 2: Velocity and Position Algorithms", Journal of
           Guidance, Control, and Dynamics 1998, Vol. 21, no. 2.
    """
    n_readings = theta.shape[0]
    lat_arr = np.empty(n_readings + 1)
    lon_arr = np.empty(n_readings + 1)
    VE_arr = np.empty(n_readings + 1)
    VN_arr = np.empty(n_readings + 1)
    Cnb_arr = np.empty((n_readings + 1, 3, 3))
    lat_arr[0] = np.deg2rad(lat)
    lon_arr[0] = np.deg2rad(lon)
    VE_arr[0] = VE
    VN_arr[0] = VN
    Cnb_arr[0] = dcm.from_hpr(h, p, r)

    earth_model = 0
    integrate_fast(dt, lat_arr, lon_arr, VE_arr, VN_arr, Cnb_arr, theta, dv,
                   earth_model)

    lat_arr = np.rad2deg(lat_arr)
    lon_arr = np.rad2deg(lon_arr)
    h, p, r = dcm.to_hpr(Cnb_arr)

    index = pd.Index(stamp + np.arange(n_readings + 1), name='stamp')
    traj = pd.DataFrame(index=index)
    traj['lat'] = lat_arr
    traj['lon'] = lon_arr
    traj['VE'] = VE_arr
    traj['VN'] = VN_arr
    traj['h'] = h
    traj['p'] = p
    traj['r'] = r

    return traj


class Integrator:
    """Class interface for integration of inertial readings.

    The algorithm described in [1]_ and [2]_ is used with slight simplifications.
    The position is updated using the trapezoid rule.

    Parameters
    ----------
    dt : float
        Sensors sampling period.
    lat, lon : float
        Initial latitude and longitude.
    VE, VN : float
        Initial East and North velocity.
    h, p, r : float
        Initial heading, pitch and roll.
    stamp : int, optional
        Time stamp of the initial point. Default is 0.

    Attributes
    ----------
    traj : DataFrame
        Computed trajectory so far.

    See Also
    --------
    coning_sculling : Apply coning and sculling corrections.

    References
    ----------
    .. [1] P. G. Savage, "Strapdown Inertial Navigation Integration Algorithm
           Design Part 1: Attitude Algorithms", Journal of Guidance, Control,
           and Dynamics 1998, Vol. 21, no. 2.
    .. [2] P. G. Savage, "Strapdown Inertial Navigation Integration Algorithm
           Design Part 2: Velocity and Position Algorithms", Journal of
           Guidance, Control, and Dynamics 1998, Vol. 21, no. 2.
    """
    INITIAL_SIZE = 10000

    def __init__(self, dt, lat, lon, VE, VN, h, p, r, stamp=0):
        self.dt = dt

        self.lat_arr = np.empty(self.INITIAL_SIZE)
        self.lon_arr = np.empty(self.INITIAL_SIZE)
        self.VE_arr = np.empty(self.INITIAL_SIZE)
        self.VN_arr = np.empty(self.INITIAL_SIZE)
        self.Cnb_arr = np.empty((self.INITIAL_SIZE, 3, 3))
        self.traj = None

        self._init_values = [lat, lon, VE, VN, h, p, r, stamp]
        self.reset()

    def reset(self):
        """Clear computed trajectory except the initial point."""
        lat, lon, VE, VN, h, p, r, stamp = self._init_values

        self.lat_arr[0] = np.deg2rad(lat)
        self.lon_arr[0] = np.deg2rad(lon)
        self.VE_arr[0] = VE
        self.VN_arr[0] = VN
        self.Cnb_arr[0] = dcm.from_hpr(h, p, r)

        self.traj = pd.DataFrame(index=pd.Index([stamp], name='stamp'))
        self.traj['lat'] = [lat]
        self.traj['lon'] = [lon]
        self.traj['VE'] = [VE]
        self.traj['VN'] = [VN]
        self.traj['h'] = [h]
        self.traj['p'] = [p]
        self.traj['r'] = [r]

    def integrate(self, theta, dv):
        """Integrate inertial readings.

        The integration continues from the last computed value.

        Parameters
        ----------
        theta, dv : array_like, shape (n_readings, 3)
            Rotation vectors and velocity increments computed from gyro and
            accelerometer readings after applying coning and sculling
            corrections.

        Returns
        -------
        traj_last : DataFrame
            Added chunk of the trajectory. It contains n_readings + 1 rows
            including the last point before `theta` and `dv` where integrated.
        """
        theta = np.asarray(theta)
        dv = np.asarray(dv)

        n_data = self.traj.shape[0]
        if n_data == 0:
            raise ValueError("No point to start integration from. "
                             "Call `init` first.")

        n_readings = theta.shape[0]
        size = self.lat_arr.shape[0]

        required_size = n_data + n_readings
        if required_size > self.lat_arr.shape[0]:
            new_size = max(2 * size, required_size)
            self.lat_arr.resize(new_size)
            self.lon_arr.resize(new_size)
            self.VE_arr.resize(new_size)
            self.VN_arr.resize(new_size)
            self.Cnb_arr.resize((new_size, 3, 3))

        earth_model = 0
        integrate_fast(self.dt, self.lat_arr, self.lon_arr, self.VE_arr,
                       self.VN_arr, self.Cnb_arr, theta, dv, earth_model,
                       offset=n_data-1)

        lat_arr = np.rad2deg(self.lat_arr[n_data: n_data + n_readings])
        lon_arr = np.rad2deg(self.lon_arr[n_data: n_data + n_readings])
        VE_arr = self.VE_arr[n_data: n_data + n_readings]
        VN_arr = self.VN_arr[n_data: n_data + n_readings]
        h, p, r = dcm.to_hpr(self.Cnb_arr[n_data: n_data + n_readings])

        index = pd.Index(self.traj.index[-1] + 1 + np.arange(n_readings),
                         name='stamp')
        traj = pd.DataFrame(index=index)
        traj['lat'] = lat_arr
        traj['lon'] = lon_arr
        traj['VE'] = VE_arr
        traj['VN'] = VN_arr
        traj['h'] = h
        traj['p'] = p
        traj['r'] = r

        self.traj = self.traj.append(traj)

        return self.traj.iloc[-n_readings - 1:]

    def _correct(self, x):
        i = self.traj.shape[0] - 1
        d_lat = x[1] / earth.R0
        d_lon = x[0] / (earth.R0 * np.cos(self.lat_arr[i]))
        self.lat_arr[i] -= d_lat
        self.lon_arr[i] -= d_lon

        phi = x[4:7]
        phi[2] += d_lon * np.sin(self.lat_arr[i])

        VE_new = self.VE_arr[i] - x[2]
        VN_new = self.VN_arr[i] - x[3]

        self.VE_arr[i] = VE_new - phi[2] * VN_new
        self.VN_arr[i] = VN_new + phi[2] * VE_new

        self.Cnb_arr[i] = dcm.from_rv(phi).dot(self.Cnb_arr[i])
        h, p, r = dcm.to_hpr(self.Cnb_arr[i])

        self.traj.iloc[-1] = [np.rad2deg(self.lat_arr[i]),
                              np.rad2deg(self.lon_arr[i]),
                              self.VE_arr[i], self.VN_arr[i], h, p, r]


def integrate_stationary(dt, lat, Cnb, theta, dv):
    n_readings = theta.shape[0]
    V_arr = np.empty((n_readings + 1, 3))
    Cnb_arr = np.empty((n_readings + 1, 3, 3))
    lat = np.deg2rad(lat)
    V_arr[0] = 0
    Cnb_arr[0] = Cnb
    integrate_fast_stationary(dt, lat, Cnb_arr, V_arr, theta, dv)

    return Cnb_arr, V_arr


@jit
def __dcm_from_rv_single(rv, out=None):
    rv1, rv2, rv3 = rv
    rv11 = rv1 * rv1
    rv12 = rv1 * rv2
    rv13 = rv1 * rv3
    rv22 = rv2 * rv2
    rv23 = rv2 * rv3
    rv33 = rv3 * rv3

    norm2 = rv11 + rv22 + rv33
    if norm2 > 1e-6:
        norm = norm2 ** 0.5
        k1 = math.sin(norm) / norm
        k2 = (1.0 - math.cos(norm)) / norm2
    else:
        norm4 = norm2 * norm2
        k1 = 1.0 - norm2 / 6.0 + norm4 / 120.0
        k2 = 0.5 - norm2 / 24.0 + norm4 / 720.0

    if out is None:
        out = np.empty((3,3))
    out[0, :] = 1.0 - k2*(rv33 + rv22), -k1*rv3 + k2*rv12, k1*rv2 + k2*rv13
    out[1, :] = k1*rv3 + k2*rv12, 1.0 - k2*(rv33 + rv11), -k1*rv1 + k2*rv23
    out[2, :] = -k1*rv2 + k2*rv13, k1*rv1 + k2*rv23, 1.0 - k2*(rv22 + rv11)
    return out


@jit
def __mv_dot3(A, b):
    b1, b2, b3 = b
    v1 = A[0, 0] * b1 + A[0, 1] * b2 + A[0, 2] * b3
    v2 = A[1, 0] * b1 + A[1, 1] * b2 + A[1, 2] * b3
    v3 = A[2, 0] * b1 + A[2, 1] * b2 + A[2, 2] * b3
    return v1, v2, v3


@jit
def __v_add3(a, b):
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


@jit
def __v_cross3(a, b):
    a1, a2, a3 = a
    b1, b2, b3 = b
    c1 = (-a3 * b2 + a2 * b3)
    c2 = ( a3 * b1 - a1 * b3)
    c3 = (-a2 * b1 + a1 * b2)

    return c1, c2, c3


@jit
def __gravity(lat, alt):
    sin_lat = math.sin(lat)
    sin_lat2 = sin_lat * sin_lat
    return (earth.GE * (1 + earth.F * sin_lat2)
            / (1 - earth.E2 * sin_lat2)**0.5
            * (1 - 2 * alt / earth.R0))


@jit
def _integrate_py_fast(lla, Vn, Cnb, theta, dv, dt, offset=0):
    """Mechanization in a rotating navigation frame.

    Parameters
    ----------
    lla : array_like, shape: (N, 3)
        Geodetic coordinates [rad].
    V : array_like, shape: (N, 3)
        Velocity in local level frame [m/s].
    Cnb : array_like, shape: (N, 3, 3)
        Direct cosine matrix.
    theta : array_like, shape: (N, 3)
        Rotation increments in body frame.
    dv : array_like, shape: (N, 3)
        Velocity increments in body frame.
    dt : float
        Time step.
    offset : int
        Offset index for intital conditions in arrays: lla, Vn, Cnb.
    """

    C = np.empty((3,3))
    dCn = np.empty((3,3))
    dCb = np.empty((3,3))
    V = np.empty(3)
    V_new = np.empty(3)

    for i in range(theta.shape[0]):
        j = i + offset

        lat, lon, alt = lla[j]
        sin_lat = math.sin(lat)
        sin_lat2 = sin_lat * sin_lat
        cos_lat = (1.0 - sin_lat2) ** 0.5
        tan_lat = sin_lat / cos_lat

        V = Vn[j]
        VE, VN, VU = V

        x = 1 - earth.E2 * sin_lat2
        re = earth.R0 / (x ** 0.5)
        rn = re * (1 - earth.E2) / x
        re += alt
        rn += alt

        u = (0, earth.RATE * cos_lat, earth.RATE * sin_lat)
        rho = (-VN / rn, VE / re, tan_lat * VE / re)
        omega = __v_add3(u, rho)
        w = __v_add3(u, omega)

        dv_n = __mv_dot3(Cnb[j], dv[i])
        corriolis1 = __v_cross3(w, V)
        corriolis2 = __v_cross3(omega, dv_n)

        VE_new = VE + (dv_n[0] - dt * (corriolis1[0] + 0.5 * corriolis2[0]))
        VN_new = VN + (dv_n[1] - dt * (corriolis1[1] + 0.5 * corriolis2[1]))
        VU_new = VU + (dv_n[2] - dt * (corriolis1[2] + 0.5 * corriolis2[2]))
        VU_new -= dt * __gravity(lat, (alt + 0.5 * dt * VU))

        Vn[j + 1] = VE_new, VN_new, VU_new

        VE = 0.5 * (VE + VE_new)
        VN = 0.5 * (VN + VN_new)
        VU = 0.5 * (VU + VU_new)

        rho = (-VN / rn, VE / re, tan_lat * VE / re)
        omega = __v_add3(u, rho)

        lla[j + 1, 0] = lat - dt * rho[0]
        lla[j + 1, 1] = lon + dt * rho[1] / cos_lat
        lla[j + 1, 2] = alt + dt * VU

        xi = (-dt * omega[0], -dt * omega[1], -dt * omega[2])
        dCn = __dcm_from_rv_single(xi, dCn)
        dCb = __dcm_from_rv_single(theta[i], dCb)

        C[:, 0] = __mv_dot3(dCn, Cnb[j, :, 0])
        C[:, 1] = __mv_dot3(dCn, Cnb[j, :, 1])
        C[:, 2] = __mv_dot3(dCn, Cnb[j, :, 2])

        Cnb[j+1, :, 0] = __mv_dot3(C, dCb[:, 0])
        Cnb[j+1, :, 1] = __mv_dot3(C, dCb[:, 1])
        Cnb[j+1, :, 2] = __mv_dot3(C, dCb[:, 2])

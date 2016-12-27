"""Compute a navigation solution by integration of inertial readings."""
import numpy as np
import pandas as pd
from . import earth
from . import dcm
from ._integrate import integrate_fast


def coning_sculling(gyro, accel, order=1):
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
        Default is 1.

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

    if earth.MODEL == 'WGS84':
        earth_model = 0
    elif earth.MODEL == 'PZ90':
        earth_model = 1
    else:
        raise ValueError("Set Earth model by calling `earth.set_model`.")
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

        if earth.MODEL == 'WGS84':
            earth_model = 0
        elif earth.MODEL == 'PZ90':
            earth_model = 1
        else:
            raise ValueError("Set Earth model by calling `earth.set_model`.")

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

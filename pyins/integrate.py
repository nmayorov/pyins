"""Compute a navigation solution by integration of inertial readings."""
import numpy as np
import pandas as pd
from . import dcm, earth
from ._integrate import integrate_fast


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


class Integrator:
    """Integrate inertial readings by strapdown algorithm.

    The algorithm described in [1]_ and [2]_ is used with slight
    simplifications. The position is updated using the trapezoid rule.

    Parameters
    ----------
    dt : float
        Sensors sampling period.
    lla : array_like, shape (3,)
        Initial latitude, longitude and altitude.
    velocity_n: array_like, shape (3,)
        Initial velocity in ENU frame.
    rph : array_like, shape (3,)
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
    TRAJECTORY_COLUMNS = ['lat', 'lon', 'alt', 'VE', 'VN', 'VU', 'r', 'p', 'h']
    INITIAL_SIZE = 10000

    def __init__(self, dt, lla, velocity_n, rph, stamp=0):
        self.dt = dt

        self.lla = np.empty((self.INITIAL_SIZE, 3))
        self.velocity_n = np.empty((self.INITIAL_SIZE, 3))
        self.Cnb = np.empty((self.INITIAL_SIZE, 3, 3))

        self.traj = None

        self._init_values = [lla, velocity_n, rph, stamp]
        self.reset()

    def reset(self):
        """Clear computed trajectory except the initial point."""
        lla, velocity_n, rph, stamp = self._init_values
        self.lla[0, :2] = np.deg2rad(lla[:2])
        self.lla[0, 2] = lla[2]
        self.velocity_n[0] = velocity_n
        self.Cnb[0] = dcm.from_rph(rph)
        self.traj = pd.DataFrame(
            data=np.atleast_2d(np.hstack((lla, velocity_n, rph))),
            columns=self.TRAJECTORY_COLUMNS,
            index=pd.Index([stamp], name='stamp'))

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
        n_readings = theta.shape[0]
        size = self.lla.shape[0]

        required_size = n_data + n_readings
        if required_size > size:
            new_size = max(2 * size, required_size)
            self.lla.resize((new_size, 3), refcheck=False)
            self.velocity_n.resize((new_size, 3), refcheck=False)
            self.Cnb.resize((new_size, 3, 3), refcheck=False)

        integrate_fast(self.dt, self.lla, self.velocity_n, self.Cnb,
                       theta, dv, offset=n_data-1)
        rph = dcm.to_rph(self.Cnb[n_data:n_data + n_readings])
        index = pd.Index(self.traj.index[-1] + 1 + np.arange(n_readings),
                         name='stamp')
        traj = pd.DataFrame(index=index)
        traj[['lat', 'lon']] = np.rad2deg(
            self.lla[n_data:n_data + n_readings, :2])
        traj['alt'] = self.lla[n_data:n_data + n_readings, 2]
        traj[['VE', 'VN', 'VU']] = self.velocity_n[n_data:n_data + n_readings]
        traj[['r', 'p', 'h']] = rph

        self.traj = self.traj.append(traj)

        return self.traj.iloc[-n_readings - 1:]

    def _correct(self, x):
        i = self.traj.shape[0] - 1
        d_lon = x[0] / (earth.R0 * np.cos(self.lla[i, 0]))
        self.lla[i, 0] -= x[1] / earth.R0
        self.lla[i, 1] -= d_lon
        self.lla[i, 2] -= x[2]

        phi = x[6:9]
        phi[2] += d_lon * np.sin(self.lla[i, 0])

        Ctp = dcm.from_rv(phi)
        self.velocity_n[i] = Ctp @ (self.velocity_n[i] - x[3:6])

        self.Cnb[i] = Ctp @ self.Cnb[i]
        rph = dcm.to_rph(self.Cnb[i])

        self.traj.iloc[-1] = np.hstack((np.rad2deg(self.lla[i, :2]),
                                        self.lla[i, 2],
                                        self.velocity_n[i],
                                        rph))

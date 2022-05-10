"""Strapdown INS integration algorithms."""
import numpy as np
import pandas as pd
from . import transform
from ._integrate import integrate_fast


def compute_theta_and_dv(gyro, accel, dt=None):
    """Compute attitude and velocity increments from IMU readings.

    This function transforms raw gyro and accelerometer readings into
    rotation vectors and velocity increments by applying coning and sculling
    corrections and accounting for IMU rotation during a sampling period.

    The algorithm assumes a linear model for the angular velocity and the
    specific force described in [1]_ and [2]_.

    Parameters
    ----------
    gyro : array_like, shape (n_readings, 3)
        Gyro readings.
    accel : array_like, shape (n_readings, 3)
        Accelerometer readings.
    dt : float or None, optional
        If None (default), `gyro` and `accel` are assumed to contain integral
        increments. Float is interpreted as the sampling rate of rate sensors.

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
    gyro = np.asarray(gyro)
    accel = np.asarray(accel)

    if dt is not None:
        a_gyro = gyro[:-1]
        b_gyro = gyro[1:] - gyro[:-1]
        a_accel = accel[:-1]
        b_accel = accel[1:] - accel[:-1]
        alpha = (a_gyro + 0.5 * b_gyro) * dt
        dv = (a_accel + 0.5 * b_accel) * dt

        coning = np.cross(a_gyro, b_gyro) * dt**2 / 12
        sculling = (np.cross(a_gyro, b_accel) +
                    np.cross(a_accel, b_gyro)) * dt**2/12

        return alpha + coning, dv + sculling + 0.5 * np.cross(alpha, dv)

    coning = np.vstack((np.zeros(3), np.cross(gyro[:-1], gyro[1:]) / 12))
    sculling = np.vstack((np.zeros(3),
                          (np.cross(gyro[:-1], accel[1:]) +
                           np.cross(accel[:-1], gyro[1:])) / 12))

    return gyro + coning, accel + sculling + 0.5 * np.cross(gyro, accel)


class StrapdownIntegrator:
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
        Initial velocity in NED frame.
    rph : array_like, shape (3,)
        Initial heading, pitch and roll.
    stamp : int, optional
        Time stamp of the initial point. Default is 0.

    Attributes
    ----------
    trajectory : DataFrame
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
    TRAJECTORY_COLUMNS = ['lat', 'lon', 'alt', 'VN', 'VE', 'VD',
                          'roll', 'pitch', 'heading']
    INITIAL_SIZE = 10000

    def __init__(self, dt, lla, velocity_n, rph, stamp=0):
        self.dt = dt

        self.lla = np.empty((self.INITIAL_SIZE, 3))
        self.velocity_n = np.empty((self.INITIAL_SIZE, 3))
        self.Cnb = np.empty((self.INITIAL_SIZE, 3, 3))

        self.trajectory = None

        self._init_values = [lla, velocity_n, rph, stamp]
        self.reset()

    def reset(self):
        """Clear computed trajectory except the initial point."""
        lla, velocity_n, rph, stamp = self._init_values
        self.lla[0] = lla
        self.velocity_n[0] = velocity_n
        self.Cnb[0] = transform.mat_from_rph(rph)
        self.trajectory = pd.DataFrame(
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

        n_data = self.trajectory.shape[0]
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
        rph = transform.mat_to_rph(self.Cnb[n_data:n_data + n_readings])
        index = pd.Index(self.trajectory.index[-1] + 1 + np.arange(n_readings),
                         name='stamp')
        trajectory = pd.DataFrame(index=index)
        trajectory[['lat', 'lon', 'alt']] = self.lla[n_data:
                                                     n_data + n_readings]
        trajectory[['VN', 'VE', 'VD']] = self.velocity_n[n_data:
                                                         n_data + n_readings]
        trajectory[['roll', 'pitch', 'heading']] = rph

        self.trajectory = pd.concat([self.trajectory, trajectory])

        return self.trajectory.iloc[-n_readings - 1:]

    def get_state(self):
        """Get current integrator state.

        Returns
        -------
        trajectory_point : pd.Series
            Trajectory point.
        """
        return self.trajectory.iloc[-1]

    def set_state(self, trajectory_point):
        """Set (overwrite) the current integrator state.

        Parameters
        ----------
        trajectory_point : pd.Series
            Trajectory point.
        """
        i = len(self.trajectory) - 1
        self.lla[i] = trajectory_point[['lat', 'lon', 'alt']]
        self.velocity_n[i] = trajectory_point[['VN', 'VE', 'VD']]
        self.Cnb[i] = transform.mat_from_rph(
            trajectory_point[['roll', 'pitch', 'heading']])
        self.trajectory.iloc[-1] = trajectory_point

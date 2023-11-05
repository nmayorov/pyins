"""Strapdown INS integration algorithms."""
import numpy as np
import pandas as pd
from . import transform
from .util import LLA_COLS, RPH_COLS, VEL_COLS, GYRO_COLS, ACCEL_COLS
from ._integrate import integrate_fast


def compute_theta_and_dv(imu, sensor_type):
    """Compute attitude and velocity increments from IMU readings.

    This function transforms raw gyro and accelerometer readings into
    rotation vectors and velocity increments by applying coning and sculling
    corrections and accounting for IMU rotation during a sampling period.

    The algorithm assumes a linear model for the angular velocity and the
    specific force described in [1]_ and [2]_.

    Parameters
    ----------
    imu : pd.DataFrame
        IMU data.
    sensor_type : 'rate' or 'increment'

    Returns
    -------
    increments : pd.DataFrame
        Angle and velocity increments.

    References
    ----------
    .. [1] P. G. Savage, "Strapdown Inertial Navigation Integration Algorithm
           Design Part 1: Attitude Algorithms", Journal of Guidance, Control,
           and Dynamics 1998, Vol. 21, no. 2.
    .. [2] P. G. Savage, "Strapdown Inertial Navigation Integration Algorithm
           Design Part 2: Velocity and Position Algorithms", Journal of
           Guidance, Control, and Dynamics 1998, Vol. 21, no. 2.
    """
    if sensor_type not in ['rate', 'increment']:
        raise ValueError("`sensor_type` must be either 'rate' or 'increment'")

    gyro = imu[GYRO_COLS].values
    accel = imu[ACCEL_COLS].values    
    dt = np.diff(imu.index).reshape(-1, 1)
    if sensor_type == 'increment':
        gyro_increment = gyro[1:]
        accel_increment = accel[1:]
        coning = np.cross(gyro[:-1], gyro[1:]) / 12
        sculling = (np.cross(gyro[:-1], accel[1:]) +
                    np.cross(accel[:-1], gyro[1:])) / 12
    elif sensor_type == 'rate':
        dt = np.diff(imu.index).reshape(-1, 1)
        a_gyro = gyro[:-1]
        b_gyro = gyro[1:] - gyro[:-1]
        a_accel = accel[:-1]
        b_accel = accel[1:] - accel[:-1]
        gyro_increment = (a_gyro + 0.5 * b_gyro) * dt
        accel_increment = (a_accel + 0.5 * b_accel) * dt
        coning = np.cross(a_gyro, b_gyro) * dt ** 2 / 12
        sculling = (np.cross(a_gyro, b_accel) +
                    np.cross(a_accel, b_gyro)) * dt ** 2 / 12
    else:
        assert False

    theta = gyro_increment + coning
    dv = accel_increment + sculling + 0.5 * np.cross(gyro_increment, accel_increment)

    return pd.DataFrame(data=np.hstack((dt, theta, dv)), index=imu.index[1:],
                        columns=['dt', 'theta_x', 'theta_y', 'theta_z',
                                 'dv_x', 'dv_y', 'dv_z'])


class Integrator:
    """Integrate inertial readings by strapdown algorithm.

    The algorithm described in [1]_ and [2]_ is used with slight
    simplifications. The position is updated using the trapezoid rule.

    Parameters
    ----------
    dt : float
        Sensors sampling period.
    trajectory_point : pd.Series
        Initial trajectory point.
    with_altitude : bool, optional
        Whether to compute altitude and vertical velocity. Default is True.
        If False, then vertical velocity is set to zero and altitude is kept
        as constant.

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
    INITIAL_SIZE = 10000

    def __init__(self, trajectory_point, with_altitude=True):
        self.with_altitude = with_altitude
        if not with_altitude:
            trajectory_point.VD = 0.0

        self.lla = np.empty((self.INITIAL_SIZE, 3))
        self.velocity_n = np.empty((self.INITIAL_SIZE, 3))
        self.Cnb = np.empty((self.INITIAL_SIZE, 3, 3))

        self.trajectory = None

        self._init_values = trajectory_point
        self.reset()

    def reset(self):
        """Clear computed trajectory except the initial point."""
        self.lla[0] = self._init_values[LLA_COLS]
        self.velocity_n[0] = self._init_values[VEL_COLS]
        self.Cnb[0] = transform.mat_from_rph(self._init_values[RPH_COLS])
        self.trajectory = self._init_values.to_frame().transpose(copy=True)
        self.trajectory.index.name = 'stamp'

    def integrate(self, increments):
        """Integrate inertial readings.

        The integration continues from the last computed value.

        Parameters
        ----------
        increments : pd.DataFrame
            Rotation vectors and velocity increments computed from gyro and
            accelerometer readings after applying coning and sculling
            corrections.

        Returns
        -------
        traj_last : DataFrame
            Added chunk of the trajectory. It contains n_readings + 1 rows
            including the last point before `theta` and `dv` where integrated.
        """
        theta = np.ascontiguousarray(increments[['theta_x', 'theta_y', 'theta_z']])
        dv = np.ascontiguousarray(increments[['dv_x', 'dv_y', 'dv_z']])

        n_data = self.trajectory.shape[0]
        n_readings = theta.shape[0]
        size = self.lla.shape[0]

        required_size = n_data + n_readings
        if required_size > size:
            new_size = max(2 * size, required_size)
            self.lla.resize((new_size, 3), refcheck=False)
            self.velocity_n.resize((new_size, 3), refcheck=False)
            self.Cnb.resize((new_size, 3, 3), refcheck=False)

        integrate_fast(np.asarray(increments.dt), self.lla, self.velocity_n, self.Cnb,
                       theta, dv, n_data - 1, self.with_altitude)
        rph = transform.mat_to_rph(self.Cnb[n_data:n_data + n_readings])
        trajectory = pd.DataFrame(
            np.hstack([self.lla[n_data : n_data + n_readings],
                       self.velocity_n[n_data : n_data + n_readings],
                       rph]),
            index=increments.index, columns=LLA_COLS + VEL_COLS + RPH_COLS
        )
        self.trajectory = pd.concat([self.trajectory, trajectory])

        return self.trajectory.iloc[-n_readings - 1:]

    def get_time(self):
        return self.trajectory.index[-1]

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
        self.lla[i] = trajectory_point[LLA_COLS]
        self.velocity_n[i] = trajectory_point[VEL_COLS]
        self.Cnb[i] = transform.mat_from_rph(trajectory_point[RPH_COLS])
        self.trajectory.iloc[-1] = trajectory_point

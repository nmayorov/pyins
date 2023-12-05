"""Strapdown INS integration algorithms.

This module provides implementation of the classic "strapdown algorithm" to obtain
position, velocity and attitude by integration of IMU readings.
The implementation follows [1]_ and [2]_ with some simplifications.

Functions
---------
.. autosummary::
    :toctree: generated/

    compute_increments_from_imu

Classes
-------
.. autosummary::
    :toctree: generated/

    Integrator

References
----------
.. [1] P. G. Savage, "Strapdown Inertial Navigation Integration Algorithm
       Design Part 1: Attitude Algorithms", Journal of Guidance, Control,
       and Dynamics 1998, Vol. 21, no. 2.
.. [2] P. G. Savage, "Strapdown Inertial Navigation Integration Algorithm
       Design Part 2: Velocity and Position Algorithms", Journal of
       Guidance, Control, and Dynamics 1998, Vol. 21, no. 2.
"""
import numpy as np
import pandas as pd
from . import transform
from .util import LLA_COLS, RPH_COLS, VEL_COLS, GYRO_COLS, ACCEL_COLS, TRAJECTORY_COLS
from ._numba_integrate import integrate


def compute_increments_from_imu(imu, sensor_type):
    """Compute attitude and velocity increments from IMU readings.

    This function transforms raw gyro and accelerometer readings into
    rotation vectors and velocity increments by applying coning and sculling
    corrections and accounting for IMU rotation during a sampling period.

    The algorithm assumes a linear model for the angular velocity and the
    specific force.

    The number of returned increments is always one less than the number
    of IMU readings, even for increment-type sensors (by convention). It means that
    for increment sensors you need to supply an additional "before" sample.
    The function `pyins.sim.generate_imu` already does that.

    Parameters
    ----------
    imu : Imu
        Dataframe with IMU data.
    sensor_type : 'rate' or 'increment'
        IMU type.

    Returns
    -------
    Increments
        DataFrame containing attitude and velocity increments with one less row than
        the passed `imu`.
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
    """Strapdown INS integration algorithm.

    The position is updated using the trapezoid rule.

    Parameters
    ----------
    pva : Pva
        Initial position-velocity-attitude.
    with_altitude : bool, optional
        Whether to compute altitude and vertical velocity. Default is True.
        If False, then vertical velocity is set to zero and altitude is kept
        as constant.

    Attributes
    ----------
    trajectory : Trajectory
        Computed trajectory so far.
    """
    INITIAL_SIZE = 10000

    def __init__(self, pva, with_altitude=True):
        self.initial_pva = pva.copy()
        self.with_altitude = with_altitude
        if not with_altitude:
            self.initial_pva.VD = 0.0

        self.lla = np.empty((self.INITIAL_SIZE, 3))
        self.velocity_n = np.empty((self.INITIAL_SIZE, 3))
        self.mat_nb = np.empty((self.INITIAL_SIZE, 3, 3))

        self.lla[0] = self.initial_pva[LLA_COLS]
        self.velocity_n[0] = self.initial_pva[VEL_COLS]
        self.mat_nb[0] = transform.mat_from_rph(self.initial_pva[RPH_COLS])
        self.trajectory = self.initial_pva.to_frame().transpose(copy=True)
        self.trajectory.index.name = 'time'

    def _integrate(self, increments, mode):
        n_data = len(self.trajectory)
        n_readings = len(increments)
        size = len(self.lla)

        required_size = n_data + n_readings
        if required_size > size:
            new_size = max(2 * size, required_size)
            self.lla.resize((new_size, 3), refcheck=False)
            self.velocity_n.resize((new_size, 3), refcheck=False)
            self.mat_nb.resize((new_size, 3, 3), refcheck=False)

        theta = np.ascontiguousarray(increments[['theta_x', 'theta_y', 'theta_z']])
        dv = np.ascontiguousarray(increments[['dv_x', 'dv_y', 'dv_z']])
        integrate(np.asarray(increments.dt), self.lla, self.velocity_n,
                  self.mat_nb, theta, dv, n_data - 1, self.with_altitude)
        rph = transform.mat_to_rph(self.mat_nb[n_data : n_data + n_readings])
        trajectory = pd.DataFrame(
            np.hstack([self.lla[n_data : n_data + n_readings],
                       self.velocity_n[n_data : n_data + n_readings],
                       rph]),
            index=increments.index, columns=TRAJECTORY_COLS)

        if mode == 'integrate':
            self.trajectory = pd.concat([self.trajectory, trajectory])
            return self.trajectory.iloc[-n_readings - 1:]
        elif mode == 'predict':
            return trajectory
        else:
            assert False

    def integrate(self, increments):
        """Update trajectory by given inertial increments.

        The integration continues from the last computed values.

        Parameters
        ----------
        increments : Increments
            Attitude and velocity increments computed from gyro and accelerometer
            readings.

        Returns
        -------
        Trajectory
            Added chunk of the trajectory including the last point before
            `increments` were integrated.
        """
        return self._integrate(increments, 'integrate')

    def predict(self, increment):
        """Predict position-velocity-increment given a single increment.

        The stored trajectory is not updated.

        Parameters
        ----------
        increment : Series
            Single increment as a row of Increments DataFrame.

        Returns
        -------
        Pva
            Predicted position-velocity-attitude.
        """
        return self._integrate(increment.to_frame().transpose(), 'predict').iloc[0]

    def get_time(self):
        """Get time of the latest position-velocity-attitude."""
        return self.trajectory.index[-1]

    def get_pva(self):
        """Get the latest position-velocity-attitude."""
        return self.trajectory.iloc[-1]

    def set_pva(self, pva):
        """Set (overwrite) the latest position-velocity-attitude."""
        i = len(self.trajectory) - 1
        self.lla[i] = pva[LLA_COLS]
        self.velocity_n[i] = pva[VEL_COLS]
        self.mat_nb[i] = transform.mat_from_rph(pva[RPH_COLS])
        self.trajectory.iloc[-1] = pva

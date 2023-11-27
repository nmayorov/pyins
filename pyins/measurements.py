"""Measurement models for navigation Kalman filters.

In context of inertial navigation measurements are obtained from sensors other than
IMU. In a Kalman filter a measurement is processed by forming a difference between
the predicted and the measured vectors and linearly relating it to the error vector::

    z = Z_ins - Z = H @ x + v

Where

    - ``Z`` - measured vector
    - ``Z_ins`` - predicted vector using the current INS state
    - ``z`` - innovation vector
    - ``x`` - error state vector
    - ``H`` - measurement Jacobian
    - ``v`` - noise vector, typically assumed to have zero mean and known variance

The module provides a base class `Measurement` which abstracts this concept and
implementations for common measurements.

Refer to [1]_ and [2]_ for the discussion of measurements in navigation and in
Kalman filtering in general.

Classes
-------
.. autosummary::
    :toctree: generated/

    Measurement
    Position
    NedVelocity
    BodyVelocity

References
----------
.. [1] P. D. Groves, "Principles of GNSS, Inertial, and Multisensor Integrated
       Navigation Systems", 2nd edition
.. [2] P\. S\. Maybeck, "Stochastic Models, Estimation and Control", volume 1
"""
import numpy as np
from . import transform
from .util import LLA_COLS, VEL_COLS, RPH_COLS, RATE_COLS


class Measurement:
    """Base class for measurement models.

    To introduce a new measurement `compute_matrices` method needs to be implemented.
    See Also section contains links to already implemented measurements.

    Parameters
    ----------
    data : DataFrame
        Measured values as a DataFrame indexed by time.

    Attributes
    ----------
    data : DataFrame
        Data saved from the constructor.

    See Also
    --------
    Position
    NedVelocity
    BodyVelocity
    """
    def __init__(self, data):
        self.data = data

    def compute_matrices(self, time, pva, error_model):
        """Compute matrices for a single linearized measurement.

        It must compute the linearized measurement model (z, H, R) at a given time.
        If the measurement is not available at the given `time`, it must return
        None.

        Parameters
        ----------
        time : float
            Time.
        pva : Pva
            Position-velocity-attitude estimates from INS at `time`.
        error_model : `pyins.error_model.InsErrorModel`
            InsErrorModel instance. The method must account  for
            ``error_model.with_altitude`` as appropriate.

        Returns
        -------
        z : ndarray, shape (n_obs,)
            Observation vector. A difference between the value derived from `pva`
            and an observed value.
        H : ndarray, shape (n_obs, n_states)
            Observation model matrix. It relates the vector `z` to the INS error states.
        R : ndarray, shape (n_obs, n_obs)
            Covariance matrix of the measurement error.
        """
        raise NotImplementedError


class Position(Measurement):
    """Measurement of latitude, longitude and altitude (from GNSS or any other source).

    Parameters
    ----------
    data : DataFrame
        Must be indexed by time and contain columns 'lat', 'lon' and 'alt' columns for
        latitude, longitude and altitude.
    sd : float
        Measurement accuracy in meters.
    imu_to_antenna_b : array_like with shape (3,) or None, optional
        Vector from IMU to antenna (measurement point) expressed in body
        frame. If None, assumed to be zero.

    Attributes
    ----------
    data : DataFrame
        Data saved from the constructor.
    """
    def __init__(self, data, sd, imu_to_antenna_b=None):
        super(Position, self).__init__(data[LLA_COLS])
        self.R = sd**2 * np.eye(3)
        self.imu_to_antenna_b = imu_to_antenna_b

    def compute_matrices(self, time, pva, error_model):
        if time not in self.data.index:
            return None

        z = transform.compute_lla_difference(pva[LLA_COLS],
                                             self.data.loc[time, LLA_COLS])
        if self.imu_to_antenna_b is not None:
            mat_nb = transform.mat_from_rph(pva[RPH_COLS])
            z += mat_nb @ self.imu_to_antenna_b
        H = error_model.position_error_jacobian(pva, self.imu_to_antenna_b)
        R = self.R
        if not error_model.with_altitude:
            z = z[:2]
            H = H[:2]
            R = R[:2, :2]
        return z, H, R


class NedVelocity(Measurement):
    """Measurement of velocity resolved in NED frame (from GNSS or any other source).

    Parameters
    ----------
    data : DataFrame
        Must be indexed by time and contain 'VN', 'VE' and 'VD' columns.
    sd : float
        Measurement accuracy in m/s.
    imu_to_antenna_b : array_like with shape (3,) or None, optional
        Vector from IMU to antenna (measurement point) expressed in body
        frame. If None (default), assumed to be zero.

    Attributes
    ----------
    data : DataFrame
        Data saved from the constructor.
    """
    def __init__(self, data, sd, imu_to_antenna_b=None):
        super(NedVelocity, self).__init__(data[VEL_COLS])
        self.R = sd**2 * np.eye(3)
        self.imu_to_antenna_b = imu_to_antenna_b

    def compute_matrices(self, time, pva, error_model):
        if time not in self.data.index:
            return None

        z = pva[VEL_COLS] - self.data.loc[time, VEL_COLS]
        if self.imu_to_antenna_b is not None and all(col in pva for col in RATE_COLS):
            mat_nb = transform.mat_from_rph(pva[RPH_COLS])
            z += mat_nb @ np.cross(pva[RATE_COLS], self.imu_to_antenna_b)
        H = error_model.ned_velocity_error_jacobian(pva)
        R = self.R
        if not error_model.with_altitude:
            z = z[:2]
            H = H[:2]
            R = R[:2, :2]

        return z, H, R


class BodyVelocity(Measurement):
    """Measurement of velocity resolved in body frame (from odometry, radar, etc.)

    Parameters
    ----------
    data : DataFrame
        Must be indexed by time and contain columns 'VX', 'VY' and 'VZ'.
    sd : float
        Measurement accuracy in m/s.

    Attributes
    ----------
    data : DataFrame
        Data saved from the constructor.
    """
    def __init__(self, data, sd):
        super(BodyVelocity, self).__init__(data[['VX', 'VY', 'VZ']])
        self.R = sd**2 * np.eye(3)

    def compute_matrices(self, time, pva, error_model):
        if time not in self.data.index:
            return None

        mat_nb = transform.mat_from_rph(pva[RPH_COLS])
        z = mat_nb.T @ pva[VEL_COLS] - self.data.loc[time, ['VX', 'VY', 'VZ']]
        H = error_model.body_velocity_error_jacobian(pva)
        return z, H, self.R

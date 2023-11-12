"""Observation models for navigation Kalman filters.

In context of inertial navigation observations are obtained from sensors other than
IMU. In a Kalman filter an observation is processed by forming a difference between
the predicted and the measured vectors and linearly relating it to the error vector::

    z = Z_ins - Z = H @ x + v

Where

    - ``Z`` - measured vector
    - ``Z_ins`` - predicted vector using the current INS state
    - ``z`` - innovation vector
    - ``x`` - error state vector
    - ``H`` - measurement Jacobian
    - ``v`` - noise vector. typically assumed to have zero mean and known variance

The module provides a base class `Observation` which conceptualises this concept and
implementations for common observations.

Refer to [1]_ and [2]_ for the discussion of the observations in navigation and in
Kalman filtering in general.

References
----------
.. [1] P. D. Groves, "Principles of GNSS, Inertial, and Multisensor Integrated
       Navigation Systems, Second Edition"
.. [2] P. S. Maybeck, "Stochastic Models, Estimation and Control, Volume 1"
"""
import numpy as np
from . import error_model, transform
from .util import LLA_COLS, VEL_COLS, RPH_COLS, RATE_COLS


class Observation:
    """Base class for observation models.

    Documentation is given to explain how you can implement a new observation
    model. All you need to do is to implement `compute_obs` function. See Also
    section contains links to already implemented models.

    Parameters
    ----------
    data : DataFrame
        Observed values as a DataFrame indexed by time.

    Attributes
    ----------
    data : DataFrame
        Data saved from the constructor.

    See Also
    --------
    PositionObs
    NedVelocityObs
    BodyVelocityObs
    """
    def __init__(self, data):
        self.data = data

    def compute_matrices(self, time, pva):
        """Compute matrices for a single linearized observation.

        It must compute the observation model (z, H, R) at a given time stamp.
        If the observation is not available at the given `time`, it must return
        None.

        Parameters
        ----------
        time : float
            Time.
        pva : Pva
            Position-velocity-attitude estimates from INS at `time`.

        Returns
        -------
        z : ndarray, shape (n_obs,)
            Observation vector. A difference between the value derived from `pva`
            and an observed value.
        H : ndarray, shape (n_obs, 9)
            Observation model matrix. It relates the vector `z` to the INS error states.
        R : ndarray, shape (n_obs, n_obs)
            Covariance matrix of the observation error.
        """
        raise NotImplementedError


class PositionObs(Observation):
    """Observation of latitude, longitude and altitude (from GNSS or any other source).

    Parameters
    ----------
    data : DataFrame
        Must be indexed by time and contain columns 'lat', 'lon' and `alt` columns for
        latitude, longitude and altitude.
    sd : float
        Measurement accuracy in meters.
    imu_to_antenna_b : array_like, shape (3,) or None, optional
        Vector from IMU to antenna (measurement point) expressed in body
        frame. If None, assumed to be zero.

    Attributes
    ----------
    data : DataFrame
        Data saved from the constructor.
    """
    def __init__(self, data, sd, imu_to_antenna_b=None):
        super(PositionObs, self).__init__(data)
        self.R = sd**2 * np.eye(3)
        self.imu_to_antenna_b = imu_to_antenna_b

    def compute_matrices(self, time, pva):
        if time not in self.data.index:
            return None

        z = transform.compute_lla_difference(pva[LLA_COLS],
                                             self.data.loc[time, LLA_COLS])
        if self.imu_to_antenna_b is not None:
            mat_nb = transform.mat_from_rph(pva[RPH_COLS])
            z += mat_nb @ self.imu_to_antenna_b

        H = error_model.position_error_jacobian(pva, self.imu_to_antenna_b)

        return z, H, self.R


class NedVelocityObs(Observation):
    """Observation of velocity resolved in NED frame (typically from GNSS).

    Parameters
    ----------
    data : DataFrame
        Must be indexed by time and contain 'VN', 'VE' and 'VD' columns.
    sd : float
        Measurement accuracy in m/s.
    imu_to_antenna_b : array_like, shape (3,) or None, optional
        Vector from IMU to antenna (measurement point) expressed in body
        frame. If None (default), assumed to be zero.

    Attributes
    ----------
    data : DataFrame
        Data saved from the constructor.
    """
    def __init__(self, data, sd, imu_to_antenna_b=None):
        super(NedVelocityObs, self).__init__(data)
        self.R = sd**2 * np.eye(3)
        self.imu_to_antenna_b = imu_to_antenna_b

    def compute_matrices(self, time, pva):
        if time not in self.data.index:
            return None

        z = pva[VEL_COLS] - self.data.loc[time, VEL_COLS]
        if self.imu_to_antenna_b is not None and all(col in pva for col in RATE_COLS):
            mat_nb = transform.mat_from_rph(pva[RPH_COLS])
            z += mat_nb @ np.cross(pva[RATE_COLS], self.imu_to_antenna_b)
        H = error_model.ned_velocity_error_jacobian(pva)

        return z, H, self.R


class BodyVelocityObs(Observation):
    """Observation of velocity resolved in body frame.

    Parameters
    ----------
    data : DataFrame
        Must be indexed by `time` and contain columns 'VX', 'VY' and 'VZ'.
    sd : float
        Measurement accuracy in m/s.

    Attributes
    ----------
    data : DataFrame
        Data saved from the constructor.
    """
    def __init__(self, data, sd):
        super(BodyVelocityObs, self).__init__(data)
        self.R = sd**2 * np.eye(3)

    def compute_matrices(self, time, pva):
        if time not in self.data.index:
            return None

        mat_nb = transform.mat_from_rph(pva[RPH_COLS])
        z = mat_nb.T @ pva[VEL_COLS] - self.data.loc[time, ['VX', 'VY', 'VZ']]
        H = error_model.body_velocity_error_jacobian(pva)
        return z, H, self.R

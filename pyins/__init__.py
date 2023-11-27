"""pyins: Inertial Navigation System algorithms in Python.

Type naming conventions
-----------------------
Most of the data operated within the package are represented as pandas DataFrame or
Series. The same kinds of data have the same set of columns (or index in case of
Series). In this sense we define the following "types":

    - `Trajectory` - DataFrame containing INS trajectory with columns 'lat', 'lon',
      'alt', 'VN', 'VE', 'VD', 'roll', 'pitch', 'heading'. These comprise geodetic
      position, velocity resolved in North-East-Down frame and Euler angles for
      the attitude
    - `Pva` - Series representing position-velocity-attitude - a single row of
      `Trajectory`
    - `Imu` - DataFrame containing IMU measurements with columns 'gyro_x', 'gyro_y',
      'gyro_z', 'accel_x, 'accel_y', 'accel_z'
    - `Increments` - DataFrame containing attitude and velocity increments computed from
      raw IMU measurements. Have columns 'dt' - time delta for the increment,
      'theta_x, 'theta_y', 'theta_z' - components of the rotation vector,
      'dv_x', 'dv_y', 'dv_z' - inertial velocity increments
    - `TrajectoryError` - DataFrame with INS trajectory errors with columns 'north,
      'east', 'down' for the position error in meters resolved in North-East-Down
      frame, 'VN', 'VE', 'VD' for the errors of North-East-Down velocity components,
      'roll', 'pitch, 'heading' for the Euler angle errors. Data frames for trajectory
      parameters standard deviations have the same columns
    - `PvaError` - Series representing errors of position-velocity-attitude - a single
      row of `TrajectoryError`

All data (`Trajectory`, `Imu`, measurements) are indexed by time in seconds measured by
a common clock. This achieves time synchronization of IMU and measurements and allow
comparison between trajectories and other states.

Variable naming convention
--------------------------
Geometric vectors and rotation matrices are associated with frames of reference.
A vector ``vec`` expressed in a frame ``a`` is typically denoted as ``vec_a``.
A rotation matrix projecting from frame ``b`` to frame ``a`` is denoted as ``mat_ab``.

The following one-letter notation for the frames of reference is used:

    - e - Earth-centered Earth-fixed frame (ECEF)
    - i - Earth-centered inertial frame (ECI)
    - n - North-East-Down local horizon frame
    - b - frame associated with IMU axes also known as "body frame"

Refer to [1]_ for detailed definitions of the aforementioned frames.

Units of measurement
--------------------
Generally all parameters are measured in International System of Units.
Gyro readings and associated quantities (noise, bias, etc.) are based on radians
(like rad/s, etc.) Angle parameters (latitude, longitude, roll, pitch, heading) are
measured in degrees.

A continuous white noise intensity is expressed as root of power spectral density
(root PSD). Refer to [2]_ for the discussion of continuous white noise process and its
power spectral density.

Modules
-------
.. autosummary::
   :toctree: generated/

   earth
   error_model
   filters
   inertial_sensor
   kalman
   measurements
   sim
   strapdown
   transform
   util

References
----------
.. [1] P. D. Groves, "Principles of GNSS, Inertial, and Multisensor Integrated
       Navigation Systems", 2nd edition
.. [2] P\. S\. Maybeck, "Stochastic Models, Estimation and Control", volume 1
"""
from . import (earth, error_model, filters, inertial_sensor, kalman, measurements, sim,
               strapdown, transform, util)

__version__ = "1.0"

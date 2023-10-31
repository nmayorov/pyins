"""Strapdown sensors simulator."""
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline
from scipy._lib._util import check_random_state
from . import earth, transform, util


def _compute_increment_readings(dt, a, b, c, d, e):
    ab = np.cross(a, b)
    ac = np.cross(a, c)
    bc = np.cross(b, c)

    omega = np.empty((8,) + a.shape)
    omega[0] = a
    omega[1] = 2 * b
    omega[2] = 3 * c - 0.5 * ab
    omega[3] = -ac + np.cross(a, ab) / 6
    omega[4] = -0.5 * bc + np.cross(a, ac) / 3 + np.cross(b, ab) / 6
    omega[5] = np.cross(a, bc) / 6 + np.cross(b, ac) / 3 + np.cross(c, ab) / 6
    omega[6] = np.cross(b, bc) / 6 + np.cross(c, ac) / 3
    omega[7] = np.cross(c, bc) / 6

    gyros = 0
    for k in reversed(range(8)):
        gyros += omega[k] / (k + 1)
        gyros *= dt

    ad = np.cross(a, d)
    ae = np.cross(a, e)
    bd = np.cross(b, d)
    be = np.cross(b, e)
    cd = np.cross(c, d)
    ce = np.cross(c, e)

    f = np.empty((8,) + d.shape)
    f[0] = d
    f[1] = e - ad
    f[2] = -ae - bd + 0.5 * np.cross(a, ad)
    f[3] = -be - cd + 0.5 * (np.cross(a, ae + bd) + np.cross(b, ad))
    f[4] = -ce + 0.5 * (np.cross(a, be + cd) + np.cross(b, ae + bd) +
                        np.cross(c, ad))
    f[5] = 0.5 * (np.cross(a, ce) + np.cross(b, be + cd) +
                  np.cross(c, ae + bd))
    f[6] = 0.5 * (np.cross(b, ce) + np.cross(c, be + cd))
    f[7] = 0.5 * np.cross(c, ce)

    accels = 0
    for k in reversed(range(8)):
        accels += f[k] / (k + 1)
        accels *= dt

    return gyros, accels


def from_position(dt, lla, rph, sensor_type='increment'):
    """Generate inertial readings given position and attitude.

    Parameters
    ----------
    dt : float
        Time step.
    lla : array_like, shape (n_points, 3)
        Time series of latitude, longitude and altitude.
    rph : array_like, shape (n_points, 3)
        Time series of roll, pitch and heading angles.
    sensor_type: 'increment' or 'rate', optional
        Type of sensor to generate. If 'increment' (default), then integrals
        over sampling intervals are generated (in rad and m/s).
        If 'rate', then instantaneous rate values are generated
        (in rad/s and /m/s/s).

    Returns
    -------
    trajectory : DataFrame
        Trajectory. Contains n_points rows.
    gyro : ndarray, shape (n_points - 1, 3) or (n_points, 3)
        Gyro readings.
    accel : ndarray, shape (n_points - 1, 3) or (n_points, 3)
        Accelerometer readings.
    """
    if sensor_type not in ['rate', 'increment']:
        raise ValueError("`sensor_type` must be 'rate' or 'increment'.")

    lla = np.asarray(lla, dtype=float)
    rph = np.asarray(rph, dtype=float)
    n_points = len(lla)
    time = dt * np.arange(n_points)

    lla_inertial = lla.copy()
    lla_inertial[:, 1] += np.rad2deg(earth.RATE) * time
    Cin = transform.mat_en_from_ll(lla_inertial[:, 0], lla_inertial[:, 1])

    R = transform.lla_to_ecef(lla_inertial)
    v_s = CubicSpline(time, R).derivative()
    v = v_s(time)

    V = v.copy()
    V[:, 0] += earth.RATE * R[:, 1]
    V[:, 1] -= earth.RATE * R[:, 0]
    V = util.mv_prod(Cin, V, at=True)

    Cnb = transform.mat_from_rph(rph)
    Cib = util.mm_prod(Cin, Cnb)

    Cib_spline = RotationSpline(time, Rotation.from_matrix(Cib))
    g = earth.gravitation_ecef(lla_inertial)

    if sensor_type == 'rate':
        gyros = Cib_spline(time, 1)
        accels = util.mv_prod(Cib, v_s(time, 1) - g, at=True)
    elif sensor_type == 'increment':
        a = Cib_spline.interpolator.c[2]
        b = Cib_spline.interpolator.c[1]
        c = Cib_spline.interpolator.c[0]

        a_s = v_s.derivative()
        d = a_s.c[1] - g[:-1]
        e = a_s.c[0] - np.diff(g, axis=0) / dt

        d = util.mv_prod(Cib[:-1], d, at=True)
        e = util.mv_prod(Cib[:-1], e, at=True)

        gyros, accels = _compute_increment_readings(dt, a, b, c, d, e)
    else:
        assert False

    trajectory = pd.DataFrame(index=np.arange(time.shape[0]))
    trajectory[['lat', 'lon', 'alt']] = lla
    trajectory[['VN', 'VE', 'VD']] = V
    trajectory[['roll', 'pitch', 'heading']] = rph
    return trajectory, gyros, accels


def from_velocity(dt, lla0, velocity_n, rph, sensor_type='increment'):
    """Generate inertial readings given velocity and attitude.

    Parameters
    ----------
    dt : float
        Time step.
    lla0 : array_like, shape (3,)
        Initial values of latitude, longitude and altitude.
    velocity_n : array_like, shape (n_points, 3)
        Time series of East, North and vertical velocity components.
    rph : array_like, shape (n_points, 3)
        Time series of roll, pitch and heading.
    sensor_type: 'increment' or 'rate', optional
        Type of sensor to generate. If 'increment' (default), then integrals
        over sampling intervals are generated (in rad and m/s).
        If 'rate', then instantaneous rate values are generated
        (in rad/s and /m/s/s).

    Returns
    -------
    trajectory : DataFrame
        Trajectory. Contains n_points rows.
    gyro : ndarray, shape (n_points - 1, 3)
        Gyro readings.
    accel : ndarray, shape (n_points - 1, 3)
        Accelerometer readings.
    """
    MAX_ITER = 3
    ACCURACY = 0.01

    velocity_n = np.asarray(velocity_n, dtype=float)
    rph = np.asarray(rph, dtype=float)
    n_points = len(velocity_n)
    time = np.arange(n_points) * dt

    VU_spline = CubicSpline(time, -velocity_n[:, 2])
    alt_spline = VU_spline.antiderivative()
    alt = lla0[2] + alt_spline(time)

    lat = lat0 = np.deg2rad(lla0[0])
    for iteration in range(MAX_ITER):
        rn, _, _ = earth.principal_radii(np.rad2deg(lat), alt)
        dlat_spline = CubicSpline(time, velocity_n[:, 0] / rn)
        lat_spline = dlat_spline.antiderivative()
        lat_new = lat_spline(time) + lat0
        delta = (lat - lat_new) * rn
        lat = lat_new
        if np.all(np.abs(delta) < ACCURACY):
            break

    _, _, rp = earth.principal_radii(np.rad2deg(lat), alt)
    dlon_spline = CubicSpline(time, velocity_n[:, 1] / rp)
    lon_spline = dlon_spline.antiderivative()

    lla = np.empty((n_points, 3))
    lla[:, 0] = np.rad2deg(lat)
    lla[:, 1] = lla0[1] + np.rad2deg(lon_spline(time))
    lla[:, 2] = alt

    return from_position(dt, lla, rph, sensor_type)


def sinusoid_velocity_motion(dt, total_time, lla0, velocity_mean,
                             velocity_change_amplitude=0,
                             velocity_change_period=60,
                             velocity_change_phase_offset=[0, 90, 0],
                             sensor_type='increment'):
    """Generate trajectory with NED velocity changing as sinus.

    The NED velocity changes as::

        V = V_mean + V_ampl * sin(2 * pi * t / period + phase_offset)

    Roll is set to zero, pitch and heading angles are computed with zero
    lateral and vertical velocity assumptions.

    Parameters
    ----------
    dt : float
        Time step.
    total_time : float
        Total motion time.
    lla0 : array_like, shape (3,)
        Initial latitude, longitude and altitude.
    velocity_mean : array_like, shape (3,)
        Mean velocity resolved in NED.
    velocity_change_amplitude : array_like, optional
        Velocity change amplitude. Default is 0.
    velocity_change_period : float, optional
        Period of sinusoidal velocity change in seconds. Default is 60.
    velocity_change_phase_offset : array_like, shape (3,), optional
        Phase offset for sinusoid part in degrees. Default is [0, 90, 0]
        which will create an ellipse for latitude-longitude trajectory when
        the mean velocity is zero.
    sensor_type: 'increment' or 'rate', optional
        Type of sensor to generate. If 'increment' (default), then integrals
        over sampling intervals are generated (in rad and m/s).
        If 'rate', then instantaneous rate values are generated
        (in rad/s and /m/s/s).

    Returns
    -------
    trajectory : DataFrame
        Trajectory. Contains n_points rows.
    gyro : ndarray, shape (n_points - 1, 3)
        Gyro readings.
    accel : ndarray, shape (n_points - 1, 3)
        Accelerometer readings.
    """
    time = np.arange(0, total_time, dt)
    phase = 2 * np.pi * time[:, None] / velocity_change_period + \
            np.deg2rad(velocity_change_phase_offset)
    velocity_n = (np.atleast_2d(velocity_mean) +
                  np.atleast_2d(velocity_change_amplitude) * np.sin(phase))
    rph = np.zeros_like(velocity_n)
    rph[:, 1] = np.rad2deg(np.arctan2(
        velocity_n[:, 2], np.hypot(velocity_n[:, 0], velocity_n[:, 1])))
    rph[:, 2] = np.rad2deg(np.arctan2(velocity_n[:, 1], velocity_n[:, 0]))
    return from_velocity(dt, lla0, velocity_n, rph, sensor_type)


def generate_position_observations(trajectory, error_sd, rng=None):
    rng = check_random_state(rng)
    error = error_sd * rng.randn(len(trajectory), 3)
    lla = transform.perturb_lla(trajectory[['lat', 'lon', 'alt']],
                                error)
    return pd.DataFrame(data=lla, index=trajectory.index,
                        columns=['lat', 'lon', 'alt'])


def generate_ned_velocity_observations(trajectory, error_sd, rng=None):
    rng = check_random_state(rng)
    error = error_sd * rng.randn(len(trajectory), 3)
    velocity_n = trajectory[['VN', 'VE', 'VD']] + error
    return pd.DataFrame(data=velocity_n, index=trajectory.index,
                        columns=['VN', 'VE', 'VD'])


def generate_body_velocity_observations(trajectory, error_sd, rng=None):
    rng = check_random_state(rng)
    error = error_sd * rng.randn(len(trajectory), 3)
    Cnb = transform.mat_from_rph(trajectory[['roll', 'pitch', 'heading']])
    velocity_b = (util.mv_prod(Cnb, trajectory[['VN', 'VE', 'VD']], at=True)
                  + error)
    return pd.DataFrame(data=velocity_b, index=trajectory.index,
                        columns=['VX', 'VY', 'VZ'])


def perturb_navigation_state(lla, velocity_n, rph, position_sd, velocity_sd,
                             level_sd, azimuth_sd, rng=None):
    rng = check_random_state(rng)
    return (transform.perturb_lla(lla, position_sd * rng.randn(3)),
            velocity_n + velocity_sd * rng.randn(3),
            rph + [level_sd, level_sd, azimuth_sd] * rng.randn(3))


def _align_matrix(align_angles):
    align_angles = np.deg2rad(align_angles)
    theta1 = align_angles[0]
    phi1 = align_angles[1]
    theta2 = align_angles[2]
    phi2 = align_angles[3]
    theta3 = align_angles[4]
    phi3 = align_angles[5]

    return np.array([
        [np.cos(theta1), np.sin(theta1) * np.cos(phi1),
         np.sin(theta1) * np.sin(phi1)],
        [np.sin(theta2) * np.sin(phi2), np.cos(theta2),
         np.sin(theta2) * np.cos(phi2)],
        [np.sin(theta3) * np.cos(phi3), np.sin(theta3) * np.sin(phi3),
         np.cos(theta3)]
    ])


def _apply_errors(sensor_type, dt, readings, scale_error, scale_asym, align,
                  bias, noise, rng):
    out = util.mv_prod(align, readings)
    if sensor_type == 'increment':
        out += bias * dt
        out += noise * dt ** 0.5 * rng.randn(*readings.shape)
    elif sensor_type == 'rate':
        out += bias
        out += noise * dt ** -0.5 * rng.randn(*readings.shape)
    else:
        assert False
    scale = 1 + scale_error + scale_asym * np.sign(out)
    out *= scale
    return out


class ImuErrors:
    def __init__(self, transform=None, bias=None, noise=None, bias_walk=None,
                 rng=None):
        if transform is None:
            transform = np.identity(3)
        if bias is None:
            bias = 0
        if bias_walk is None:
            bias_walk = 0
        if noise is None:
            noise = 0

        self.transform = np.asarray(transform)
        self.bias = np.asarray(bias)
        self.bias_walk = np.asarray(bias_walk)
        self.noise = np.asarray(noise)
        self.rng = check_random_state(rng)

    @classmethod
    def from_inertial_sensor_model(cls, inertial_sensor_model, rng=None):
        rng = check_random_state(rng)
        transform = (np.eye(3) +
                     inertial_sensor_model.scale_misal * rng.randn(3, 3))
        if inertial_sensor_model.bias is None:
            bias = None
        else:
            bias = inertial_sensor_model.bias * rng.randn(3)
        return cls(transform, bias, inertial_sensor_model.noise,
                   inertial_sensor_model.bias_walk, rng)

    def apply(self, readings, dt, sensor_type):
        readings = np.asarray(readings)
        if readings.ndim != 2 or readings.shape[1] != 3:
            raise ValueError("`readings` must be a (n, 3) array")

        bias = self.bias + self.bias_walk * np.cumsum(
            self.rng.randn(*readings.shape), axis=0) * dt**0.5
        result = util.mv_prod(self.transform, readings)
        if sensor_type == 'rate':
            result += bias
            result += self.noise * dt**-0.5 * self.rng.randn(*readings.shape)
        elif sensor_type == 'increment':
            result += bias * dt
            result += self.noise * dt**0.5 * self.rng.randn(*readings.shape)
        else:
            raise ValueError(
                "`sensor_type` must be either 'rate' or 'increment ")
        return result

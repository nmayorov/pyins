"""Strapdown sensors simulator."""
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PPoly
from scipy.linalg import solve_banded
from scipy.spatial.transform import Rotation, RotationSpline
from scipy._lib._util import check_random_state
from . import earth, transform, util


#: Degrees per hour to radians per second (SI units)
DH2SI = np.pi / (180 * 3600)
#: Radians per second (SI units) to degrees per hour.
SI2DH = 1 / DH2SI
#: Degrees per root hour to radians per root second (SI units).
DRH2SI = np.pi / (180 * 60)
#: Radians per root second (SI units) to degrees per root hour.
SI2DRH = 1 / DRH2SI


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
    traj : DataFrame
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

    traj = pd.DataFrame(index=np.arange(time.shape[0]))
    traj[['lat', 'lon', 'alt']] = lla
    traj[['VN', 'VE', 'VD']] = V
    traj[['roll', 'pitch', 'heading']] = rph
    return traj, gyros, accels


class _QuadraticSpline(PPoly):
    def __init__(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        n = x.shape[0]
        dx = np.diff(x)
        dy = np.diff(y, axis=0)
        dxr = dx.reshape([dx.shape[0]] + [1] * (y.ndim - 1))

        c = np.empty((3, n - 1) + y.shape[1:])
        if n > 2:
            A = np.ones((2, n))
            b = np.empty((n,) + y.shape[1:])
            b[0] = 0
            b[1:] = 2 * dy / dxr
            s = solve_banded((1, 0), A, b, overwrite_ab=True, overwrite_b=True,
                             check_finite=False)
            c[0] = np.diff(s, axis=0) / (2 * dxr)
            c[1] = s[:-1]
            c[2] = y[:-1]
        else:
            c[0] = 0
            c[1] = dy / dxr
            c[2] = y[:-1]

        super(_QuadraticSpline, self).__init__(c, x)


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
    traj : DataFrame
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

    VU_spline = _QuadraticSpline(time, -velocity_n[:, 2])
    alt_spline = VU_spline.antiderivative()
    alt = lla0[2] + alt_spline(time)

    lat = lat0 = np.deg2rad(lla0[0])
    for iteration in range(MAX_ITER):
        rn, _, _ = earth.principal_radii(np.rad2deg(lat), alt)
        dlat_spline = _QuadraticSpline(time, velocity_n[:, 0] / rn)
        lat_spline = dlat_spline.antiderivative()
        lat_new = lat_spline(time) + lat0
        delta = (lat - lat_new) * rn
        lat = lat_new
        if np.all(np.abs(delta) < ACCURACY):
            break

    _, _, rp = earth.principal_radii(np.rad2deg(lat), alt)
    dlon_spline = _QuadraticSpline(time, velocity_n[:, 1] / rp)
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
    traj : DataFrame
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
    rph[:, 2] = np.rad2deg(np.arctan2(velocity_n[:, 0], velocity_n[:, 1]))
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
                  bias, noise):
    out = util.mv_prod(align, readings)
    if sensor_type == 'increment':
        out += bias * dt
        out += noise * dt ** 0.5 * np.random.randn(*readings.shape)
    elif sensor_type == 'rate':
        out += bias
        out += noise * dt ** -0.5 * np.random.randn(*readings.shape)
    else:
        assert False
    scale = 1 + scale_error + scale_asym * np.sign(out)
    out *= scale
    return out


class ImuErrors:
    def __init__(self, gyro_scale_error=None, gyro_scale_asym=None,
                 gyro_align=None, gyro_bias=None, gyro_noise=None,
                 accel_scale_error=None, accel_scale_asym=None,
                 accel_align=None, accel_bias=None, accel_noise=None):
        if gyro_scale_error is None:
            gyro_scale_error = 0
        else:
            gyro_scale_error = np.asarray(gyro_scale_error)

        if gyro_scale_asym is None:
            gyro_scale_asym = 0
        else:
            gyro_scale_asym = np.asarray(gyro_scale_asym)

        if gyro_align is None:
            gyro_align = np.eye(3)
        else:
            gyro_align = _align_matrix(gyro_align)

        if gyro_bias is None:
            gyro_bias = 0
        else:
            gyro_bias = np.asarray(gyro_bias)

        if gyro_noise is None:
            gyro_noise = 0
        else:
            gyro_noise = np.asarray(gyro_noise)

        if accel_scale_error is None:
            accel_scale_error = 0
        else:
            accel_scale_error = np.asarray(accel_scale_error)

        if accel_scale_asym is None:
            accel_scale_asym = 0
        else:
            accel_scale_asym = np.asarray(accel_scale_asym)

        if accel_align is None:
            accel_align = np.eye(3)
        else:
            accel_align = _align_matrix(accel_align)

        if accel_bias is None:
            accel_bias = 0
        else:
            accel_bias = np.asarray(accel_bias)

        if accel_noise is None:
            accel_noise = 0
        else:
            accel_noise = np.asarray(accel_noise)

        U, S, VT = np.linalg.svd(gyro_align)
        Cmb = np.dot(U, VT)
        gyro_align_mars = gyro_align.dot(Cmb.T)
        accel_align_mars = accel_align.dot(Cmb.T)

        self.gyro_scale_error = gyro_scale_error
        self.gyro_scale_asym = gyro_scale_asym
        self.gyro_align = gyro_align
        self.gyro_bias = gyro_bias
        self.gyro_noise = gyro_noise
        self.accel_scale_error = accel_scale_error
        self.accel_scale_asym = accel_scale_asym
        self.accel_align = accel_align
        self.accel_bias = accel_bias
        self.accel_noise = accel_noise

        self.gyro_align_mars = gyro_align_mars
        self.accel_align_mars = accel_align_mars
        self.Cmb = Cmb

    def apply(self, dt, gyro, accel, sensor_type='increment'):
        gyro_out = _apply_errors(sensor_type, dt, gyro,
                                 self.gyro_scale_error, self.gyro_scale_asym,
                                 self.gyro_align, self.gyro_bias,
                                 self.gyro_noise)
        accel_out = _apply_errors(sensor_type, dt, accel,
                                  self.accel_scale_error, self.accel_scale_asym,
                                  self.accel_align, self.accel_bias,
                                  self.accel_noise)

        return gyro_out, accel_out

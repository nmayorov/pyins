"""Strapdown sensors simulator."""
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PPoly
from scipy.linalg import solve_banded
from . import dcm, earth, coord, util


def _compute_readings(dt, a, b, c, d, e):
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


def from_position(dt, lat, lon, alt, h, p, r):
    """Generate inertial readings given position and attitude.

    Parameters
    ----------
    dt : float
        Time step.
    lat, lon, alt : array_like, shape (n_points,)
        Time series of latitude, longitude and altitude.
    h, p, r : array_like, shape (n_points,)
        Time series of heading, pitch and roll angles.

    Returns
    -------
    traj : DataFrame
        Trajectory. Contains n_points rows.
    gyro : ndarray, shape (n_points - 1, 3)
        Gyro readings.
    accel : ndarray, shape (n_points - 1, 3)
        Accelerometer readings.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    alt = np.asarray(alt, dtype=float)
    h = np.asarray(h, dtype=float)
    p = np.asarray(p, dtype=float)
    r = np.asarray(r, dtype=float)
    n_points = lat.shape[0]

    time = dt * np.arange(n_points)
    lat_inertial = lat.copy()
    lon_inertial = lon.copy()
    lon_inertial += np.rad2deg(earth.RATE) * time
    Cin = dcm.from_llw(lat_inertial, lon_inertial)

    R = coord.lla_to_ecef(lat_inertial, lon_inertial, alt)
    v_s = CubicSpline(time, R).derivative()
    v = v_s(time)

    V = v.copy()
    V[:, 0] += earth.RATE * R[:, 1]
    V[:, 1] -= earth.RATE * R[:, 0]
    V = util.mv_prod(Cin, V, at=True)

    Cnb = dcm.from_hpr(h, p, r)
    Cib = util.mm_prod(Cin, Cnb)

    Cib_spline = dcm.Spline(time, Cib)
    a = Cib_spline.c[2]
    b = Cib_spline.c[1]
    c = Cib_spline.c[0]

    g = earth.gravitation_ecef(lat_inertial, lon_inertial, alt)
    a_s = v_s.derivative()
    d = a_s.c[1] - g[:-1]
    e = a_s.c[0] - np.diff(g, axis=0) / dt

    d = util.mv_prod(Cib[:-1], d, at=True)
    e = util.mv_prod(Cib[:-1], e, at=True)

    gyros, accels = _compute_readings(dt, a, b, c, d, e)

    traj = pd.DataFrame(index=np.arange(time.shape[0]))
    traj['lat'] = lat
    traj['lon'] = lon
    traj['alt'] = alt
    traj['VE'] = V[:, 0]
    traj['VN'] = V[:, 1]
    traj['VU'] = V[:, 2]
    traj['h'] = h
    traj['p'] = p
    traj['r'] = r

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


def from_velocity(dt, lat0, lon0, alt0, VE, VN, VU, h, p, r):
    """Generate inertial readings given velocity and attitude.

    Parameters
    ----------
    dt : float
        Time step.
    lat0, lon0, alt0 : float
        Initial values of latitude, longitude and altitude.
    VE, VN, VU : array_like, shape (n_points,)
        Time series of East, North and vertical velocity components.
    h, p, r : array_like, shape (n_points,)
        Time series of heading, pitch and roll angles.

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

    VE = np.asarray(VE, dtype=float)
    VN = np.asarray(VN, dtype=float)
    VU = np.asarray(VU, dtype=float)
    h = np.asarray(h, dtype=float)
    p = np.asarray(p, dtype=float)
    r = np.asarray(r, dtype=float)
    n_points = VE.shape[0]
    time = np.arange(n_points) * dt

    VU_spline = _QuadraticSpline(time, VU)
    alt_spline = VU_spline.antiderivative()
    alt = alt0 + alt_spline(time)

    lat0 = np.deg2rad(lat0)
    lon0 = np.deg2rad(lon0)
    lat = lat0

    for iteration in range(MAX_ITER):
        _, rn = earth.principal_radii(np.sin(lat))
        rn += alt
        dlat_spline = _QuadraticSpline(time, VN / rn)
        lat_spline = dlat_spline.antiderivative()
        lat_new = lat_spline(time) + lat0
        delta = (lat - lat_new) * rn
        lat = lat_new
        if np.all(np.abs(delta) < ACCURACY):
            break

    re, _ = earth.principal_radii(np.sin(lat))
    re += alt
    dlon_spline = _QuadraticSpline(time, VE / (re * np.cos(lat)))
    lon_spline = dlon_spline.antiderivative()
    lon = lon_spline(time) + lon0

    lat = np.rad2deg(lat)
    lon = np.rad2deg(lon)
    lon_inertial = lon + np.rad2deg(earth.RATE) * time
    Cin = dcm.from_llw(lat, lon_inertial)

    v = np.vstack((VE, VN, VU)).T
    v = util.mv_prod(Cin, v)
    R = coord.lla_to_ecef(lat, lon_inertial, alt)
    v[:, 0] -= earth.RATE * R[:, 1]
    v[:, 1] += earth.RATE * R[:, 0]
    v_s = _QuadraticSpline(time, v)

    Cnb = dcm.from_hpr(h, p, r)
    Cib = util.mm_prod(Cin, Cnb)

    Cib_spline = dcm.Spline(time, Cib)
    a = Cib_spline.c[2]
    b = Cib_spline.c[1]
    c = Cib_spline.c[0]

    g = earth.gravitation_ecef(lat, lon_inertial, alt)
    a_s = v_s.derivative()
    d = a_s.c[1] - g[:-1]
    e = a_s.c[0] - np.diff(g, axis=0) / dt

    d = util.mv_prod(Cib[:-1], d, at=True)
    e = util.mv_prod(Cib[:-1], e, at=True)

    gyros, accels = _compute_readings(dt, a, b, c, d, e)

    traj = pd.DataFrame(index=np.arange(n_points))
    traj['lat'] = lat
    traj['lon'] = lon
    traj['alt'] = alt
    traj['VE'] = VE
    traj['VN'] = VN
    traj['VU'] = VU
    traj['h'] = h
    traj['p'] = p
    traj['r'] = r

    return traj, gyros, accels

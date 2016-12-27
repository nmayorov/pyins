"""Strapdown sensor simulator."""
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d
from . import dcm, earth, coord
from .util import mm_prod, mv_prod


def _interpolate_ll(lat, lon, t, t_out):
    Cen = dcm.from_llw(lat, lon)
    dCen = mm_prod(Cen[:-1], Cen[1:], at=True)
    rv = dcm.to_rv(dCen)
    rv_acc = np.vstack((np.zeros(3), np.cumsum(rv, axis=0)))
    rv_acc_interp = interp1d(t, rv_acc, axis=0)
    rv_acc_out = rv_acc_interp(t_out)

    Cen_out = []
    for i in range(t.shape[0] - 1):
        ind = (t_out >= t[i]) & (t_out < t[i + 1])
        rv_i = rv_acc_out[ind] - rv_acc[i]
        dC = dcm.from_rv(rv_i)
        Cen_out.append(mm_prod(Cen[i], dC))

    Cen_out = np.vstack(Cen_out)
    lat, lon, _ = dcm.to_llw(Cen_out)
    return lat, lon


def _interpolate_hpr(h, p, r, t, t_out):
    C = dcm.from_hpr(h, p, r)
    dCen = mm_prod(C[:-1], C[1:], at=True)
    rv = dcm.to_rv(dCen)
    rv_acc = np.vstack((np.zeros(3), np.cumsum(rv, axis=0)))
    rv_acc_interp = interp1d(t, rv_acc, axis=0)
    rv_acc_out = rv_acc_interp(t_out)

    C_out = []
    for i in range(t.shape[0] - 1):
        ind = (t_out >= t[i]) & (t_out < t[i + 1])
        rv_i = rv_acc_out[ind] - rv_acc[i]
        dC = dcm.from_rv(rv_i)
        C_out.append(mm_prod(C[i], dC))

    C_out = np.vstack(C_out)
    return dcm.to_hpr(C_out)


def from_position(dt, lat, lon, t=None, h=None, p=None, r=None, alt=None):
    """Generate inertial readings given position and optionally attitude.

    Parameters
    ----------
    dt : float
        Desired sensor sampling period.
    lat, lon : array_like, shape (n_points,)
        Time series of latitude and longitude.
    t : array_like with shape (n_points,) or None, optional
        If array, time stamps corresponding to `lat` and `lon`. If None
        (default), then `lat` and `lon` are assumed to be spaced by `dt`
    h, p, r : array_like with shape (n_points,) or None
        Time series of heading, pitch and roll angles. If (None) default,
        heading and pitch will be computed assuming that the velocity is
        directed along the body longitudinal axis, and roll will be set to 0.
        This is not going to work if the system is stationary during some
        period (to be redesigned). Note that `h`, `p`, `r` must be all set or
        left as None.
    alt : array_like with shape (n_points,) or None, optional
        Time series of altitude. If None (default), then altitude is assumed
        to be 0 all the time.

    Returns
    -------
    traj : DataFrame
        Trajectory. Contains n_readings + 1 rows.
    gyro : ndarray, shape (n_readings, 3)
        Gyro readings.
    accel : ndarray, shape (n_readings, 3)
        Accelerometer readings.
    """
    test = (h is None) + (p is None) + (r is None)
    if test != 0 and test != 3:
        raise ValueError("`h`, `p`, `r` must be all set or left as None.")

    need_interpolation = t is not None
    compute_attitude = test == 3

    if t is None:
        t = np.arange(lat.shape[0]) * dt
        t_out = t
    else:
        t = np.asarray(t)
        t_out = np.arange(t[0], t[-1], dt)

    if np.any(np.diff(t) <= 0):
        raise ValueError("`t` must be strictly increasing.")

    if alt is None:
        alt = np.zeros_like(t_out)

    if need_interpolation:
        lat, lon = _interpolate_ll(lat, lon, t, t_out)

        if not compute_attitude:
            h, p, r = _interpolate_hpr(h, p, r, t, t_out)

    return _generate_sensors(dt, lat, lon, alt, h, p, r)


def _generate_sensors(dt, lat, lon, alt, h, p, r):
    time = dt * np.arange(lat.shape[0])
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
    V = mv_prod(Cin, V, at=True)

    if h is None:
        h = np.rad2deg(np.arctan2(V[:, 0], V[:, 1]))

    if p is None:
        p = np.rad2deg(np.arctan2(V[:, 2], np.hypot(V[:, 0], V[:, 1])))

    if r is None:
        r = np.zeros_like(p)

    Cnb = dcm.from_hpr(h, p, r)
    Cib = mm_prod(Cin, Cnb)

    Cib_spline = dcm.Spline(time, Cib)
    a = Cib_spline.c[2]
    b = Cib_spline.c[1]
    c = Cib_spline.c[0]
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

    g = earth.gravitation_ecef(lat_inertial, lon_inertial)
    a_s = v_s.derivative()
    d = a_s.c[1] - g[:-1]
    e = a_s.c[0] - np.diff(g, axis=0) / dt

    d = mv_prod(Cib[:-1], d, at=True)
    e = mv_prod(Cib[:-1], e, at=True)
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

    traj = pd.DataFrame(index=np.arange(time.shape[0]))
    traj['lat'] = lat
    traj['lon'] = lon
    traj['VE'] = V[:, 0]
    traj['VN'] = V[:, 1]
    traj['h'] = h
    traj['p'] = p
    traj['r'] = r

    return traj, gyros, accels

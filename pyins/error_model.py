"""Navigation error models to use in EKF-like estimation filters."""
import numpy as np
import pandas as pd
from . import dcm, earth, util


N_BASE_STATES = 9
DR1 = 0
DR2 = 1
DR3 = 2
DV1 = 3
DV2 = 4
DV3 = 5
PHI1 = 6
PHI2 = 7
PSI3 = 8

DRE = 0
DRN = 1
DRU = 2
DVE = 3
DVN = 4
DVU = 5
DH = 6
DP = 7
DR = 8


def transform_to_output_errors(traj):
    lat = np.deg2rad(traj.lat)
    VE = traj.VE
    VN = traj.VN
    VU = traj.VU
    h = np.deg2rad(traj.h)
    p = np.deg2rad(traj.p)

    tlat = np.tan(lat)
    sh, ch = np.sin(h), np.cos(h)
    cp, tp = np.cos(p), np.tan(p)

    T = np.zeros((traj.shape[0], N_BASE_STATES, N_BASE_STATES))
    T[:, DRE, DR1] = 1
    T[:, DRN, DR2] = 1
    T[:, DRU, DR3] = 1
    T[:, DVE, DR1] = VN * tlat / earth.R0
    T[:, DVE, DV1] = 1
    T[:, DVE, PHI2] = -VU
    T[:, DVE, PSI3] = VN
    T[:, DVN, DR1] = -VE * tlat / earth.R0
    T[:, DVN, DV2] = 1
    T[:, DVN, PHI1] = VU
    T[:, DVN, PSI3] = -VE
    T[:, DVU, DV3] = 1
    T[:, DVU, PHI1] = -VN
    T[:, DVU, PHI2] = VE
    T[:, DH, DR1] = tlat / earth.R0
    T[:, DH, PHI1] = -sh * tp
    T[:, DH, PHI2] = -ch * tp
    T[:, DH, PSI3] = 1
    T[:, DP, PHI1] = -ch
    T[:, DP, PHI2] = sh
    T[:, DR, PHI1] = -sh / cp
    T[:, DR, PHI2] = -ch / cp

    return T


def compute_output_errors(traj, x, P, output_stamps,
                          gyro_model, accel_model):
    T = transform_to_output_errors(traj.loc[output_stamps])
    y = util.mv_prod(T, x[:, :N_BASE_STATES])
    Py = util.mm_prod(T, P[:, :N_BASE_STATES, :N_BASE_STATES])
    Py = util.mm_prod(Py, T, bt=True)
    sd_y = np.diagonal(Py, axis1=1, axis2=2) ** 0.5

    err = pd.DataFrame(index=output_stamps)
    err['lat'] = y[:, DRN]
    err['lon'] = y[:, DRE]
    err['alt'] = y[:, DRU]
    err['VE'] = y[:, DVE]
    err['VN'] = y[:, DVN]
    err['VU'] = y[:, DVU]
    err['h'] = np.rad2deg(y[:, DH])
    err['p'] = np.rad2deg(y[:, DP])
    err['r'] = np.rad2deg(y[:, DR])

    sd = pd.DataFrame(index=output_stamps)
    sd['lat'] = sd_y[:, DRN]
    sd['lon'] = sd_y[:, DRE]
    sd['alt'] = sd_y[:, DRU]
    sd['VE'] = sd_y[:, DVE]
    sd['VN'] = sd_y[:, DVN]
    sd['VU'] = sd_y[:, DVU]
    sd['h'] = np.rad2deg(sd_y[:, DH])
    sd['p'] = np.rad2deg(sd_y[:, DP])
    sd['r'] = np.rad2deg(sd_y[:, DR])

    gyro_err = pd.DataFrame(index=output_stamps)
    gyro_sd = pd.DataFrame(index=output_stamps)
    n = N_BASE_STATES
    for i, name in enumerate(gyro_model.states):
        gyro_err[name] = x[:, n + i]
        gyro_sd[name] = P[:, n + i, n + i] ** 0.5

    accel_err = pd.DataFrame(index=output_stamps)
    accel_sd = pd.DataFrame(index=output_stamps)
    ng = gyro_model.n_states
    for i, name in enumerate(accel_model.states):
        accel_err[name] = x[:, n + ng + i]
        accel_sd[name] = P[:, n + ng + i, n + ng + i] ** 0.5

    return err, sd, gyro_err, gyro_sd, accel_err, accel_sd


def fill_system_matrix(traj):
    n_samples = traj.shape[0]
    lat = np.deg2rad(traj.lat)
    slat, clat = np.sin(lat), np.cos(lat)
    tlat = slat / clat

    u = np.zeros((n_samples, 3))
    u[:, 1] = earth.RATE * clat
    u[:, 2] = earth.RATE * slat

    rho = np.empty((n_samples, 3))
    rho[:, 0] = -traj.VN / earth.R0
    rho[:, 1] = traj.VE / earth.R0
    rho[:, 2] = rho[:, 1] * tlat

    Cnb = dcm.from_hpr(traj.h, traj.p, traj.r)

    F = np.zeros((n_samples, N_BASE_STATES, N_BASE_STATES))

    F[:, DR1, DR2] = rho[:, 2]
    F[:, DR1, DV1] = 1
    F[:, DR1, PHI2] = -traj.VU
    F[:, DR1, PSI3] = traj.VN

    F[:, DR2, DR1] = -rho[:, 2]
    F[:, DR2, DV2] = 1
    F[:, DR2, PHI1] = traj.VU
    F[:, DR2, PSI3] = -traj.VE

    F[:, DR3, DV3] = 1
    F[:, DR3, PHI1] = -traj.VN
    F[:, DR3, PHI2] = traj.VE

    F[:, DV1, DV2] = 2 * u[:, 2] + rho[:, 2]
    F[:, DV1, PHI2] = -earth.G0

    F[:, DV2, DV1] = -2 * u[:, 2] - rho[:, 2]
    F[:, DV2, PHI1] = earth.G0

    F[:, DV3, DR3] = 2 * earth.G0 / earth.R0
    F[:, DV3, DV1] = 2 * u[:, 1] + rho[:, 1]
    F[:, DV3, DV2] = -2 * u[:, 0] - rho[:, 0]

    F[:, PHI1, DR1] = -u[:, 2] / earth.R0
    F[:, PHI1, DV2] = -1 / earth.R0
    F[:, PHI1, PHI1] = -traj.VU / earth.R0
    F[:, PHI1, PHI2] = u[:, 2] + rho[:, 2]
    F[:, PHI1, PSI3] = -u[:, 1]

    F[:, PHI2, DR2] = -u[:, 2] / earth.R0
    F[:, PHI2, DV1] = 1 / earth.R0
    F[:, PHI2, PHI1] = -u[:, 2] - rho[:, 2]
    F[:, PHI2, PHI2] = -traj.VU / earth.R0
    F[:, PHI2, PSI3] = u[:, 0]

    F[:, PSI3, DR1] = (u[:, 0] + rho[:, 0]) / earth.R0
    F[:, PSI3, DR2] = (u[:, 1] + rho[:, 1]) / earth.R0
    F[:, PSI3, PHI1] = u[:, 1] + rho[:, 1]
    F[:, PSI3, PHI2] = -u[:, 0] - rho[:, 0]

    B_gyro = np.zeros((n_samples, N_BASE_STATES, 3))
    B_gyro[np.ix_(np.arange(n_samples), [PHI1, PHI2, PSI3], [0, 1, 2])] = -Cnb

    B_accel = np.zeros((n_samples, N_BASE_STATES, 3))
    B_accel[np.ix_(np.arange(n_samples), [DV1, DV2, DV3], [0, 1, 2])] = Cnb

    return F, B_gyro, B_accel


def propagate_errors(dt, traj, d_lat=0, d_lon=0, d_alt=0,
                     d_VE=0, d_VN=0, d_VU=0,
                     d_h=0, d_p=0, d_r=0, d_gyro=0, d_accel=0):
    """Deterministic linear propagation of INS errors.

    Parameters
    ----------
    dt : float
        Time step per stamp.
    traj : DataFrame
        Trajectory.
    d_lat, d_lon, d_alt : float
        Initial position errors in meters.
    d_VE, d_VN, d_VU : float
        Initial velocity errors.
    d_h, d_p, d_r : float
        Initial heading, pitch and roll errors.
    d_gyro, d_accel : array_like
        Gyro and accelerometer errors (in SI units). Can be constant or
        specified for each time stamp in `traj`.

    Returns
    -------
    traj_err : DataFrame
        Trajectory errors.
    """
    Fi, Fig, Fia = fill_system_matrix(traj)
    Phi = 0.5 * (Fi[1:] + Fi[:-1]) * dt
    Phi[:] += np.identity(Phi.shape[-1])

    d_gyro = np.asarray(d_gyro)
    d_accel = np.asarray(d_accel)
    if d_gyro.ndim == 0:
        d_gyro = np.resize(d_gyro, 3)
    if d_accel.ndim == 0:
        d_accel = np.resize(d_accel, 3)

    d_gyro = util.mv_prod(Fig, d_gyro)
    d_accel = util.mv_prod(Fia, d_accel)
    d_sensor = 0.5 * (d_gyro[1:] + d_gyro[:-1] + d_accel[1:] + d_accel[:-1])

    T = transform_to_output_errors(traj)
    d_h = np.deg2rad(d_h)
    d_p = np.deg2rad(d_p)
    d_r = np.deg2rad(d_r)
    x0 = np.array([d_lon, d_lat, d_alt, d_VE, d_VN, d_VU, d_h, d_p, d_r])
    x0 = np.linalg.inv(T[0]).dot(x0)

    n_samples = Fi.shape[0]
    x = np.empty((n_samples, N_BASE_STATES))
    x[0] = x0
    for i in range(n_samples - 1):
        x[i + 1] = Phi[i].dot(x[i]) + d_sensor[i] * dt

    x = util.mv_prod(T, x)
    error = pd.DataFrame(index=traj.index)
    error['lat'] = x[:, DRN]
    error['lon'] = x[:, DRE]
    error['alt'] = x[:, DRU]
    error['VE'] = x[:, DVE]
    error['VN'] = x[:, DVN]
    error['VU'] = x[:, DVU]
    error['h'] = np.rad2deg(x[:, DH])
    error['p'] = np.rad2deg(x[:, DP])
    error['r'] = np.rad2deg(x[:, DR])

    return error

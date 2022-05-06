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
DROLL = 6
DPITCH = 7
DHEADING = 8


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
    T[:, DHEADING, DR1] = tlat / earth.R0
    T[:, DHEADING, PHI1] = -sh * tp
    T[:, DHEADING, PHI2] = -ch * tp
    T[:, DHEADING, PSI3] = 1
    T[:, DPITCH, PHI1] = -ch
    T[:, DPITCH, PHI2] = sh
    T[:, DROLL, PHI1] = -sh / cp
    T[:, DROLL, PHI2] = -ch / cp

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
    err['h'] = np.rad2deg(y[:, DHEADING])
    err['p'] = np.rad2deg(y[:, DPITCH])
    err['r'] = np.rad2deg(y[:, DROLL])

    sd = pd.DataFrame(index=output_stamps)
    sd['lat'] = sd_y[:, DRN]
    sd['lon'] = sd_y[:, DRE]
    sd['alt'] = sd_y[:, DRU]
    sd['VE'] = sd_y[:, DVE]
    sd['VN'] = sd_y[:, DVN]
    sd['VU'] = sd_y[:, DVU]
    sd['h'] = np.rad2deg(sd_y[:, DHEADING])
    sd['p'] = np.rad2deg(sd_y[:, DPITCH])
    sd['r'] = np.rad2deg(sd_y[:, DROLL])

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

    Cnb = dcm.from_rph(traj[['r', 'p', 'h']])

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


def propagate_errors(dt, traj,
                     delta_position_n=np.zeros(3),
                     delta_velocity_n=np.zeros(3),
                     delta_rph=np.zeros(3),
                     delta_gyro=np.zeros(3),
                     delta_accel=np.zeros(3)):
    """Deterministic linear propagation of INS errors.

    Parameters
    ----------
    dt : float
        Time step per stamp.
    traj : DataFrame
        Trajectory.
    delta_position_n : array_like, shape (3,)
        Initial position errors in meters resolved in ENU.
    delta_velocity_n : array_like, shape (3,)
        Initial velocity errors resolved in ENU.
    delta_rph : array_like, shape (3,)
        Initial heading, pitch and roll errors.
    delta_gyro, delta_accel : float or array_like
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

    delta_gyro = util.mv_prod(Fig, delta_gyro)
    delta_accel = util.mv_prod(Fia, delta_accel)
    delta_sensor = 0.5 * (delta_gyro[1:] + delta_gyro[:-1] +
                          delta_accel[1:] + delta_accel[:-1])

    T = transform_to_output_errors(traj)
    x0 = np.hstack([delta_position_n, delta_velocity_n, np.deg2rad(delta_rph)])
    x0 = np.linalg.inv(T[0]).dot(x0)

    n_samples = Fi.shape[0]
    x = np.empty((n_samples, N_BASE_STATES))
    x[0] = x0
    for i in range(n_samples - 1):
        x[i + 1] = Phi[i].dot(x[i]) + delta_sensor[i] * dt

    x = util.mv_prod(T, x)
    error = pd.DataFrame(index=traj.index)
    error['lat'] = x[:, DRN]
    error['lon'] = x[:, DRE]
    error['alt'] = x[:, DRU]
    error['VE'] = x[:, DVE]
    error['VN'] = x[:, DVN]
    error['VU'] = x[:, DVU]
    error['h'] = np.rad2deg(x[:, DHEADING])
    error['p'] = np.rad2deg(x[:, DPITCH])
    error['r'] = np.rad2deg(x[:, DROLL])

    return error

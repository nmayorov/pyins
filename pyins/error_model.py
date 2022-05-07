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
PHI3 = 8

DR = [DR1, DR2, DR3]
DV = [DV1, DV2, DV3]
PHI = [PHI1, PHI2, PHI3]

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
    heading = np.deg2rad(traj.heading)
    pitch = np.deg2rad(traj.pitch)

    sh, ch = np.sin(heading), np.cos(heading)
    cp, tp = np.cos(pitch), np.tan(pitch)

    T = np.zeros((traj.shape[0], N_BASE_STATES, N_BASE_STATES))
    samples = np.arange(len(traj))
    T[np.ix_(samples, DR, DR)] = np.eye(3)
    T[np.ix_(samples, DV, DV)] = np.eye(3)
    T[np.ix_(samples, DV, PHI)] = dcm.skew_matrix(traj[['VE', 'VN', 'VU']])

    T[:, DHEADING, PHI1] = -sh * tp
    T[:, DHEADING, PHI2] = -ch * tp
    T[:, DHEADING, PHI3] = 1
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
    err['roll'] = np.rad2deg(y[:, DROLL])
    err['pitch'] = np.rad2deg(y[:, DPITCH])
    err['heading'] = np.rad2deg(y[:, DHEADING])

    sd = pd.DataFrame(index=output_stamps)
    sd['lat'] = sd_y[:, DRN]
    sd['lon'] = sd_y[:, DRE]
    sd['alt'] = sd_y[:, DRU]
    sd['VE'] = sd_y[:, DVE]
    sd['VN'] = sd_y[:, DVN]
    sd['VU'] = sd_y[:, DVU]
    sd['roll'] = np.rad2deg(sd_y[:, DROLL])
    sd['pitch'] = np.rad2deg(sd_y[:, DPITCH])
    sd['heading'] = np.rad2deg(sd_y[:, DHEADING])

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

    V_skew = dcm.skew_matrix(traj[['VE', 'VN', 'VU']])
    R = earth.curvature_matrix(traj.lat, traj.alt)
    Omega_n = earth.rate_n(traj.lat)
    rho_n = util.mv_prod(R, traj[['VE', 'VN', 'VU']])
    g_n = earth.gravity_n(traj.lat, traj.alt)
    Cnb = dcm.from_rph(traj[['roll', 'pitch', 'heading']])

    F = np.zeros((n_samples, N_BASE_STATES, N_BASE_STATES))
    rows = np.arange(n_samples)

    F[np.ix_(rows, DR, DV)] = np.eye(3)
    F[np.ix_(rows, DR, PHI)] = V_skew

    F[np.ix_(rows, DV, DV)] = -dcm.skew_matrix(2 * Omega_n + rho_n)
    F[np.ix_(rows, DV, PHI)] = -dcm.skew_matrix(g_n)
    F[:, DV3, DR3] = 2 * earth.gravity(traj.lat, traj.alt) / earth.R0

    F[np.ix_(rows, PHI, DR)] = util.mm_prod(dcm.skew_matrix(Omega_n), R)
    F[np.ix_(rows, PHI, DV)] = R
    F[np.ix_(rows, PHI, PHI)] = \
        -dcm.skew_matrix(rho_n + Omega_n) + util.mm_prod(R, V_skew)

    B_gyro = np.zeros((n_samples, N_BASE_STATES, 3))
    B_gyro[np.ix_(rows, DV, [0, 1, 2])] = util.mm_prod(V_skew, Cnb)
    B_gyro[np.ix_(rows, PHI, [0, 1, 2])] = -Cnb

    B_accel = np.zeros((n_samples, N_BASE_STATES, 3))
    B_accel[np.ix_(rows, DV, [0, 1, 2])] = Cnb

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
    error['roll'] = np.rad2deg(x[:, DROLL])
    error['pitch'] = np.rad2deg(x[:, DPITCH])
    error['heading'] = np.rad2deg(x[:, DHEADING])

    return error

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from . import kalman, util, earth, transform
from .util import (VEL_COLS, RPH_COLS, THETA_COLS, DV_COLS, TRAJECTORY_COLS,
                   TRAJECTORY_ERROR_COLS)


def interpolate_pva(first, second, alpha):
    first = first.values
    second = second.values
    rot_1 = Rotation.from_euler('xyz', first[-3:], True)
    rot_2 = Rotation.from_euler('xyz', second[-3:], True)
    rph_average = (
        Rotation.concatenate([rot_1, rot_2])
        .mean([1 - alpha, alpha])
        .as_euler("xyz", True)
    )
    return pd.Series(
        np.hstack([(1 - alpha) * first[:-3] + alpha * second[:-3], rph_average]),
        index=TRAJECTORY_COLS,
    )


def initialize_covariance(pva, pos_sd, vel_sd, level_sd, azimuth_sd,
                          error_model, gyro_model, accel_model):
    P_pva = np.zeros((9, 9))
    P_pva[error_model.DRN, error_model.DRN] = pos_sd ** 2
    P_pva[error_model.DRE, error_model.DRE] = pos_sd ** 2
    P_pva[error_model.DRD, error_model.DRD] = pos_sd ** 2
    P_pva[error_model.DVN, error_model.DVN] = vel_sd ** 2
    P_pva[error_model.DVE, error_model.DVE] = vel_sd ** 2
    P_pva[error_model.DVD, error_model.DVD] = vel_sd ** 2
    P_pva[error_model.DROLL, error_model.DROLL] = level_sd ** 2
    P_pva[error_model.DPITCH, error_model.DPITCH] = level_sd ** 2
    P_pva[error_model.DHEADING, error_model.DHEADING] = azimuth_sd ** 2

    n_states = error_model.n_states + gyro_model.n_states + accel_model.n_states

    ins_block = slice(error_model.n_states)
    gyro_block = slice(error_model.n_states, error_model.n_states + gyro_model.n_states)
    accel_block = slice(error_model.n_states + gyro_model.n_states, n_states)

    P = np.zeros((n_states, n_states))
    T = error_model.transform_to_internal(pva)

    P[ins_block, ins_block] = T @ P_pva @ T.transpose()
    P[gyro_block, gyro_block] = gyro_model.P
    P[accel_block, accel_block] = accel_model.P

    return P


def compute_error_propagation_matrices(pva, gyro, accel, time_delta,
                                       error_model, gyro_model, accel_model):
    n_states = error_model.n_states + gyro_model.n_states + accel_model.n_states
    n_noises = (gyro_model.n_output_noises + gyro_model.n_noises +
                accel_model.n_output_noises + accel_model.n_noises)

    Fii, Fig, Fia = error_model.system_matrices(pva)

    Hg = gyro_model.output_matrix(gyro)
    Ha = accel_model.output_matrix(accel)

    ins_block = slice(error_model.n_states)
    gyro_block = slice(error_model.n_states, error_model.n_states + gyro_model.n_states)
    accel_block = slice(error_model.n_states + gyro_model.n_states, n_states)

    gyro_out_noise_block = slice(gyro_model.n_output_noises)
    accel_out_noise_block = slice(
        gyro_model.n_output_noises,
        gyro_model.n_output_noises + accel_model.n_output_noises)
    gyro_noise_block = slice(
        gyro_model.n_output_noises + accel_model.n_output_noises,
        gyro_model.n_output_noises + accel_model.n_output_noises + gyro_model.n_noises)
    accel_noise_block = slice(
        gyro_model.n_output_noises + accel_model.n_output_noises + gyro_model.n_noises,
        n_noises)

    F = np.zeros((n_states, n_states))
    F[ins_block, ins_block] = Fii
    F[ins_block, gyro_block] = Fig @ Hg
    F[ins_block, accel_block] = Fia @ Ha
    F[gyro_block, gyro_block] = gyro_model.F
    F[accel_block, accel_block] = accel_model.F

    G = np.zeros((n_states, n_noises))
    G[ins_block, gyro_out_noise_block] = Fig @ gyro_model.J
    G[ins_block, accel_out_noise_block] = Fia @ accel_model.J
    G[gyro_block, gyro_noise_block] = gyro_model.G
    G[accel_block, accel_noise_block] = accel_model.G

    q = np.hstack((gyro_model.v, accel_model.v, gyro_model.q, accel_model.q))

    return kalman.compute_process_matrices(F, G @ np.diag(q ** 2) @ G.transpose(),
                                           time_delta)


def compute_sd(P, trajectory, error_model, gyro_model, accel_model):
    ins_block = slice(error_model.n_states)
    gyro_block = slice(error_model.n_states, error_model.n_states + gyro_model.n_states)
    accel_block = slice(
        error_model.n_states + gyro_model.n_states,
        error_model.n_states + gyro_model.n_states + accel_model.n_states)

    P_ins = P[:, ins_block, ins_block]
    P_gyro = P[:, gyro_block, gyro_block]
    P_accel = P[:, accel_block, accel_block]

    T = error_model.transform_to_output(trajectory)
    trajectory_sd = np.diagonal(
        util.mm_prod_symmetric(T, P_ins), axis1=1, axis2=2) ** 0.5
    gyro_sd = np.diagonal(P_gyro, axis1=1, axis2=2) ** 0.5
    accel_sd = np.diagonal(P_accel, axis1=1, axis2=2) ** 0.5

    trajectory_sd = pd.DataFrame(trajectory_sd, index=trajectory.index,
                                 columns=TRAJECTORY_ERROR_COLS)
    gyro_sd = pd.DataFrame(gyro_sd, index=trajectory.index, columns=gyro_model.states)
    accel_sd = pd.DataFrame(accel_sd, index=trajectory.index,
                            columns=accel_model.states)
    return trajectory_sd, gyro_sd, accel_sd


def compute_feedforward_result(x, P, trajectory_nominal, trajectory,
                               error_model, gyro_model, accel_model):
    ins_block = slice(error_model.n_states)
    gyro_block = slice(error_model.n_states, error_model.n_states + gyro_model.n_states)
    accel_block = slice(
        error_model.n_states + gyro_model.n_states,
        error_model.n_states + gyro_model.n_states + accel_model.n_states)

    P_ins = P[:, ins_block, ins_block]
    P_gyro = P[:, gyro_block, gyro_block]
    P_accel = P[:, accel_block, accel_block]

    x_ins = x[:, ins_block]
    x_gyro = x[:, gyro_block]
    x_accel = x[:, accel_block]

    T = error_model.transform_to_output(trajectory_nominal)
    error_nav = pd.DataFrame(util.mv_prod(T, x_ins), index=trajectory.index,
                             columns=TRAJECTORY_ERROR_COLS)

    rn, _, rp = earth.principal_radii(trajectory_nominal.lat, trajectory_nominal.alt)
    trajectory = trajectory.copy()
    trajectory.lat -= error_nav.north / rn * transform.RAD_TO_DEG
    trajectory.lon -= error_nav.east / rp * transform.RAD_TO_DEG
    trajectory.alt += error_nav.down
    trajectory[VEL_COLS] -= error_nav[VEL_COLS]
    trajectory[RPH_COLS] -= error_nav[RPH_COLS]

    trajectory_sd = pd.DataFrame(
        np.diagonal(util.mm_prod_symmetric(T, P_ins), axis1=1, axis2=2) ** 0.5,
        index=trajectory.index, columns=TRAJECTORY_ERROR_COLS)

    gyro = pd.DataFrame(x_gyro, index=trajectory.index, columns=gyro_model.states)
    gyro_sd = pd.DataFrame(np.diagonal(P_gyro, axis1=1, axis2=2) ** 0.5,
                           index=trajectory.index, columns=gyro_model.states)

    accel = pd.DataFrame(x_accel, index=trajectory.index, columns=accel_model.states)
    accel_sd = pd.DataFrame(np.diagonal(P_accel, axis1=1, axis2=2) ** 0.5,
                            index=trajectory.index, columns=accel_model.states)

    return trajectory, trajectory_sd, gyro, gyro_sd, accel, accel_sd


def correct_increments(increments, gyro_model, accel_model):
    result = increments.copy()
    result[THETA_COLS] = gyro_model.correct_increments(increments['dt'],
                                                       increments[THETA_COLS])
    result[DV_COLS] = accel_model.correct_increments(increments['dt'],
                                                     increments[DV_COLS])
    return result

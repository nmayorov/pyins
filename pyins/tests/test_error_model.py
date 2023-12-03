import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation
from pyins import error_model, sim, transform
from pyins.strapdown import compute_increments_from_imu, Integrator
from pyins.transform import compute_state_difference
from pyins.util import (VEL_COLS, RPH_COLS, NED_COLS, GYRO_COLS, ACCEL_COLS,
                        TRAJECTORY_ERROR_COLS)


def test_phi_to_delta_rph():
    rph = [10, -20, 30]
    mat = transform.mat_from_rph(rph)
    phi = np.array([-0.02, 0.01, -0.03])
    mat_perturbed = Rotation.from_rotvec(-phi).as_matrix() @ mat

    rph_perturbed = transform.mat_to_rph(mat_perturbed)
    delta_rph_true = rph_perturbed - rph

    T = error_model._phi_to_delta_rph(rph)
    delta_rph_linear = T @ phi

    assert_allclose(delta_rph_linear, delta_rph_true, rtol=1e-1)


def test_ErrorModel():
    np.random.seed(0)
    em = error_model.InsErrorModel()
    pva = pd.Series({
        'lat': 58.0,
        'lon': 55.0,
        'alt': 150.0,
        'VN': 10.0,
        'VE': -12.2,
        'VD': 0.5,
        'roll': 0.5,
        'pitch': 1.5,
        'heading': 45.0
    })
    pva_error = pd.Series({
        'north': -10.0,
        'east': 15.0,
        'down': -5.0,
        'VN': 1.0,
        'VE': 0.5,
        'VD': 0.1,
        'roll': 0.1,
        'pitch': -0.15,
        'heading': 0.5,
    })
    assert em.n_states == 9
    x = np.random.randn(9)
    T_io = em.transform_to_internal(pva)
    T_oi = em.transform_to_output(pva)
    assert_allclose(x, T_oi @ T_io @ x)
    assert_allclose(x, T_io @ T_oi @ x)

    x = T_io @ pva_error.values
    pva_perturbed = sim.perturb_pva(pva, pva_error)
    pva_corrected = em.correct_pva(pva_perturbed, x)
    assert_allclose(transform.compute_state_difference(pva_corrected, pva), 0,
                    atol=1e-3)

    translation_b = [1, -2, 0.5]
    omega_b = pd.Series([0.2, -0.3, 0.5], index=['rate_x', 'rate_y', 'rate_z'])
    pva = pd.concat([pva, omega_b])
    pva_perturbed = pd.concat([pva_perturbed, omega_b])

    pva_t = transform.translate_trajectory(pva, translation_b)
    pva_perturbed_t = transform.translate_trajectory(pva_perturbed, translation_b)

    pva_diff = transform.compute_state_difference(pva_perturbed_t, pva_t)
    H_position = em.position_error_jacobian(pva, translation_b)
    assert_allclose(pva_diff[NED_COLS], H_position @ x, rtol=1e-5)

    H_ned_velocity = em.ned_velocity_error_jacobian(pva, translation_b)
    assert_allclose(pva_diff[VEL_COLS], H_ned_velocity @ x, rtol=1e-4)

    velocity_b = transform.mat_from_rph(pva[RPH_COLS]).T @ pva[VEL_COLS]
    velocity_perturbed_b = (transform.mat_from_rph(pva_perturbed[RPH_COLS]).T
                            @ pva_perturbed[VEL_COLS])
    H_body_velocity = em.body_velocity_error_jacobian(pva)
    assert_allclose(velocity_perturbed_b - velocity_b, H_body_velocity @ x, rtol=1e-1)


def test_ErrorModel_no_altitude():
    np.random.seed(0)
    em = error_model.InsErrorModel(with_altitude=False)
    pva = pd.Series({
        'lat': 58.0,
        'lon': 55.0,
        'alt': 150.0,
        'VN': 10.0,
        'VE': -12.2,
        'VD': 0.5,
        'roll': 0.5,
        'pitch': 1.5,
        'heading': 45.0
    })
    pva_error = pd.Series({
        'north': -10.0,
        'east': 15.0,
        'down': 0.0,
        'VN': 1.0,
        'VE': 0.5,
        'VD': 0.0,
        'roll': 0.1,
        'pitch': -0.15,
        'heading': 0.5,
    })
    assert em.n_states == 7
    T_io = em.transform_to_internal(pva)
    T_oi = em.transform_to_output(pva)

    x_out = np.random.randn(9)
    x_out[error_model.InsErrorModel.DVD] = 0
    x_out[error_model.InsErrorModel.DRD] = 0
    assert_allclose(x_out, T_oi @ T_io @ x_out)

    x_in = np.random.randn(7)
    assert_allclose(x_in, T_io @ T_oi @ x_in)

    x = T_io @ pva_error.values
    pva_perturbed = sim.perturb_pva(pva, pva_error)
    pva_corrected = em.correct_pva(pva_perturbed, x)
    assert_allclose(transform.compute_state_difference(pva_corrected, pva), 0,
                    atol=1e-3)

    translation_b = [1, -2, 0.5]
    omega_b = pd.Series([0.2, -0.3, 0.5], index=['rate_x', 'rate_y', 'rate_z'])
    pva = pd.concat([pva, omega_b])
    pva_perturbed = pd.concat([pva_perturbed, omega_b])

    pva_t = transform.translate_trajectory(pva, translation_b)
    pva_perturbed_t = transform.translate_trajectory(pva_perturbed, translation_b)

    pva_diff = transform.compute_state_difference(pva_perturbed_t, pva_t)
    H_position = em.position_error_jacobian(pva, translation_b)
    assert_allclose(pva_diff[['north', 'east']], H_position @ x, rtol=1e-5)

    H_ned_velocity = em.ned_velocity_error_jacobian(pva, translation_b)
    assert_allclose(pva_diff[['VN', 'VE']], H_ned_velocity @ x, rtol=1e-4)

    velocity_b = transform.mat_from_rph(pva[RPH_COLS]).T @ pva[VEL_COLS]
    velocity_perturbed_b = (transform.mat_from_rph(pva_perturbed[RPH_COLS]).T
                            @ pva_perturbed[VEL_COLS])
    H_body_velocity = em.body_velocity_error_jacobian(pva)
    assert_allclose(velocity_perturbed_b - velocity_b, H_body_velocity @ x, rtol=1e-1)


@pytest.mark.parametrize("with_altitude", [True, False])
def test_propagate_errors(with_altitude):
    dt = 0.5
    t = 0.5 * 3600
    trajectory, imu = sim.generate_sine_velocity_motion(
        dt, t, [50, 60, 0], [-5, 10, 0.5 if with_altitude else 0],
        [1, 1, 0.5 if with_altitude else 0])
    b = 1e-2
    gyro_bias = np.array([1, -2, 0.5]) * transform.DH_TO_RS
    accel_bias = np.array([b, -b, 2 * b])

    imu[GYRO_COLS] += gyro_bias
    imu[ACCEL_COLS] += accel_bias

    increments = compute_increments_from_imu(imu, 'rate')

    pva_error = pd.Series(index=TRAJECTORY_ERROR_COLS)
    pva_error[NED_COLS] = [200, 100, 20 if with_altitude else 0]
    pva_error[VEL_COLS] = [0.1, -0.2, -0.05 if with_altitude else 0]
    pva_error[RPH_COLS] = [np.rad2deg(-2 * b / 9.8), np.rad2deg(b / 9.8), 0.5]

    initial = sim.perturb_pva(trajectory.iloc[0], pva_error)
    integrator = Integrator(initial)
    traj_c = integrator.integrate(increments)
    error_true = compute_state_difference(traj_c, trajectory)

    error_linear, _ = error_model.propagate_errors(trajectory, pva_error,
                                                   gyro_bias, accel_bias)

    error_scale = error_true.abs().mean()
    rel_diff = (error_linear - error_true) / error_scale
    assert_allclose(rel_diff, 0, atol=0.12)

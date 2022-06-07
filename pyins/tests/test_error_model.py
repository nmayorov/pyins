import numpy as np
import pytest
from numpy.testing import assert_allclose
from pyins import earth, error_models, sim, transform
from pyins.strapdown import compute_theta_and_dv, StrapdownIntegrator
from pyins.transform import perturb_lla, difference_trajectories


@pytest.mark.parametrize("error_model", [error_models.ModifiedPhiModel,
                                         error_models.ModifiedPsiModel])
def test_propagate_errors(error_model):
    dt = 0.5
    t = 0.5 * 3600

    trajectory, gyro, accel = sim.sinusoid_velocity_motion(
        dt, t, [50, 60, 0], [-5, 10, 0.5], [1, 1, 0.5])

    b = 1e-2
    gyro_bias = np.array([1, -2, 0.5]) * transform.DH_TO_RS
    accel_bias = np.array([b, -b, 2 * b])

    gyro += gyro_bias * dt
    accel += accel_bias * dt
    theta, dv = compute_theta_and_dv(gyro, accel)

    delta_position_n = [-200, 100, 20]
    delta_velocity_n = [0.1, -0.2, -0.05]
    delta_rph = [np.rad2deg(-2 * b / earth.G0), np.rad2deg(b / earth.G0), 0.5]

    lla0 = perturb_lla(trajectory.loc[0, ['lat', 'lon', 'alt']],
                       delta_position_n)
    V0_n = trajectory.loc[0, ['VN', 'VE', 'VD']] + delta_velocity_n
    rph0 = trajectory.loc[0, ['roll', 'pitch', 'heading']] + delta_rph

    integrator = StrapdownIntegrator(dt, lla0, V0_n, rph0)
    traj_c = integrator.integrate(theta, dv)
    error_true = difference_trajectories(traj_c, trajectory)

    error_linear, _ = error_models.propagate_errors(dt, trajectory,
                                                    delta_position_n,
                                                    delta_velocity_n, delta_rph,
                                                    gyro_bias, accel_bias,
                                                    error_model=error_model())

    error_scale = np.mean(np.abs(error_true))
    rel_diff = (error_linear - error_true) / error_scale
    assert_allclose(rel_diff, 0, atol=0.1)


@pytest.mark.parametrize("error_model", [error_models.ModifiedPhiModel,
                                         error_models.ModifiedPsiModel])
def test_propagate_errors_2d(error_model):
    dt = 0.5
    t = 0.5 * 3600

    trajectory, gyro, accel = sim.sinusoid_velocity_motion(
        dt, t, [50, 60, 0], [-5, 10, 0.0], [1, 1, 0.0])

    b = 1e-2
    gyro_bias = np.array([1, -2, 0.5]) * transform.DH_TO_RS
    accel_bias = np.array([b, -b, 2 * b])

    gyro += gyro_bias * dt
    accel += accel_bias * dt
    theta, dv = compute_theta_and_dv(gyro, accel)

    delta_position_n = [-200, 100, 0]
    delta_velocity_n = [0.1, -0.2, 0]
    delta_rph = [np.rad2deg(-2 * b / earth.G0), np.rad2deg(b / earth.G0), 0.5]

    lla0 = perturb_lla(trajectory.loc[0, ['lat', 'lon', 'alt']],
                       delta_position_n)
    V0_n = trajectory.loc[0, ['VN', 'VE', 'VD']] + delta_velocity_n
    rph0 = trajectory.loc[0, ['roll', 'pitch', 'heading']] + delta_rph

    integrator = StrapdownIntegrator(dt, lla0, V0_n, rph0, with_altitude=False)
    traj_c = integrator.integrate(theta, dv)
    error_true = difference_trajectories(traj_c, trajectory)
    error_true = error_true[['north', 'east', 'VN', 'VE',
                             'roll', 'pitch', 'heading']]

    error_linear, _ = error_models.propagate_errors(dt, trajectory,
        delta_position_n, delta_velocity_n, delta_rph, gyro_bias, accel_bias,
        error_model=error_model(False))

    error_scale = np.mean(np.abs(error_true))
    rel_diff = (error_linear - error_true) / error_scale
    assert_allclose(rel_diff, 0, atol=0.1)

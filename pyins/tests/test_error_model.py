import numpy as np
from numpy.testing import assert_allclose
from pyins import sim
from pyins.error_models import propagate_errors
from pyins.integrate import coning_sculling, Integrator
from pyins.transform import perturb_lla, difference_trajectories


def test_propagate_errors():
    # This test is complex and hardly a unit test, but it is strong.
    # I believe it's better than a formal test.
    dt = 0.5
    t = 0.5 * 3600
    n_samples = int(t / dt)
    lla = np.empty((n_samples, 3))
    lla[:, 0] = 50.0
    lla[:, 1] = 60.0
    lla[:, 2] = 0
    rph = np.empty((n_samples, 3))
    rph[:, 0] = -5.0
    rph[:, 1] = 3.0
    rph[:, 2] = 10.0

    traj, gyro, accel = sim.from_position(dt, lla, rph)

    gyro_bias = np.array([1e-8, -2e-8, 3e-8])
    accel_bias = np.array([3e-3, -4e-3, 2e-3])

    gyro += gyro_bias * dt
    accel += accel_bias * dt
    theta, dv = coning_sculling(gyro, accel)

    delta_position_n = [-200, 100, 20]
    delta_velocity_n = [1, -2, -0.5]
    delta_rph = [0.03, -0.02, 0.01]

    lla0 = perturb_lla(traj.loc[0, ['lat', 'lon', 'alt']], delta_position_n)
    V0_n = traj.loc[0, ['VE', 'VN', 'VU']] + delta_velocity_n
    rph0 = traj.loc[0, ['roll', 'pitch', 'heading']] + delta_rph

    integrator = Integrator(dt, lla0, V0_n, rph0)
    traj_c = integrator.integrate(theta, dv)
    error_true = difference_trajectories(traj_c, traj)

    error_linear = propagate_errors(dt, traj, delta_position_n,
                                    delta_velocity_n, delta_rph,
                                    gyro_bias, accel_bias)

    error_scale = np.mean(np.abs(error_true))
    rel_diff = (error_linear - error_true) / error_scale
    assert_allclose(rel_diff, 0, atol=0.1)

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pyins import earth, error_models, sim, transform
from pyins.strapdown import compute_theta_and_dv, Integrator
from pyins.transform import compute_state_difference
from pyins.util import (VEL_COLS, RPH_COLS, NED_COLS, GYRO_COLS, ACCEL_COLS,
                        TRAJECTORY_ERROR_COLS)


@pytest.mark.parametrize("error_model", [error_models.ModifiedPhiModel,
                                         error_models.ModifiedPsiModel])
def test_propagate_errors(error_model):
    dt = 0.5
    t = 0.5 * 3600

    trajectory, imu = sim.sinusoid_velocity_motion(dt, t, [50, 60, 0], [-5, 10, 0.5],
                                                   [1, 1, 0.5])

    b = 1e-2
    gyro_bias = np.array([1, -2, 0.5]) * transform.DH_TO_RS
    accel_bias = np.array([b, -b, 2 * b])

    imu[GYRO_COLS] += gyro_bias * dt
    imu[ACCEL_COLS] += accel_bias * dt

    increments = compute_theta_and_dv(imu, 'increment')

    pva_error = pd.Series(index=TRAJECTORY_ERROR_COLS)
    pva_error[NED_COLS] = [200, 100, 20]
    pva_error[VEL_COLS] = [0.1, -0.2, -0.05]
    pva_error[RPH_COLS] = [np.rad2deg(-2 * b / earth.G0), np.rad2deg(b / earth.G0), 0.5]

    initial = sim.perturb_pva(trajectory.iloc[0], pva_error)
    integrator = Integrator(initial)
    traj_c = integrator.integrate(increments)
    error_true = compute_state_difference(traj_c, trajectory)

    error_linear, _ = error_models.propagate_errors(trajectory, pva_error,
                                                    gyro_bias, accel_bias,
                                                    error_model=error_model())

    error_scale = error_true.abs().mean()
    rel_diff = (error_linear - error_true) / error_scale
    assert_allclose(rel_diff, 0, atol=0.12)

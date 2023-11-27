import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation
from pyins import earth, error_model, sim, transform
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
    delta_rph_linear = np.rad2deg(T @ phi)

    assert_allclose(delta_rph_linear, delta_rph_true, rtol=1e-1)


def test_propagate_errors():
    dt = 0.5
    t = 0.5 * 3600

    trajectory, imu = sim.sinusoid_velocity_motion(dt, t, [50, 60, 0], [-5, 10, 0.5],
                                                   [1, 1, 0.5])

    b = 1e-2
    gyro_bias = np.array([1, -2, 0.5]) * transform.DH_TO_RS
    accel_bias = np.array([b, -b, 2 * b])

    imu[GYRO_COLS] += gyro_bias
    imu[ACCEL_COLS] += accel_bias

    increments = compute_increments_from_imu(imu, 'rate')

    pva_error = pd.Series(index=TRAJECTORY_ERROR_COLS)
    pva_error[NED_COLS] = [200, 100, 20]
    pva_error[VEL_COLS] = [0.1, -0.2, -0.05]
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

from numpy.testing import (assert_, assert_allclose, run_module_suite,
                           assert_equal)
import numpy as np
import pandas as pd
from pyins.filt import (InertialSensor, PositionObs,
                        FeedforwardFilter, _refine_stamps)
from pyins.error_models import propagate_errors
from pyins import filt, sim, strapdown, transform, util
from pyins.transform import perturb_lla, correct_trajectory


def test_InertialSensor():
    s = InertialSensor()
    assert_equal(s.n_states, 0)
    assert_equal(s.n_noises, 0)
    assert_equal(len(s.states), 0)
    assert_equal(s.P.shape, (0, 0))
    assert_equal(s.q.shape, (0,))
    assert_equal(s.F.shape, (0, 0))
    assert_equal(s.G.shape, (0, 0))
    assert_equal(s.output_matrix().shape, (3, 0))

    s = InertialSensor(bias=0.1, bias_walk=0.2)
    assert_equal(s.n_states, 3)
    assert_equal(s.n_noises, 3)
    assert_equal(list(s.states.keys()), ['BIAS_1', 'BIAS_2', 'BIAS_3'])
    assert_equal(list(s.states.values()), [0, 1, 2])
    assert_allclose(s.P, 0.01 * np.identity(3))
    assert_equal(s.q, [0.2, 0.2, 0.2])
    assert_equal(s.F, np.zeros((3, 3)))
    assert_equal(s.G, np.identity(3))
    assert_equal(s.output_matrix(), np.identity(3))

    s = InertialSensor(scale=0.2, scale_walk=0.3)
    assert_equal(s.n_states, 3)
    assert_equal(s.n_noises, 3)
    assert_equal(list(s.states.keys()), ['SCALE_1', 'SCALE_2', 'SCALE_3'])
    assert_equal(list(s.states.values()), [0, 1, 2])
    assert_allclose(s.P, 0.04 * np.identity(3))
    assert_equal(s.q, [0.3, 0.3, 0.3])
    assert_equal(s.F, np.zeros((3, 3)))
    assert_equal(s.G, np.identity(3))
    assert_equal(s.output_matrix([1, 2, 3]), np.diag([1, 2, 3]))
    assert_equal(s.output_matrix([[1, -2, 2], [0.1, 2, 0.5]]),
                 np.array((np.diag([1, -2, 2]), np.diag([0.1, 2, 0.5]))))

    s = InertialSensor(corr_sd=0.1, corr_time=5)
    assert_equal(s.n_states, 3)
    assert_equal(s.n_noises, 3)
    assert_equal(list(s.states.keys()), ['CORR_1', 'CORR_2', 'CORR_3'])
    assert_equal(list(s.states.values()), [0, 1, 2])
    assert_allclose(s.P, 0.01 * np.identity(3))
    q = 0.1 * (2 / 5) ** 0.5
    assert_equal(s.q, [q, q, q])
    assert_allclose(s.F, -np.identity(3) / 5)
    assert_equal(s.G, np.identity(3))

    s = InertialSensor(bias=0.1, bias_walk=0.2, scale=0.3, scale_walk=0.4,
                       corr_sd=0.5, corr_time=10)
    assert_equal(s.n_states, 9)
    assert_equal(s.n_noises, 9)
    assert_equal(list(s.states.keys()),
                 ['BIAS_1', 'BIAS_2', 'BIAS_3', 'SCALE_1', 'SCALE_2',
                  'SCALE_3', 'CORR_1', 'CORR_2', 'CORR_3'])
    assert_equal(list(s.states.values()), np.arange(9))
    assert_allclose(s.P, np.diag([0.01, 0.01, 0.01, 0.09, 0.09, 0.09,
                                  0.25, 0.25, 0.25]))
    q_corr = 0.5 * (2 / 10) ** 0.5
    assert_equal(s.q, [0.2, 0.2, 0.2, 0.4, 0.4, 0.4, q_corr, q_corr, q_corr])
    assert_allclose(s.F, np.diag([0, 0, 0, 0, 0, 0, -1/10, -1/10, -1/10]))
    assert_equal(s.G, np.identity(9))

    H = s.output_matrix([1, 2, 3])
    assert_allclose(H, [[1, 0, 0, 1, 0, 0, 1, 0, 0],
                        [0, 1, 0, 0, 2, 0, 0, 1, 0],
                        [0, 0, 1, 0, 0, 3, 0, 0, 1]])

    H = s.output_matrix([[1, 2, 3], [-1, 2, 0.5]])
    assert_allclose(H[0], [[1, 0, 0, 1, 0, 0, 1, 0, 0],
                           [0, 1, 0, 0, 2, 0, 0, 1, 0],
                           [0, 0, 1, 0, 0, 3, 0, 0, 1]])
    assert_allclose(H[1], [[1, 0, 0, -1, 0, 0, 1, 0, 0],
                           [0, 1, 0, 0, 2, 0, 0, 1, 0],
                           [0, 0, 1, 0, 0, 0.5, 0, 0, 1]])


def test_refine_stamps():
    stamps = [2, 2, 5, 1, 10, 20]
    stamps = _refine_stamps(stamps, 2)
    stamps_true = [1, 2, 4, 5, 7, 9, 10, 12, 14, 16, 18, 20]
    assert_equal(stamps, stamps_true)


def test_FeedbackFilter():
    dt = 0.1
    rng = np.random.RandomState(0)

    trajectory, gyro_true, accel_true = sim.sinusoid_velocity_motion(
        dt, 300, [50, 60, 100], [1, -1, 0.5], [3, 3, 0.5])

    position_obs = filt.PositionObs(
        sim.generate_position_observations(trajectory.iloc[::10], 1, rng), 1)
    ned_velocity_obs = filt.NedVelocityObs(
        sim.generate_ned_velocity_observations(trajectory.iloc[3::10], 0.5,
                                               rng), 0.5)
    body_velocity_obs = filt.BodyVelocityObs(
        sim.generate_body_velocity_observations(trajectory.iloc[6::10], 0.2,
                                                rng), 0.2)

    imu_errors = sim.ImuErrors(
        gyro_bias=np.array([-100, 50, 40]) * transform.DH_TO_RS,
        gyro_noise=1 * transform.DRH_TO_RRS,
        accel_bias=[0.1, -0.1, 0.2],
        accel_noise=1.0 / 60,
        rng=rng)

    gyro_model = filt.InertialSensor(bias=100 * transform.DH_TO_RS,
                                     noise=1 * transform.DRH_TO_RRS)
    accel_model = filt.InertialSensor(bias=0.1, noise=1.0 / 60)

    gyro, accel = imu_errors.apply(dt, gyro_true, accel_true)
    theta, dv = strapdown.compute_theta_and_dv(gyro, accel)

    pos_sd = 10
    vel_sd = 2
    level_sd = 1.0
    azimuth_sd = 5.0

    lla, velocity_n, rph = sim.perturb_navigation_state(
        trajectory.loc[0, ['lat', 'lon', 'alt']],
        trajectory.loc[0, ['VN', 'VE', 'VD']],
        trajectory.loc[0, ['roll', 'pitch', 'heading']],
        pos_sd, vel_sd, level_sd, azimuth_sd,
        rng=rng)

    f = filt.FeedbackFilter(dt, pos_sd=pos_sd, vel_sd=vel_sd,
                            azimuth_sd=azimuth_sd, level_sd=level_sd,
                            gyro_model=gyro_model, accel_model=accel_model)
    integrator = strapdown.StrapdownIntegrator(dt, lla, velocity_n, rph)

    result = f.run(integrator, theta, dv,
                   observations=[position_obs, ned_velocity_obs,
                                 body_velocity_obs],
                   feedback_period=5)

    error = transform.difference_trajectories(result.trajectory, trajectory)

    relative_error = error / result.sd
    assert (util.compute_rms(relative_error) < 1.5).all()

    gyro_bias_relative_error = (np.abs(result.gyro_estimates.iloc[-1] -
                                       imu_errors.gyro_bias)
                                / result.gyro_sd.iloc[-1])
    assert (gyro_bias_relative_error < 2.0).all()

    accel_bias_relative_error = (np.abs(result.accel_estimates.iloc[-1] -
                                        imu_errors.accel_bias)
                                 / result.accel_sd.iloc[-1])
    assert (accel_bias_relative_error < 2.0).all()

    result = f.run_smoother(integrator, theta, dv,
                            observations=[position_obs, ned_velocity_obs,
                                          body_velocity_obs],
                            feedback_period=5)

    error = transform.difference_trajectories(result.trajectory, trajectory)

    relative_error = error / result.sd
    assert (util.compute_rms(relative_error) < 1.6).all()

    gyro_bias_relative_error = np.abs(result.gyro_estimates -
                                      imu_errors.gyro_bias) / result.gyro_sd
    assert (gyro_bias_relative_error < 2.0).all(axis=None)

    accel_bias_relative_error = np.abs(result.accel_estimates -
                                       imu_errors.accel_bias) / result.accel_sd
    assert (accel_bias_relative_error < 2.0).all(axis=None)


def test_FeedforwardFilter():
    # Test that the results are reasonable on a static bench.
    dt = 1
    trajectory = pd.DataFrame(index=np.arange(1 * 3600))
    trajectory['lat'] = 50
    trajectory['lon'] = 60
    trajectory['alt'] = 100
    trajectory['VN'] = 0
    trajectory['VE'] = 0
    trajectory['VD'] = 0
    trajectory['roll'] = 0
    trajectory['pitch'] = 0
    trajectory['heading'] = 0

    np.random.seed(1)
    obs_data = pd.DataFrame(index=trajectory.index[::10])
    obs_data[['lat', 'lon', 'alt']] = perturb_lla(
        trajectory.loc[::10, ['lat', 'lon', 'alt']],
        10 * np.random.randn(len(obs_data), 3))
    position_obs = PositionObs(obs_data, 10)

    delta_position_n = [-3, 5, 0]
    delta_velocity_n = [1, -1, 0]
    delta_rph = [-0.02, 0.03, 0.1]

    errors, _ = propagate_errors(dt, trajectory, delta_position_n,
                                 delta_velocity_n, delta_rph)
    traj_error = correct_trajectory(trajectory, -errors)

    f = FeedforwardFilter(dt, trajectory, 5, 1, 0.2, 0.05)
    res = f.run(traj_error, [position_obs])

    x = errors.loc[3000:]
    y = res.error.loc[3000:]

    assert_allclose(x.east, y.east, rtol=0, atol=10)
    assert_allclose(x.north, y.north, rtol=0, atol=10)
    assert_allclose(x.VE, y.VE, rtol=0, atol=2e-2)
    assert_allclose(x.VN, y.VN, rtol=0, atol=2e-2)
    assert_allclose(x.roll, y.roll, rtol=0, atol=1e-4)
    assert_allclose(x.pitch, y.pitch, rtol=0, atol=1e-4)
    assert_allclose(x.heading, y.heading, rtol=0, atol=1.5e-3)
    assert_(np.all(np.abs(res.residuals[0] < 4)))

    res = f.run_smoother(traj_error, [position_obs])

    # This smoother we don't need to wait until the filter converges,
    # the estimation accuracy is also improved some
    x = errors
    y = res.error

    assert_allclose(x.east, y.east, rtol=0, atol=10)
    assert_allclose(x.north, y.north, rtol=0, atol=10)
    assert_allclose(x.VE, y.VE, rtol=0, atol=2e-2)
    assert_allclose(x.VN, y.VN, rtol=0, atol=2e-2)
    assert_allclose(x.roll, y.roll, rtol=0, atol=1e-4)
    assert_allclose(x.pitch, y.pitch, rtol=0, atol=1e-4)
    assert_allclose(x.heading, y.heading, rtol=0, atol=1.5e-3)
    assert_(np.all(np.abs(res.residuals[0] < 4)))



if __name__ == '__main__':
    run_module_suite()

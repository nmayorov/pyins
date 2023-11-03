from numpy.testing import assert_equal
import numpy as np
from pyins import filt, sim, strapdown, transform, util


def test_refine_stamps():
    stamps = [2, 2, 5, 1, 10, 20]
    stamps = filt._refine_stamps(stamps, 2)
    stamps_true = [1, 2, 4, 5, 7, 9, 10, 12, 14, 16, 18, 20]
    assert_equal(stamps, stamps_true)


def test_FeedbackFilter():
    dt = 0.01
    rng = np.random.RandomState(123456789)

    trajectory, gyro_true, accel_true = sim.sinusoid_velocity_motion(
        dt, 300, [50, 60, 100], [1, -1, 0.5], [3, 3, 0.5])

    position_obs = filt.PositionObs(
        sim.generate_position_observations(trajectory.iloc[::100], 1, rng), 1)
    ned_velocity_obs = filt.NedVelocityObs(
        sim.generate_ned_velocity_observations(trajectory.iloc[30::100], 0.5,
                                               rng), 0.5)
    body_velocity_obs = filt.BodyVelocityObs(
        sim.generate_body_velocity_observations(trajectory.iloc[60::100], 0.2,
                                                rng), 0.2)

    gyro_errors = sim.ImuErrors(
        bias=np.array([-100, 50, 40]) * transform.DH_TO_RS,
        noise=1 * transform.DRH_TO_RRS, rng=rng)
    accel_errors = sim.ImuErrors(bias=[0.1, -0.1, 0.2], noise=1.0 / 60, rng=rng)

    gyro_model = filt.InertialSensor(bias=100 * transform.DH_TO_RS,
                                     noise=1 * transform.DRH_TO_RRS)
    accel_model = filt.InertialSensor(bias=0.1, noise=1.0 / 60)

    gyro = gyro_errors.apply(gyro_true, dt, 'increment')
    accel = accel_errors.apply(accel_true, dt, 'increment')
    theta, dv = strapdown.compute_theta_and_dv(gyro, accel)

    pos_sd = 10
    vel_sd = 2
    level_sd = 1.0
    azimuth_sd = 5.0

    initial = sim.perturb_trajectory_point(trajectory.iloc[0], pos_sd, vel_sd, level_sd,
                                           azimuth_sd, rng=rng)
    f = filt.FeedbackFilter(dt, pos_sd=pos_sd, vel_sd=vel_sd,
                            azimuth_sd=azimuth_sd, level_sd=level_sd,
                            gyro_model=gyro_model, accel_model=accel_model)
    integrator = strapdown.Integrator(dt, initial)

    result = f.run(integrator, theta, dv,
                   observations=[position_obs, ned_velocity_obs,
                                 body_velocity_obs],
                   feedback_period=5)

    error = transform.difference_trajectories(result.trajectory, trajectory)

    relative_error = error / result.sd
    assert (util.compute_rms(relative_error) < 1.5).all()

    gyro_bias_relative_error = (np.abs(result.gyro_estimates.iloc[-1] -
                                       gyro_errors.bias)
                                / result.gyro_sd.iloc[-1])
    assert (gyro_bias_relative_error < 2.0).all()

    accel_bias_relative_error = (np.abs(result.accel_estimates.iloc[-1] -
                                        accel_errors.bias)
                                 / result.accel_sd.iloc[-1])
    assert (accel_bias_relative_error < 2.0).all()


def test_FeedforwardFilter():
    dt = 0.01
    rng = np.random.RandomState(123456789)

    trajectory, gyro_true, accel_true = sim.sinusoid_velocity_motion(
        dt, 300, [50, 60, 100], [1, -1, 0.5], [3, 3, 0.5])

    position_obs = filt.PositionObs(
        sim.generate_position_observations(trajectory.iloc[::100], 1, rng), 1)
    ned_velocity_obs = filt.NedVelocityObs(
        sim.generate_ned_velocity_observations(trajectory.iloc[30::100], 0.5,
                                               rng), 0.5)
    body_velocity_obs = filt.BodyVelocityObs(
        sim.generate_body_velocity_observations(trajectory.iloc[60::100], 0.2,
                                                rng), 0.2)

    gyro_errors = sim.ImuErrors(bias=np.array([-1, 2, 1]) * transform.DH_TO_RS,
                                noise=0.001 * transform.DRH_TO_RRS,
                                rng=rng)
    accel_errors = sim.ImuErrors(bias=[0.01, -0.01, 0.02],
                                 noise=0.01 / 60,
                                 rng=rng)
    gyro_model = filt.InertialSensor(bias=1 * transform.DH_TO_RS,
                                     noise=0.001 * transform.DRH_TO_RRS)
    accel_model = filt.InertialSensor(bias=0.01, noise=0.01 / 60)

    gyro = gyro_errors.apply(gyro_true, dt, 'increment')
    accel = accel_errors.apply(accel_true, dt, 'increment')
    theta, dv = strapdown.compute_theta_and_dv(gyro, accel)

    pos_sd = 10
    vel_sd = 0.1
    level_sd = 0.05
    azimuth_sd = 0.2

    initial = sim.perturb_trajectory_point(trajectory.iloc[0], pos_sd, vel_sd, level_sd,
                                           azimuth_sd, rng=rng)
    integrator = strapdown.Integrator(dt, initial)
    trajectory_computed = integrator.integrate(theta, dv)

    f = filt.FeedforwardFilter(dt, trajectory, pos_sd=pos_sd, vel_sd=vel_sd,
                               azimuth_sd=azimuth_sd, level_sd=level_sd,
                               gyro_model=gyro_model, accel_model=accel_model)

    result = f.run(trajectory_computed,
                   observations=[position_obs, ned_velocity_obs,
                                 body_velocity_obs])

    error = transform.difference_trajectories(result.trajectory, trajectory)

    relative_error = error / result.sd
    assert (util.compute_rms(relative_error) < 1.6).all()

    gyro_bias_relative_error = (np.abs(result.gyro_estimates.iloc[-1] -
                                       gyro_errors.bias)
                                / result.gyro_sd.iloc[-1])
    assert (gyro_bias_relative_error < 2.0).all()

    accel_bias_relative_error = (np.abs(result.accel_estimates.iloc[-1] -
                                        accel_errors.bias)
                                 / result.accel_sd.iloc[-1])
    assert (accel_bias_relative_error < 2.0).all()

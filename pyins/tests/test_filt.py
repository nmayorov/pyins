from numpy.testing import assert_equal
import numpy as np
from pyins import filt, sim, strapdown, transform, util


def test_refine_stamps():
    stamps = [2, 2, 5, 1, 10, 20]
    stamps = filt._refine_stamps(stamps, 2)
    stamps_true = [1, 2, 4, 5, 7, 9, 10, 12, 14, 16, 18, 20]
    assert_equal(stamps, stamps_true)


def test_run_feedback_filter():
    dt = 0.01
    rng = np.random.RandomState(123456789)

    trajectory_true, imu_true = sim.sinusoid_velocity_motion(
        dt, 300, [50, 60, 100], [1, -1, 0.5], [3, 3, 0.5])

    position_obs = filt.PositionObs(
        sim.generate_position_observations(trajectory_true.iloc[::100], 1, rng), 1)
    ned_velocity_obs = filt.NedVelocityObs(
        sim.generate_ned_velocity_observations(trajectory_true.iloc[30::100], 0.5,
                                               rng), 0.5)
    body_velocity_obs = filt.BodyVelocityObs(
        sim.generate_body_velocity_observations(trajectory_true.iloc[60::100], 0.2,
                                                rng), 0.2)

    gyro_errors = sim.ImuErrors(
        bias=np.array([-100, 50, 40]) * transform.DH_TO_RS,
        noise=1 * transform.DRH_TO_RRS, rng=rng)
    accel_errors = sim.ImuErrors(bias=[0.1, -0.1, 0.2], noise=1.0 / 60, rng=rng)

    gyro_model = filt.InertialSensor(bias_sd=100 * transform.DH_TO_RS,
                                     noise=1 * transform.DRH_TO_RRS)
    accel_model = filt.InertialSensor(bias_sd=0.1, noise=1.0 / 60)

    imu = sim.apply_imu_errors(imu_true, 'increment', gyro_errors, accel_errors)
    increments = strapdown.compute_theta_and_dv(imu, 'increment')

    pos_sd = 10
    vel_sd = 2
    level_sd = 1.0
    azimuth_sd = 5.0

    initial = sim.perturb_trajectory_point(trajectory_true.iloc[0], pos_sd, vel_sd,
                                           level_sd, azimuth_sd, rng=rng)

    result = filt.run_feedback_filter(
        initial, pos_sd, vel_sd, level_sd, azimuth_sd, increments, gyro_model,
        accel_model, observations=[position_obs, ned_velocity_obs, body_velocity_obs],
        time_step=1)

    error = transform.compute_state_difference(result.trajectory, trajectory_true)

    assert (util.compute_rms(error / result.trajectory_sd) < 1.5).all()

    gyro_error = transform.compute_state_difference(result.gyro, gyro_errors.dataframe)
    assert ((gyro_error.iloc[-1] / result.gyro_sd.iloc[-1]).abs() < 2.0).all()

    accel_error = transform.compute_state_difference(result.accel,
                                                     accel_errors.dataframe)
    assert ((accel_error.iloc[-1] / result.accel_sd.iloc[-1]).abs() < 2.0).all()

    assert (util.compute_rms(result.innovations['PositionObs']) < 3.0).all()
    assert (util.compute_rms(result.innovations['NedVelocityObs']) < 3.0).all()
    assert (util.compute_rms(result.innovations['BodyVelocityObs']) < 3.0).all()


def test_run_feedforward_filter():
    dt = 0.01
    rng = np.random.RandomState(123456789)

    trajectory, imu_true = sim.sinusoid_velocity_motion(dt, 300, [50, 60, 100],
                                                        [1, -1, 0.5], [3, 3, 0.5])

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
    gyro_model = filt.InertialSensor(bias_sd=1 * transform.DH_TO_RS,
                                     noise=0.001 * transform.DRH_TO_RRS)
    accel_model = filt.InertialSensor(bias_sd=0.01, noise=0.01 / 60)

    imu = sim.apply_imu_errors(imu_true, 'increment', gyro_errors, accel_errors)
    increments = strapdown.compute_theta_and_dv(imu, 'increment')

    pos_sd = 10
    vel_sd = 0.1
    level_sd = 0.05
    azimuth_sd = 0.2

    initial = sim.perturb_trajectory_point(trajectory.iloc[0], pos_sd, vel_sd, level_sd,
                                           azimuth_sd, rng=rng)
    integrator = strapdown.Integrator(initial)
    trajectory_computed = integrator.integrate(increments)

    result = filt.run_feedforward_filter(
        trajectory, trajectory_computed, pos_sd, vel_sd, level_sd, azimuth_sd,
        gyro_model, accel_model,
        observations=[position_obs, ned_velocity_obs, body_velocity_obs], time_step=1)

    error = transform.compute_state_difference(result.trajectory, trajectory)

    relative_error = error / result.trajectory_sd
    assert (util.compute_rms(relative_error) < 1.6).all()

    gyro_bias_relative_error = (np.abs(result.gyro.iloc[-1] - gyro_errors.bias)
                                / result.gyro_sd.iloc[-1])
    assert (gyro_bias_relative_error < 2.0).all()

    accel_bias_relative_error = (np.abs(result.accel.iloc[-1] - accel_errors.bias)
                                 / result.accel_sd.iloc[-1])
    assert (accel_bias_relative_error < 2.0).all()

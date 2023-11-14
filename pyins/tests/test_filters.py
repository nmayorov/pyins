import numpy as np
from pyins import (inertial_sensor, filters, sim, strapdown, transform, util,
                   measurements)


def test_run_feedback_filter():
    dt = 0.01
    rng = np.random.RandomState(123456789)

    trajectory_true, imu_true = sim.sinusoid_velocity_motion(
        0.5 * dt, 300, [50, 60, 100], [1, -1, 0.5], [3, 3, 0.5], sensor_type='rate')

    position_obs = measurements.Position(
        sim.generate_position_measurements(trajectory_true.iloc[1::200], 1, rng), 1)
    ned_velocity_obs = measurements.NedVelocity(
        sim.generate_ned_velocity_measurements(trajectory_true.iloc[31::200], 0.5,
                                               rng), 0.5)
    body_velocity_obs = measurements.BodyVelocity(
        sim.generate_body_velocity_measurements(trajectory_true.iloc[61::200], 0.2,
                                                rng), 0.2)

    gyro_errors = inertial_sensor.InertialSensorError(
        bias=np.array([-100, 50, 40]) * transform.DH_TO_RS,
        noise=1 * transform.DRH_TO_RRS, rng=rng)
    accel_errors = inertial_sensor.InertialSensorError(bias=[0.1, -0.1, 0.2],
                                                       noise=1.0 / 60, rng=rng)

    gyro_model = inertial_sensor.InertialSensorModel(bias_sd=100 * transform.DH_TO_RS,
                                                     noise=1 * transform.DRH_TO_RRS)
    accel_model = inertial_sensor.InertialSensorModel(bias_sd=0.1, noise=1.0 / 60)

    imu = inertial_sensor.apply_imu_errors(imu_true.iloc[::2], 'rate',
                                           gyro_errors, accel_errors)
    increments = strapdown.compute_increments_from_imu(imu, 'rate')

    pos_sd = 10
    vel_sd = 2
    level_sd = 1.0
    azimuth_sd = 5.0

    pva_error = sim.generate_pva_error(pos_sd, vel_sd, level_sd, azimuth_sd, rng=rng)
    initial = sim.perturb_pva(trajectory_true.iloc[0], pva_error)

    result = filters.run_feedback_filter(
        initial, pos_sd, vel_sd, level_sd, azimuth_sd, increments, gyro_model,
        accel_model, measurements=[position_obs, ned_velocity_obs, body_velocity_obs],
        time_step=1)

    error = transform.compute_state_difference(result.trajectory, trajectory_true)

    assert (util.compute_rms(error / result.trajectory_sd) < 1.5).all()

    gyro_error = transform.compute_state_difference(result.gyro, gyro_errors.parameters)
    assert ((gyro_error.iloc[-1] / result.gyro_sd.iloc[-1]).abs() < 2.0).all()

    accel_error = transform.compute_state_difference(result.accel,
                                                     accel_errors.parameters)
    assert ((accel_error.iloc[-1] / result.accel_sd.iloc[-1]).abs() < 2.0).all()

    assert (util.compute_rms(result.innovations['Position']) < 3.0).all()
    assert (util.compute_rms(result.innovations['NedVelocity']) < 3.0).all()
    assert (util.compute_rms(result.innovations['BodyVelocity']) < 3.0).all()


def test_run_feedforward_filter():
    dt = 0.01
    factor = 5
    rng = np.random.RandomState(123456789)

    trajectory, imu_true = sim.sinusoid_velocity_motion(dt / factor, 300, [50, 60, 100],
                                                        [1, -1, 0.5], [3, 3, 0.5])

    position_obs = measurements.Position(
        sim.generate_position_measurements(trajectory.iloc[1::100 * factor], 1, rng), 1)
    ned_velocity_obs = measurements.NedVelocity(
        sim.generate_ned_velocity_measurements(trajectory.iloc[32::100 * factor], 0.5,
                                               rng), 0.5)
    body_velocity_obs = measurements.BodyVelocity(
        sim.generate_body_velocity_measurements(trajectory.iloc[64::100 * factor], 0.2,
                                                rng), 0.2)

    gyro_errors = inertial_sensor.InertialSensorError(
        bias=np.array([-1, 2, 1]) * transform.DH_TO_RS,
        noise=0.001 * transform.DRH_TO_RRS, rng=rng)
    accel_errors = inertial_sensor.InertialSensorError(bias=[0.01, -0.01, 0.02],
                                                       noise=0.01 / 60, rng=rng)
    gyro_model = inertial_sensor.InertialSensorModel(bias_sd=1 * transform.DH_TO_RS,
                                                     noise=0.001 * transform.DRH_TO_RRS)
    accel_model = inertial_sensor.InertialSensorModel(bias_sd=0.01, noise=0.01 / 60)

    imu = inertial_sensor.apply_imu_errors(imu_true, 'rate', gyro_errors, accel_errors)
    increments = strapdown.compute_increments_from_imu(imu, 'rate')

    pos_sd = 10
    vel_sd = 0.1
    level_sd = 0.05
    azimuth_sd = 0.2

    pva_error = sim.generate_pva_error(pos_sd, vel_sd, level_sd, azimuth_sd, rng=rng)
    initial = sim.perturb_pva(trajectory.iloc[0], pva_error)
    integrator = strapdown.Integrator(initial)
    trajectory_computed = integrator.integrate(increments)

    result = filters.run_feedforward_filter(
        trajectory.iloc[::factor], trajectory_computed.iloc[::factor],
        pos_sd, vel_sd, level_sd, azimuth_sd, gyro_model, accel_model,
        measurements=[position_obs, ned_velocity_obs, body_velocity_obs], time_step=1)

    error = transform.compute_state_difference(result.trajectory, trajectory)

    relative_error = error / result.trajectory_sd
    assert (util.compute_rms(relative_error) < 1.6).all()

    gyro_bias_relative_error = (np.abs(result.gyro.iloc[-1] - gyro_errors.bias)
                                / result.gyro_sd.iloc[-1])
    assert (gyro_bias_relative_error < 2.0).all()

    accel_bias_relative_error = (np.abs(result.accel.iloc[-1] - accel_errors.bias)
                                 / result.accel_sd.iloc[-1])
    assert (accel_bias_relative_error < 2.0).all()

    assert (util.compute_rms(result.innovations['Position']) < 3.0).all()
    assert (util.compute_rms(result.innovations['NedVelocity']) < 3.0).all()
    assert (util.compute_rms(result.innovations['BodyVelocity']) < 3.0).all()

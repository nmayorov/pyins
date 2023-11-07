import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from pyins import sim
from pyins import earth
from pyins.imu_model import InertialSensor
from pyins.util import GYRO_COLS, ACCEL_COLS


def test_sim_on_stationary():
    dt = 1e-1
    n_points = 1000

    time = dt * np.arange(n_points)

    lla = np.empty((n_points, 3))
    lla[:, 0] = 50.0
    lla[:, 1] = 45.0
    lla[:, 2] = 0.0
    hpr = np.zeros((n_points, 3))
    V_n = np.zeros((n_points, 3))

    slat = np.sin(np.deg2rad(50))
    clat = (1 - slat**2) ** 0.5

    gyro = earth.RATE * np.array([clat, 0, -slat])
    accel = np.array([0, 0, -earth.gravity(50, 0)])

    for lla_arg, velocity_n_arg in [(lla[0], V_n), (lla, V_n), (lla, None)]:
        for sensor_type in ['rate', 'increment']:
            trajectory, imu = sim.generate_imu(time, lla_arg, hpr, velocity_n_arg,
                                               sensor_type=sensor_type)

            assert_allclose(trajectory.lat, 50, rtol=1e-12)
            assert_allclose(trajectory.lon, 45, rtol=1e-12)
            assert_allclose(trajectory.VE, 0, atol=1e-7)
            assert_allclose(trajectory.VN, 0, atol=1e-7)
            assert_allclose(trajectory.roll, 0, atol=1e-8)
            assert_allclose(trajectory.pitch, 0, atol=1e-8)
            assert_allclose(trajectory.heading, 0, atol=1e-8)

            accel_atol = 1e-7 if sensor_type == 'increment' else 1e-6
            factor = dt if sensor_type == 'increment' else 1

            gyro_g = imu[GYRO_COLS].values
            accel_g = imu[ACCEL_COLS].values

            for i in range(3):
                assert_allclose(gyro_g[:, i], gyro[i] * factor, atol=1e-14)
                assert_allclose(accel_g[:, i], accel[i] * factor, atol=accel_atol)


def test_ImuErrors():
    rng = np.random.RandomState(0)
    readings = pd.DataFrame(data=rng.randn(100, 3), index=0.1 * np.arange(100))
    imu_errors = sim.ImuErrors(bias=[0.0, 0.2, 0.0])
    assert_allclose(imu_errors.apply(readings, 'increment'),
                    readings + [0.0, 0.2 * 0.1, 0.0], rtol=1e-14)
    assert_allclose(imu_errors.apply(readings, 'rate'), readings + [0.0, 0.2, 0.0],
                    rtol=1e-15)

    for sensor_type in ['rate', 'increment']:
        imu_errors = sim.ImuErrors()
        assert (imu_errors.apply(readings, sensor_type) == readings).all(None)

        imu_errors = sim.ImuErrors(bias_walk=0.1, rng=0)
        readings_with_error = imu_errors.apply(readings, sensor_type)
        diff = readings - readings_with_error
        n_readings = len(readings)
        assert (diff.iloc[:n_readings // 2].abs().mean() <
                diff.iloc[n_readings // 2:].abs().mean()).all()

        imu_errors = sim.ImuErrors(transform=np.diag([1.1, 1.0, 1.0]))
        readings_with_error = imu_errors.apply(readings, sensor_type)
        assert (readings_with_error[0] == 1.1 * readings[0]).all(None)
        assert (readings_with_error[[1, 2]] == readings[[1, 2]]).all(None)

        imu_errors = sim.ImuErrors(noise=[0.0, 0.0, 0.1])
        readings_with_error = imu_errors.apply(readings, sensor_type)
        assert (readings_with_error[[0, 1]] == readings[[0, 1]]).all(None)
        assert (readings_with_error[2] != readings[2]).all(None)


def test_ImuErrors_from_inertial_sensor_model():
    model = InertialSensor(bias_sd=[0.0, 1.0, 1.0],
                           scale_misal_sd=[[0.01, 0.0, 0.0],
                                           [0.0, 0.0, 0.01],
                                           [0.0, 0.01, 0.0]])
    imu_errors = sim.ImuErrors.from_inertial_sensor_model(model, 0)
    assert imu_errors.bias[0] == 0
    assert imu_errors.bias[1] != 0
    assert imu_errors.bias[2] != 0
    T = imu_errors.transform
    assert T[0, 0] != 1
    assert T[0, 1] == 0
    assert T[0, 2] == 0
    assert T[1, 0] == 0
    assert T[1, 1] == 1
    assert T[1, 2] != 0
    assert T[2, 0] == 0
    assert T[2, 1] != 0
    assert T[2, 2] == 1

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
from pyins.inertial_sensor import EstimationModel, Parameters


def test_EstimationModel():
    s = EstimationModel()
    assert_equal(s.n_states, 0)
    assert_equal(s.n_noises, 0)
    assert_equal(s.n_output_noises, 0)
    assert_equal(len(s.states), 0)
    assert_equal(s.P.shape, (0, 0))
    assert_equal(s.q.shape, (0,))
    assert_equal(s.v, [])
    assert_equal(s.F.shape, (0, 0))
    assert_equal(s.G.shape, (0, 0))
    assert_equal(s.output_matrix().shape, (3, 0))

    s = EstimationModel(bias_sd=0.1, bias_walk=0.2)
    assert_equal(s.n_states, 3)
    assert_equal(s.n_noises, 3)
    assert_equal(s.n_output_noises, 0)
    assert_equal(s.states, ['bias_x', 'bias_y', 'bias_z'])
    assert_allclose(s.P, 0.01 * np.identity(3))
    assert_equal(s.q, [0.2, 0.2, 0.2])
    assert_equal(s.v, [])
    assert_equal(s.F, np.zeros((3, 3)))
    assert_equal(s.G, np.identity(3))
    assert_equal(s.J, np.empty((3, 0)))
    assert_equal(s.output_matrix(), np.identity(3))

    s = EstimationModel(bias_sd=[0.1, 0.0, 0.2], bias_walk=[0.0, 0.0, 0.01],
                        noise=[0.0, 0.02, 0.0])
    assert_equal(s.n_states, 2)
    assert_equal(s.n_noises, 1)
    assert_equal(s.n_output_noises, 1)
    assert_equal(s.states, ['bias_x', 'bias_z'])
    assert_allclose(s.P, np.diag([0.1**2, 0.2**2]))
    assert_equal(s.q, [0.01])
    assert_equal(s.v, [0.02])
    assert_equal(s.F, np.zeros((2, 2)))
    assert_equal(s.G, [[0], [1]])
    assert_equal(s.J, [[0], [1], [0]])
    assert_equal(s.output_matrix(), [[1, 0], [0, 0], [0, 1]])

    s = EstimationModel(bias_sd=0.1, bias_walk=0.2,
                        scale_misal_sd=np.diag([0.3, 0.3, 0.3]))
    assert_equal(s.n_states, 6)
    assert_equal(s.n_noises, 3)
    assert_equal(s.n_output_noises, 0)
    assert_equal(s.states, ['bias_x', 'bias_y', 'bias_z', 'sm_xx', 'sm_yy', 'sm_zz'])
    assert_allclose(s.P, np.diag([0.01, 0.01, 0.01, 0.09, 0.09, 0.09]))
    assert_equal(s.q, [0.2, 0.2, 0.2])
    assert_equal(s.v, [])
    assert_allclose(s.F, np.zeros((6, 6)))
    assert_equal(s.G, np.vstack([np.identity(3), np.zeros((3, 3))]))

    H = s.output_matrix([1, 2, 3])
    assert_allclose(H, [[1, 0, 0, 1, 0, 0],
                        [0, 1, 0, 0, 2, 0],
                        [0, 0, 1, 0, 0, 3]])

    H = s.output_matrix([[1, 2, 3], [-1, 2, 0.5]])
    assert_allclose(H[0], [[1, 0, 0, 1, 0, 0],
                           [0, 1, 0, 0, 2, 0],
                           [0, 0, 1, 0, 0, 3]])
    assert_allclose(H[1], [[1, 0, 0, -1, 0, 0],
                           [0, 1, 0, 0, 2, 0],
                           [0, 0, 1, 0, 0, 0.5]])


def test_Parameters():
    rng = np.random.RandomState(0)
    readings = pd.DataFrame(data=rng.randn(100, 3), index=0.1 * np.arange(100))
    imu_errors = Parameters(bias=[0.0, 0.2, 0.0])
    assert_allclose(imu_errors.apply(readings, 'increment'),
                    readings + [0.0, 0.2 * 0.1, 0.0], rtol=1e-14)
    assert_allclose(imu_errors.apply(readings, 'rate'), readings + [0.0, 0.2, 0.0],
                    rtol=1e-15)

    for sensor_type in ['rate', 'increment']:
        imu_errors = Parameters()
        assert (imu_errors.apply(readings, sensor_type) == readings).all(None)

        imu_errors = Parameters(bias_walk=0.1, rng=0)
        readings_with_error = imu_errors.apply(readings, sensor_type)
        diff = readings - readings_with_error
        n_readings = len(readings)
        assert (diff.iloc[:n_readings // 2].abs().mean() <
                diff.iloc[n_readings // 2:].abs().mean()).all()

        imu_errors = Parameters(transform=np.diag([1.1, 1.0, 1.0]))
        readings_with_error = imu_errors.apply(readings, sensor_type)
        assert (readings_with_error[0] == 1.1 * readings[0]).all(None)
        assert (readings_with_error[[1, 2]] == readings[[1, 2]]).all(None)

        imu_errors = Parameters(noise=[0.0, 0.0, 0.1])
        readings_with_error = imu_errors.apply(readings, sensor_type)
        assert (readings_with_error[[0, 1]] == readings[[0, 1]]).all(None)
        assert (readings_with_error[2] != readings[2]).all(None)


def test_Parameters_from_EstimationModel():
    model = EstimationModel(bias_sd=[0.0, 1.0, 1.0],
                            scale_misal_sd=[[0.01, 0.0, 0.0],
                                                [0.0, 0.0, 0.01],
                                                [0.0, 0.01, 0.0]])
    imu_errors = Parameters.from_EstimationModel(model, 0)
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

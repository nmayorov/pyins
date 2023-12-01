import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
from pyins import util
from pyins.inertial_sensor import EstimationModel, Parameters


def test_EstimationModel():
    model = EstimationModel()
    assert_equal(model.n_states, 0)
    assert_equal(model.n_noises, 0)
    assert_equal(model.n_output_noises, 0)
    assert_equal(model.states, [])
    assert_equal(model.P.shape, (0, 0))
    assert_equal(model.q, [])
    assert_equal(model.v, [])
    assert_equal(model.F.shape, (0, 0))
    assert_equal(model.G.shape, (0, 0))
    assert_equal(model.J, [[], [], []])
    assert_equal(model.output_matrix(), [[], [], []])

    model = EstimationModel(bias_sd=0.1, bias_walk=0.2)
    assert_equal(model.n_states, 3)
    assert_equal(model.n_noises, 3)
    assert_equal(model.n_output_noises, 0)
    assert_equal(model.states, ['bias_x', 'bias_y', 'bias_z'])
    assert_allclose(model.P, 0.01 * np.identity(3))
    assert_equal(model.q, [0.2, 0.2, 0.2])
    assert_equal(model.v, [])
    assert_equal(model.F, np.zeros((3, 3)))
    assert_equal(model.G, np.identity(3))
    assert_equal(model.J, [[], [], []])
    assert_equal(model.output_matrix(), np.identity(3))

    model = EstimationModel(bias_sd=[0.1, 0.0, 0.2], bias_walk=[0.0, 0.0, 0.01],
                            noise=[0.0, 0.02, 0.0])
    assert_equal(model.n_states, 2)
    assert_equal(model.n_noises, 1)
    assert_equal(model.n_output_noises, 1)
    assert_equal(model.states, ['bias_x', 'bias_z'])
    assert_allclose(model.P, np.diag([0.1**2, 0.2**2]))
    assert_equal(model.q, [0.01])
    assert_equal(model.v, [0.02])
    assert_equal(model.F, np.zeros((2, 2)))
    assert_equal(model.G, [[0], [1]])
    assert_equal(model.J, [[0], [1], [0]])
    assert_equal(model.output_matrix(), [[1, 0], [0, 0], [0, 1]])

    model = EstimationModel(bias_sd=0.1, bias_walk=0.2,
                            scale_misal_sd=[
                                [0.1, 0.0, 0.0],
                                [0.0, 0.2, 0.15],
                                [0.0, 0.0, 0.05]
                            ])
    assert_equal(model.n_states, 7)
    assert_equal(model.n_noises, 3)
    assert_equal(model.n_output_noises, 0)
    assert_equal(model.states, ['bias_x', 'bias_y', 'bias_z',
                                'sm_xx', 'sm_yy', 'sm_yz', 'sm_zz'])
    assert_allclose(model.P, np.diag([0.01, 0.01, 0.01, 0.01, 0.04, 0.0225, 0.0025]))
    assert_equal(model.q, [0.2, 0.2, 0.2])
    assert_equal(model.v, [])
    assert_allclose(model.F, np.zeros((7, 7)))
    assert_equal(model.G, np.vstack([np.identity(3), np.zeros((4, 3))]))
    assert_equal(model.J, [[], [], []])
    assert_allclose(model.output_matrix([1, 2, 3]), [[1, 0, 0, 1, 0, 0, 0],
                                                     [0, 1, 0, 0, 2, 3, 0],
                                                     [0, 0, 1, 0, 0, 0, 3]])
    assert_allclose(model.output_matrix([[1, 2, 3], [-1, 2, 0.5]]),
                    [[[1, 0, 0, 1, 0, 0, 0],
                      [0, 1, 0, 0, 2, 3, 0],
                      [0, 0, 1, 0, 0, 0, 3]],
                     [[1, 0, 0, -1, 0,   0,   0],
                      [0, 1, 0,  0, 2, 0.5,   0],
                      [0, 0, 1,  0, 0,   0, 0.5]]
                     ])
    model.update_estimates([0.1, -0.2, 0.05, 0.01, -0.01, 0.02, 0.03])
    assert ((model.get_estimates() - pd.Series(
        [0.1, -0.2, 0.05, 0.01, -0.01, 0.02, 0.03],
        index=['bias_x', 'bias_y', 'bias_z', 'sm_xx', 'sm_yy', 'sm_yz', 'sm_zz']))
            .abs() < 1e-16).all()
    model.update_estimates([-0.02, 0.05, 0.06, 0.02, 0.01, -0.01, -0.015])
    assert ((model.get_estimates() - pd.Series(
        [0.08, -0.15, 0.11, 0.03, -0.0, 0.01, 0.015],
        index=['bias_x', 'bias_y', 'bias_z', 'sm_xx', 'sm_yy', 'sm_yz', 'sm_zz']))
            .abs() < 1e-15).all()

    increments = pd.Series([1, 2, 3], index=['dv_x', 'dv_y', 'dv_z'])
    incements_corrected = model.correct_increments(0.5, increments)
    bias = np.array([0.08, -0.15, 0.11])
    T = np.array([
        [1.03, 0.0, 0.0],
        [0.0, 1.0, 0.01],
        [0.0, 0.0, 1.015]
    ])
    assert_allclose(T @ incements_corrected + 0.5 * bias, increments)

    increments = pd.DataFrame([[1, 2, 3], [-1, 2, 0.5]],
                              columns=['theta_x', 'theta_y', 'theta_z'])
    incements_corrected = model.correct_increments(0.5, increments)
    assert_allclose(util.mv_prod(T, incements_corrected) + 0.5 * bias, increments)


def test_Parameters():
    rng = np.random.RandomState(0)
    readings = pd.DataFrame(data=rng.randn(100, 3), index=0.1 * np.arange(100))
    parameters = Parameters(bias=[-0.1, 0.2, 0.3])
    assert_allclose(parameters.apply(readings, 'increment'),
                    readings + [-0.1 * 0.1, 0.2 * 0.1, 0.3 * 0.1], rtol=1e-13)
    assert_allclose(parameters.apply(readings, 'rate'), readings + [-0.1, 0.2, 0.3],
                    rtol=1e-15)

    for sensor_type in ['rate', 'increment']:
        parameters = Parameters()
        assert (parameters.apply(readings, sensor_type) == readings).all(None)

        parameters = Parameters(bias_walk=0.1, rng=0)
        readings_with_error = parameters.apply(readings, sensor_type)
        diff = readings - readings_with_error
        n_readings = len(readings)
        assert (diff.iloc[:n_readings // 2].abs().mean() <
                diff.iloc[n_readings // 2:].abs().mean()).all()

        parameters = Parameters(transform=np.diag([1.1, 1.0, 1.0]))
        readings_with_error = parameters.apply(readings, sensor_type)
        assert (readings_with_error[0] == 1.1 * readings[0]).all(None)
        assert (readings_with_error[[1, 2]] == readings[[1, 2]]).all(None)

        parameters = Parameters(noise=[0.0, 0.0, 0.1])
        readings_with_error = parameters.apply(readings, sensor_type)
        assert (readings_with_error[[0, 1]] == readings[[0, 1]]).all(None)
        assert (readings_with_error[2] != readings[2]).all(None)


def test_Parameters_from_EstimationModel():
    model = EstimationModel(bias_sd=[0.0, 1.0, 1.0],
                            scale_misal_sd=[[0.01, 0.0, 0.0],
                                            [0.0, 0.0, 0.01],
                                            [0.0, 0.01, 0.0]])
    parameters = Parameters.from_EstimationModel(model, 0)
    assert parameters.bias[0] == 0
    assert parameters.bias[1] != 0
    assert parameters.bias[2] != 0
    T = parameters.transform
    assert T[0, 0] != 1
    assert T[0, 1] == 0
    assert T[0, 2] == 0
    assert T[1, 0] == 0
    assert T[1, 1] == 1
    assert T[1, 2] != 0
    assert T[2, 0] == 0
    assert T[2, 1] != 0
    assert T[2, 2] == 1

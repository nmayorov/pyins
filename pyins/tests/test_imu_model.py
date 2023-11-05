import numpy as np
from numpy.testing import assert_allclose, assert_equal
from pyins.imu_model import InertialSensor


def test_InertialSensor():
    s = InertialSensor()
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

    s = InertialSensor(bias=0.1, bias_walk=0.2)
    assert_equal(s.n_states, 3)
    assert_equal(s.n_noises, 3)
    assert_equal(s.n_output_noises, 0)
    assert_equal(list(s.states.keys()), ['bias_1', 'bias_2', 'bias_3'])
    assert_equal(list(s.states.values()), [0, 1, 2])
    assert_allclose(s.P, 0.01 * np.identity(3))
    assert_equal(s.q, [0.2, 0.2, 0.2])
    assert_equal(s.v, [])
    assert_equal(s.F, np.zeros((3, 3)))
    assert_equal(s.G, np.identity(3))
    assert_equal(s.J, np.empty((3, 0)))
    assert_equal(s.output_matrix(), np.identity(3))

    s = InertialSensor(bias=[0.1, 0.0, 0.2], bias_walk=[0.0, 0.0, 0.01],
                       noise=[0.0, 0.02, 0.0])
    assert_equal(s.n_states, 2)
    assert_equal(s.n_noises, 1)
    assert_equal(s.n_output_noises, 1)
    assert_equal(list(s.states.keys()), ['bias_1', 'bias_3'])
    assert_equal(list(s.states.values()), [0, 1])
    assert_allclose(s.P, np.diag([0.1**2, 0.2**2]))
    assert_equal(s.q, [0.01])
    assert_equal(s.v, [0.02])
    assert_equal(s.F, np.zeros((2, 2)))
    assert_equal(s.G, [[0], [1]])
    assert_equal(s.J, [[0], [1], [0]])
    assert_equal(s.output_matrix(), [[1, 0], [0, 0], [0, 1]])

    s = InertialSensor(bias=0.1, bias_walk=0.2,
                       scale_misal=np.diag([0.3, 0.3, 0.3]))
    assert_equal(s.n_states, 6)
    assert_equal(s.n_noises, 3)
    assert_equal(s.n_output_noises, 0)
    assert_equal(list(s.states.keys()),
                 ['bias_1', 'bias_2', 'bias_3', 'sm_11', 'sm_22', 'sm_33'])
    assert_equal(list(s.states.values()), np.arange(6))
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

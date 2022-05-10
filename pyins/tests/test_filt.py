from numpy.testing import (assert_, assert_allclose, run_module_suite,
                           assert_equal)
import numpy as np
import pandas as pd
from pyins.filt import (InertialSensor, PositionObs,
                        FeedforwardFilter, FeedbackFilter,
                        _refine_stamps, correct_trajectory)
from pyins.error_models import propagate_errors
from pyins import sim
from pyins.strapdown import compute_theta_and_dv, StrapdownIntegrator
from pyins.transform import perturb_lla, difference_trajectories


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


def test_FeedforwardFilter():
    # Test that the results are reasonable on a static bench.
    dt = 1
    traj = pd.DataFrame(index=np.arange(1 * 3600))
    traj['lat'] = 50
    traj['lon'] = 60
    traj['alt'] = 100
    traj['VN'] = 0
    traj['VE'] = 0
    traj['VD'] = 0
    traj['roll'] = 0
    traj['pitch'] = 0
    traj['heading'] = 0

    np.random.seed(1)
    obs_data = pd.DataFrame(index=traj.index[::10])
    obs_data[['lat', 'lon', 'alt']] = perturb_lla(
        traj.loc[::10, ['lat', 'lon', 'alt']],
        10 * np.random.randn(len(obs_data), 3))
    position_obs = PositionObs(obs_data, 10)

    delta_position_n = [-3, 5, 0]
    delta_velocity_n = [1, -1, 0]
    delta_rph = [-0.02, 0.03, 0.1]

    errors, _ = propagate_errors(dt, traj, delta_position_n, delta_velocity_n,
                                 delta_rph)
    traj_error = correct_trajectory(traj, -errors)

    f = FeedforwardFilter(dt, traj, 5, 1, 0.2, 0.05)
    res = f.run(traj_error, [position_obs])

    x = errors.loc[3000:]
    y = res.err.loc[3000:]

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
    y = res.err

    assert_allclose(x.east, y.east, rtol=0, atol=10)
    assert_allclose(x.north, y.north, rtol=0, atol=10)
    assert_allclose(x.VE, y.VE, rtol=0, atol=2e-2)
    assert_allclose(x.VN, y.VN, rtol=0, atol=2e-2)
    assert_allclose(x.roll, y.roll, rtol=0, atol=1e-4)
    assert_allclose(x.pitch, y.pitch, rtol=0, atol=1e-4)
    assert_allclose(x.heading, y.heading, rtol=0, atol=1.5e-3)
    assert_(np.all(np.abs(res.residuals[0] < 4)))


def test_FeedbackFilter():
    dt = 0.9
    traj = pd.DataFrame(index=np.arange(1 * 3600))
    traj['lat'] = 50
    traj['lon'] = 60
    traj['alt'] = 100
    traj['VE'] = 0
    traj['VN'] = 0
    traj['VU'] = 0
    traj['roll'] = 0
    traj['pitch'] = 0
    traj['heading'] = 0

    _, gyro, accel = sim.from_position(dt, traj[['lat', 'lon', 'alt']],
                                       traj[['roll', 'pitch', 'heading']])
    theta, dv = compute_theta_and_dv(gyro, accel)

    np.random.seed(0)
    obs_data = pd.DataFrame(index=traj.index[::10])
    obs_data[['lat', 'lon', 'alt']] = perturb_lla(
        traj.loc[::10, ['lat', 'lon', 'alt']],
        10 * np.random.randn(len(obs_data), 3))
    position_obs = PositionObs(obs_data, 10)

    f = FeedbackFilter(dt, 5, 1, 0.2, 0.05)

    d_lat = 5
    d_lon = -3
    d_alt = 0
    d_VE = 1
    d_VN = -1
    d_VU = 0
    d_r = -0.02
    d_p = 0.03
    d_h = 0.1

    lla0 = perturb_lla(traj.loc[0, ['lat', 'lon', 'alt']],
                       [d_lon, d_lat, d_alt])
    integrator = StrapdownIntegrator(dt, lla0, [d_VE, d_VN, d_VU],
                                     [d_r, d_p, d_h])
    res = f.run(integrator, theta, dv, observations=[position_obs])
    error = difference_trajectories(res.traj, traj)
    error = error.iloc[3000:]

    assert_allclose(error.east, 0, rtol=0, atol=10)
    assert_allclose(error.north, 0, rtol=0, atol=10)
    assert_allclose(error.VE, 0, rtol=0, atol=2e-2)
    assert_allclose(error.VN, 0, rtol=0, atol=2e-2)
    assert_allclose(error.heading, 0, rtol=0, atol=2e-3)
    assert_allclose(error.pitch, 0, rtol=0, atol=1e-4)
    assert_allclose(error.roll, 0, rtol=0, atol=1e-4)
    assert_(np.all(np.abs(res.residuals[0] < 4)))

    res = f.run_smoother(integrator, theta, dv, [position_obs])
    error = difference_trajectories(res.traj, traj)
    assert_allclose(error.east, 0, rtol=0, atol=10)
    assert_allclose(error.north, 0, rtol=0, atol=10)
    assert_allclose(error.VE, 0, rtol=0, atol=2e-2)
    assert_allclose(error.VN, 0, rtol=0, atol=2e-2)
    assert_allclose(error.roll, 0, rtol=0, atol=1e-4)
    assert_allclose(error.pitch, 0, rtol=0, atol=1e-4)
    assert_allclose(error.heading, 0, rtol=0, atol=1.5e-3)
    assert_(np.all(np.abs(res.residuals[0] < 4)))


if __name__ == '__main__':
    run_module_suite()

import numpy as np
from numpy.testing import assert_allclose
from pyins import sim
from pyins.integrate import coning_sculling, Integrator
from pyins.error_model import propagate_errors
from pyins.transform import perturb_ll
from pyins.filt import traj_diff


def test_propagate_errors():
    # This test is complex and hardly a unit test, but it is strong.
    # I believe it's better than a formal test.
    dt = 0.5
    t = 0.5 * 3600
    n_samples = int(t / dt)
    lat = np.full(n_samples, 50.0)
    lon = np.full(n_samples, 60.0)
    alt = np.zeros_like(lat)
    h = np.full(n_samples, 10.0)
    r = np.full(n_samples, -5.0)
    p = np.full(n_samples, 3.0)

    traj, gyro, accel = sim.from_position(dt, lat, lon, alt, h, p, r)

    gyro_bias = np.array([1e-8, -2e-8, 3e-8])
    accel_bias = np.array([3e-3, -4e-3, 2e-3])

    gyro += gyro_bias * dt
    accel += accel_bias * dt
    theta, dv = coning_sculling(gyro, accel)

    d_lat = 100
    d_lon = -200
    d_alt = 20.0
    d_VE = 1
    d_VN = -2
    d_VU = -0.5
    d_h = 0.01
    d_p = -0.02
    d_r = 0.03

    lat0, lon0 = perturb_ll(traj.lat[0], traj.lon[0], d_lat, d_lon)
    alt0 = alt[0] + d_alt
    VE0 = traj.VE[0] + d_VE
    VN0 = traj.VN[0] + d_VN
    VU0 = traj.VU[0] + d_VU
    h0 = traj.h[0] + d_h
    p0 = traj.p[0] + d_p
    r0 = traj.r[0] + d_r

    integrator = Integrator(dt, [lat0, lon0, alt0], [VE0, VN0, VU0],
                            [h0, p0, r0])
    traj_c = integrator.integrate(theta, dv)
    error_true = traj_diff(traj_c, traj)
    error_linear = propagate_errors(dt, traj, d_lat, d_lon, d_alt,
                                    d_VE, d_VN, d_VU, d_h, d_p, d_r,
                                    gyro_bias, accel_bias)

    error_scale = np.mean(np.abs(error_true))
    rel_diff = (error_linear - error_true) / error_scale
    assert_allclose(rel_diff, 0, atol=0.1)

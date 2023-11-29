import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation
from pyins import earth, transform


def test_principal_radii():
    lat = 0
    rn, re, rp = earth.principal_radii(lat, 0)
    assert_allclose(re, earth.A, rtol=1e-10)
    assert_allclose(rn, earth.A * (1 - earth.E2), rtol=1e-10)
    assert_allclose(rp, earth.A, rtol=1e-10)

    lat = [0, 90]
    rn, re, rp = earth.principal_radii(lat, 0)
    assert_allclose(re[0], earth.A, rtol=1e-10)
    assert_allclose(rn[0], earth.A * (1 - earth.E2), rtol=1e-10)
    assert_allclose(re[1], rn[1], rtol=1e-10)
    assert_allclose(rp[0], earth.A, rtol=1e-10)
    assert_allclose(rp[1], 0, atol=1e-10)


def test_gravity():
    # Another smoke test.
    g = earth.gravity(0, 0)
    assert_allclose(g, 9.7803253359, rtol=1e-10)

    g = earth.gravity(90, 0)
    assert_allclose(g, 9.8321849378, rtol=1e-10)

    g = earth.gravity(0, 0.5)
    assert_allclose(g, 9.7803253359 * (1 - 1 / earth.A), rtol=1e-10)

    g = earth.gravity([0, 0], [0, 0.5])
    assert_allclose(g, [9.7803253359, 9.7803253359 * (1 - 1 / earth.A)], rtol=1e-10)


def test_gravity_n():
    assert_allclose(earth.gravity_n(0, 0), [0, 0, earth.GE])
    assert_allclose(earth.gravity_n([0, 90, -90], 0),
                    [[0, 0, earth.GE], [0, 0, earth.GP], [0, 0, earth.GP]])


def test_gravitation_ecef():
    g = earth.gravity(90, 100)
    g0_e = earth.gravitation_ecef([90, 0, 100])
    assert_allclose(g0_e, [0, 0, -g], atol=1e-12)

    g = earth.gravity(0, 0)
    g0_e = earth.gravitation_ecef([0, 90, 0])
    assert_allclose(g0_e, [0, -g - earth.RATE ** 2 * earth.A, 0], atol=1e-12)

    g0_true = [[0, 0, -earth.gravity(90, 100)],
               [0, -earth.gravity(0, 0) - earth.RATE ** 2 * earth.A, 0]]
    assert_allclose(earth.gravitation_ecef([[90, 0, 100], [0, 90, 0]]),
                    g0_true, atol=1e-12)


def test_curvature_matrix():
    lla_0 = [43, 12, 0]
    dr_n = [1, -2, 0]
    lla_1 = transform.perturb_lla(lla_0, dr_n)

    mat_en_1 = transform.mat_en_from_ll(lla_0[0], lla_0[1])
    mat_en_2 = transform.mat_en_from_ll(lla_1[0], lla_1[1])
    rot_n1_n2 = Rotation.from_matrix(mat_en_1.T @ mat_en_2).as_rotvec()

    F = earth.curvature_matrix(lla_0[0], lla_0[1])
    assert_allclose(F @ dr_n, rot_n1_n2, rtol=1e-5)

    F = F[None, :, :]
    assert_allclose(earth.curvature_matrix([lla_0[0], lla_0[0]], [lla_0[1], lla_0[1]]),
                    np.vstack((F, F)))


def test_rate_n():
    assert_allclose(earth.rate_n(0), [earth.RATE, 0, 0], atol=1e-20)
    assert_allclose(earth.rate_n(90), [0, 0, -earth.RATE], atol=1e-20)
    assert_allclose(earth.rate_n([0, 90]), [[earth.RATE, 0, 0], [0, 0, -earth.RATE]],
                    atol=1e-20)

from numpy.testing import assert_allclose, run_module_suite
from pyins import earth


def test_principal_radii():
    lat = 0
    re, rn = earth.principal_radii(lat)
    assert_allclose(re, earth.R0, rtol=1e-10)
    assert_allclose(rn, earth.R0 * (1 - earth.E2), rtol=1e-10)

    lat = [0, 90]
    re, rn = earth.principal_radii(lat)
    assert_allclose(re[0], earth.R0, rtol=1e-10)
    assert_allclose(rn[0], earth.R0 * (1 - earth.E2), rtol=1e-10)
    assert_allclose(re[1], rn[1], rtol=1e-10)


def test_gravity():
    # Another smoke test.
    g = earth.gravity(0)
    assert_allclose(g, 9.7803253359, rtol=1e-10)

    g = earth.gravity(90)
    assert_allclose(g, 9.8321849378, rtol=1e-10)

    g = earth.gravity(0, 0.5)
    assert_allclose(g, 9.7803253359 * (1 - 1 / earth.R0), rtol=1e-10)

    g = earth.gravity([0, 0], [0, 0.5])
    assert_allclose(g, [9.7803253359, 9.7803253359 * (1 - 1 / earth.R0)],
                    rtol=1e-10)


def test_gravitation_ecef():
    g = earth.gravity(90, 100)
    g0_e = earth.gravitation_ecef([90, 0, 100])
    assert_allclose(g0_e, [0, 0, -g], atol=1e-12)

    g = earth.gravity(0)
    g0_e = earth.gravitation_ecef([0, 90, 0])
    assert_allclose(g0_e, [0, -g - earth.RATE**2 * earth.R0, 0],
                    atol=1e-12)

    g0_true = [[0, 0, -earth.gravity(90, 100)],
               [0, -earth.gravity(0) - earth.RATE**2 * earth.R0, 0]]
    assert_allclose(earth.gravitation_ecef([[90, 0, 100], [0, 90, 0]]),
                    g0_true, atol=1e-12)


if __name__ == '__main__':
    run_module_suite()

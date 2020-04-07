from numpy.testing import assert_allclose, run_module_suite
from pyins.transform import lla_to_ecef, perturb_ll
from pyins import earth


def test_lla_to_ecef():
    r_e = lla_to_ecef(0, 0, 10)
    assert_allclose(r_e, [earth.R0 + 10, 0, 0])

    r_e = lla_to_ecef(-90, 0, -10)
    b = (1 - earth.E2) ** 0.5 * earth.R0
    assert_allclose(r_e, [0, 0, -b + 10], atol=1e-9)

    r_e = lla_to_ecef([0, -90], [0, 0], [10, -10])
    assert_allclose(r_e, [[earth.R0 + 10, 0, 0],
                          [0, 0, -b + 10]], atol=1e-9)


def test_perturb_ll():
    lat = 40
    lon = 50
    lat_new, lon_new = perturb_ll(lat, lon, 10, -20)
    lat_new, lon_new = perturb_ll(lat_new, lon_new, -10, 20)
    assert_allclose([lat_new, lon_new], [lat, lon], rtol=1e-11)


if __name__ == '__main__':
    run_module_suite()

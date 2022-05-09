import numpy as np
from numpy.testing import assert_allclose, run_module_suite
from pyins.transform import lla_to_ecef, perturb_lla, phi_to_delta_rph
from pyins import earth, dcm


def test_lla_to_ecef():
    r_e = lla_to_ecef([0, 0, 10])
    assert_allclose(r_e, [earth.R0 + 10, 0, 0])

    r_e = lla_to_ecef([-90, 0, -10])
    b = (1 - earth.E2) ** 0.5 * earth.R0
    assert_allclose(r_e, [0, 0, -b + 10], atol=1e-9)

    r_e = lla_to_ecef([[0, 0, 10], [-90, 0, -10]])
    assert_allclose(r_e, [[earth.R0 + 10, 0, 0],
                          [0, 0, -b + 10]], atol=1e-9)


def test_perturb_ll():
    lla = [40, 50, 0]
    lla_new = perturb_lla(lla, [10, -20, 5])
    lla_back = perturb_lla(lla_new, [-10, 20, -5])
    assert_allclose(lla_back, lla, rtol=1e-11)


def test_phi_to_delta_rph():
    rph = [10, -20, 30]
    mat = dcm.from_rph(rph)
    phi = np.array([0.01, -0.02, 0.03])
    mat_perturbed = dcm.from_rv(-phi) @ mat

    rph_perturbed = dcm.to_rph(mat_perturbed)
    delta_rph_true = rph_perturbed - rph

    T = phi_to_delta_rph(rph)
    delta_rph_linear = np.rad2deg(T @ phi)

    assert_allclose(delta_rph_linear, delta_rph_true, rtol=1e-1)


if __name__ == '__main__':
    run_module_suite()

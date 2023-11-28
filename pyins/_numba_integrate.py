import numba
import numpy as np
from . import earth, transform


@numba.njit()
def gravity(lat, alt):
    sin2 = np.sin(lat * transform.DEG_TO_RAD) ** 2
    return (earth.GE * (1 + earth.F * sin2) /
            (1 - earth.E2 * sin2) ** 0.5 * (1 - 2 * alt / earth.A))


@numba.njit
def mat_from_rotvec(rv, mat):
    norm2 = np.sum(rv ** 2)
    if norm2 > 1e-6:
        norm = norm2 ** 0.5
        cos = np.cos(norm)
        k1 = np.sin(norm) / norm
        k2 = (1 - np.cos(norm)) / norm2
    else:
        norm4 = norm2 * norm2
        cos = 1 - norm2 / 2 + norm4 / 24
        k1 = 1 - norm2 / 6 + norm4 / 120
        k2 = 0.5 - norm2 / 24 + norm4 / 720

    mat[0, 0] = k2 * rv[0] * rv[0] + cos
    mat[0, 1] = k2 * rv[0] * rv[1] - k1 * rv[2]
    mat[0, 2] = k2 * rv[0] * rv[2] + k1 * rv[1]
    mat[1, 0] = k2 * rv[1] * rv[0] + k1 * rv[2]
    mat[1, 1] = k2 * rv[1] * rv[1] + cos
    mat[1, 2] = k2 * rv[1] * rv[2] - k1 * rv[0]
    mat[2, 0] = k2 * rv[2] * rv[0] - k1 * rv[1]
    mat[2, 1] = k2 * rv[2] * rv[1] + k1 * rv[0]
    mat[2, 2] = k2 * rv[2] * rv[2] + cos


@numba.njit
def integrate(dt_array, lla, velocity_n, mat_nb, theta, dv, offset, with_altitude):
    xi = np.empty(3)
    dv_n = np.empty(3)
    C = np.empty((3, 3))
    dBn = np.empty((3, 3))
    dBb = np.empty((3, 3))

    for i in range(len(theta)):
        j = i + offset
        dt = dt_array[i]

        lat = lla[j, 0]
        alt = lla[j, 2]

        sin_lat = np.sin(lat * transform.DEG_TO_RAD)
        cos_lat = np.sqrt(1 - sin_lat * sin_lat)
        tan_lat = sin_lat / cos_lat

        x = 1 - earth.E2 * sin_lat * sin_lat
        re = earth.A / x ** 0.5
        rn = re * (1 - earth.E2) / x + alt
        re += alt

        Omega1 = earth.RATE * cos_lat
        Omega2 = 0.0
        Omega3 = -earth.RATE * sin_lat

        V1 = velocity_n[j, 0]
        V2 = velocity_n[j, 1]
        V3 = velocity_n[j, 2]

        rho1 = V2 / re
        rho2 = -V1 / rn
        rho3 = -rho1 * tan_lat
        chi1 = Omega1 + rho1
        chi2 = Omega2 + rho2
        chi3 = Omega3 + rho3

        np.dot(mat_nb[j], dv[i], dv_n)
        dv1 = dv_n[0]
        dv2 = dv_n[1]
        dv3 = dv_n[2]

        velocity_n[j + 1, 0] = V1 + dv1 + (- (chi2 + Omega2) * V3
                                           + (chi3 + Omega3) * V2
                                           - 0.5 * (chi2 * dv3 - chi3 * dv2)
                                           ) * dt
        velocity_n[j + 1, 1] = V2 + dv2 + (- (chi3 + Omega3) * V1
                                           + (chi1 + Omega1) * V3
                                           - 0.5 * (chi3 * dv1 - chi1 * dv3)
                                           ) * dt
        if with_altitude:
            velocity_n[j + 1, 2] = V3 + dv3 + (- (chi1 + Omega1) * V2
                                               + (chi2 + Omega2) * V1
                                               - 0.5 * (chi1 * dv2 - chi2 * dv1)
                                               + gravity(lat, alt - 0.5 * V3 * dt)
                                               ) * dt
        else:
            velocity_n[j + 1, 2] = 0.0

        V1 = 0.5 * (V1 + velocity_n[j + 1, 0])
        V2 = 0.5 * (V2 + velocity_n[j + 1, 1])
        V3 = 0.5 * (V3 + velocity_n[j + 1, 2])
        rho1 = V2 / re
        rho2 = -V1 / rn
        rho3 = -rho1 * tan_lat
        chi1 = Omega1 + rho1
        chi2 = Omega2 + rho2
        chi3 = Omega3 + rho3

        lla[j + 1, 0] = lla[j, 0] - transform.RAD_TO_DEG * rho2 * dt
        lla[j + 1, 1] = lla[j, 1] + transform.RAD_TO_DEG * rho1 / cos_lat * dt
        lla[j + 1, 2] = lla[j, 2] - V3 * dt

        xi[0] = -chi1 * dt
        xi[1] = -chi2 * dt
        xi[2] = -chi3 * dt
        mat_from_rotvec(xi, dBn)
        mat_from_rotvec(theta[i], dBb)
        np.dot(mat_nb[j], dBb, C)
        np.dot(dBn, C, mat_nb[j + 1])


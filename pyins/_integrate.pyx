# cython : boundscheck=False, wraparound=False, language_level=3, cdivision=True

"""Strapdown integration implemented in Cython."""

import numpy as np
cimport cython
from scipy.linalg.cython_blas cimport dgemv, dgemm
from libc cimport math

cdef double DEG2RAD = math.pi / 180.0
cdef double RAD2DEG = 180.0 / math.pi
cdef double RATE = 7.2921157e-5
cdef double R0 = 6378137.0
cdef double E2 = 6.6943799901413e-3
cdef double GE = 9.7803253359
cdef double GP = 9.8321849378
cdef double F = (1 - E2) ** 0.5 * GP / GE - 1


cdef dcm_from_rotvec(double[:] rv, double[:, :] dcm):
    cdef double norm, norm2, norm4, cos, k1, k2

    norm2 = rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]
    if norm2 > 1e-6:
        norm = math.sqrt(norm2)
        cos = math.cos(norm)
        k1 = math.sin(norm) / norm
        k2 = (1 - math.cos(norm)) / norm2
    else:
        norm4 = norm2 * norm2
        cos = 1 - norm2 / 2 + norm4 / 24
        k1 = 1 - norm2 / 6 + norm4 / 120
        k2 = 0.5 - norm2 / 24 + norm4 / 720

    dcm[0, 0] = k2 * rv[0] * rv[0] + cos
    dcm[0, 1] = k2 * rv[0] * rv[1] - k1 * rv[2]
    dcm[0, 2] = k2 * rv[0] * rv[2] + k1 * rv[1]
    dcm[1, 0] = k2 * rv[1] * rv[0] + k1 * rv[2]
    dcm[1, 1] = k2 * rv[1] * rv[1] + cos
    dcm[1, 2] = k2 * rv[1] * rv[2] - k1 * rv[0]
    dcm[2, 0] = k2 * rv[2] * rv[0] - k1 * rv[1]
    dcm[2, 1] = k2 * rv[2] * rv[1] + k1 * rv[0]
    dcm[2, 2] = k2 * rv[2] * rv[2] + cos


cdef mv(double[:, :] A, double[:] b, double[:] ret):
    cdef int m = A.shape[0]
    cdef int n = A.shape[1]
    cdef int one = 1
    cdef int zero = 0
    cdef double dzero = 0
    cdef double done = 1
    dgemv('N', &m, &n, &done, &A[0, 0], &m, &b[0], &one, &dzero, &ret[0], &one)


cdef mm(double[:, :] A, double[:, :] B, double[:, :] ret):
    cdef int m = A.shape[0]
    cdef int n = A.shape[1]
    cdef int k = B.shape[0]
    cdef int one = 1
    cdef int zero = 0
    cdef double dzero = 0
    cdef double done = 1
    dgemm('N', 'N', &m, &n, &k, &done,
          &A[0, 0], &m,
          &B[0, 0], &n,
          &dzero,
          &ret[0, 0], &m)


cdef double gravity(double lat, double alt):
    s2 = math.sin(lat * DEG2RAD) ** 2
    return GE * (1 + F * s2) / (1 - E2 * s2) ** 0.5 * (1 - 2 * alt / R0)


def integrate_fast(double dt, double[:, :] lla, double[:, :] velocity_n,
                   double[:, :, :] Cnb, double[:, ::1] theta, double[:, ::1] dv,
                   int offset, bint with_altitude):
    cdef int i, j
    cdef double lat, alt
    cdef double slat, clat, tlat
    cdef double re, rn
    cdef double x

    cdef double[:] xi = np.empty(3)
    cdef double[:] dv_n = np.empty(3)

    cdef double[:, :] B = Cnb[offset].copy_fortran()
    cdef double[::1, :] C = np.empty((3, 3), order='F')
    cdef double[::1, :] dBn = np.empty((3, 3), order='F')
    cdef double[::1, :] dBb = np.empty((3, 3), order='F')

    cdef double V1, V2, V3
    cdef double dv1, dv2, dv3
    cdef double Omega1, Omega2, Omega3
    cdef double rho1, rho2, rho3
    cdef double chi1, chi2, chi3

    for i in range(theta.shape[0]):
        j = i + offset

        lat = lla[j, 0]
        alt = lla[j, 2]

        slat = math.sin(lat * DEG2RAD)
        clat = math.sqrt(1 - slat * slat)
        tlat = slat / clat

        x = 1 - E2 * slat * slat
        re = R0 / math.sqrt(x) + alt
        rn = re * (1 - E2) / x + alt

        Omega1 = RATE * clat
        Omega2 = 0.0
        Omega3 = -RATE * slat

        V1 = velocity_n[j, 0]
        V2 = velocity_n[j, 1]
        V3 = velocity_n[j, 2]

        rho1 = V2 / re
        rho2 = -V1 / rn
        rho3 = -rho1 * tlat
        chi1 = Omega1 + rho1
        chi2 = Omega2 + rho2
        chi3 = Omega3 + rho3

        mv(B, dv[i], dv_n)
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
        rho3 = -rho1 * tlat
        chi1 = Omega1 + rho1
        chi2 = Omega2 + rho2
        chi3 = Omega3 + rho3

        lla[j + 1, 0] = lla[j, 0] - RAD2DEG * rho2 * dt
        lla[j + 1, 1] = lla[j, 1] + RAD2DEG * rho1 / clat * dt
        lla[j + 1, 2] = lla[j, 2] - V3 * dt

        xi[0] = -chi1 * dt
        xi[1] = -chi2 * dt
        xi[2] = -chi3 * dt
        dcm_from_rotvec(xi, dBn)
        dcm_from_rotvec(theta[i], dBb)
        mm(B, dBb, C)
        mm(dBn, C, B)
        Cnb[j + 1] = B.copy()

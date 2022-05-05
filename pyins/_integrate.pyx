# cython : boundscheck=False, wraparound=False

"""Strapdown integration implemented in Cython."""

import numpy as np
cimport cython
from scipy.linalg.cython_blas cimport dgemv, dgemm
from libc cimport math


cdef double RATE = 7.2921157e-5
cdef double R0 = 6378137.0
cdef double E2 = 6.6943799901413e-3


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


cdef cross(double[:] a, double[:] b, double[:] ret):
    ret[0] = a[1] * b[2] - a[2] * b[1]
    ret[1] = a[2] * b[0] - a[0] * b[2]
    ret[2] = a[0] * b[1] - a[1] * b[0]


def integrate_fast(double dt, double[:, :] lla, double[:, :] velocity_n,
                   double[:, :, :] Cnb, double[:, ::1] theta, double[:, ::1] dv,
                   int offset=0):
    cdef int i, j
    cdef double slat, clat, tlat
    cdef double re, rn
    cdef double x

    cdef double[:] xi = np.empty(3)
    cdef double[:] dv_n = np.empty(3)

    cdef double[:, :] B = Cnb[offset].copy_fortran()
    cdef double[::1, :] C = np.empty((3, 3), order='F')
    cdef double[::1, :] dBn = np.empty((3, 3), order='F')
    cdef double[::1, :] dBb = np.empty((3, 3), order='F')

    cdef double VE, VN
    cdef double dv1, dv2, dv3
    cdef double u1 = 0
    cdef double u2, u3
    cdef double rho1, rho2, rho3
    cdef double omega1, omega2, omega3

    for i in range(theta.shape[0]):
        j = i + offset
        slat = math.sin(lla[j, 0])
        clat = math.sqrt(1 - slat * slat)
        tlat = slat / clat

        x = 1 - E2 * slat * slat
        re = R0 / math.sqrt(x)
        rn = re * (1 - E2) / x
        u2 = RATE * clat
        u3 = RATE * slat

        VE = velocity_n[j, 0]
        VN = velocity_n[j, 1]

        rho1 = -VN / rn
        rho2 = VE / re
        rho3 = rho2 * tlat
        omega1 = u1 + rho1
        omega2 = u2 + rho2
        omega3 = u3 + rho3

        mv(B, dv[i], dv_n)
        dv1 = dv_n[0]
        dv2 = dv_n[1]
        dv3 = dv_n[2]
        x = 2 * u3 + rho3
        velocity_n[j + 1, 0] = (VE + dv1 + dt *
                                (x * VN - 0.5 * (omega2 * dv3 - omega3 * dv2)))
        velocity_n[j + 1, 1] = (VN + dv2 - dt *
                                (x * VE + 0.5 * (omega3 * dv1 - omega1 * dv3)))
        velocity_n[j + 1, 2] = velocity_n[j, 2]

        VE = 0.5 * (VE + velocity_n[j + 1, 0])
        VN = 0.5 * (VN + velocity_n[j + 1, 1])
        rho1 = -VN / rn
        rho2 = VE / re
        rho3 = rho2 * tlat
        omega1 = u1 + rho1
        omega2 = u2 + rho2
        omega3 = u3 + rho3

        lla[j + 1, 0] = lla[j, 0] - rho1 * dt
        lla[j + 1, 1] = lla[j, 1] + rho2 / clat * dt
        lla[j + 1, 2] = lla[j, 2]

        xi[0] = -omega1 * dt
        xi[1] = -omega2 * dt
        xi[2] = -omega3 * dt
        dcm_from_rotvec(xi, dBn)
        dcm_from_rotvec(theta[i], dBb)
        mm(B, dBb, C)
        mm(dBn, C, B)
        Cnb[j + 1] = B.copy()

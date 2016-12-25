# cython: boundscheck=False, wraparound=False
import numpy as np


cdef inline void mat_vec(double[:, ::1] A, double[::1] b, double[::1] ret):
    ret[0] = A[0, 0] * b[0] + A[0, 1] * b[1] + A[0, 2] * b[2]
    ret[1] = A[1, 0] * b[0] + A[1, 1] * b[1] + A[1, 2] * b[2]
    ret[2] = A[2, 0] * b[0] + A[2, 1] * b[1] + A[2, 2] * b[2]


cdef inline void add(double[::1] a, double [::1] b, double x):
    a[0] += x * b[0]
    a[1] += x * b[1]
    a[2] += x * b[2]


cdef inline void scale(double[::1] a, double x):
    a[0] *= x
    a[1] *= x
    a[2] *= x


def solve_for_omega(double[::1] dt, double[::1] diag,
                    double[:, :, ::1] A, double[:, :, ::1] B,
                    double[:, ::1] rhs, double[:, ::1] omega):
    cdef:
        int n_segments = dt.shape[0]
        int i
        double mult
        double[::1] vec = np.empty(3)

    mat_vec(B[0], rhs[0], vec)
    add(rhs[1], vec, -2/dt[0])

    for i in range(1, n_segments - 1):
        mult = 2 / (diag[i] * dt[i])
        diag[i + 1] -= 2 * mult / dt[i]
        mat_vec(B[i], rhs[i], vec)
        add(rhs[i + 1], vec, -mult)

    for i in reversed(range(1, n_segments)):
        omega[i] = rhs[i].copy()
        mat_vec(A[i], omega[i + 1], vec)
        add(omega[i], vec, -2/dt[i])
        scale(omega[i], 1/diag[i])

"""Thin wrapper of CuPy Thrust wrapper."""
# NOTE: This wrapper does not cover all APIs of Thrust.


###############################################################################
# Extern
###############################################################################

cdef extern from "cupy_thrust.h" namespace "cupy::thrust":
    void stable_sort_short(short *first, short *last)
    void stable_sort_int(int *first, int *last)
    void stable_sort_long(long *first, long *last)
    void stable_sort_float(float *first, float *last)
    void stable_sort_double(double *first, double *last)

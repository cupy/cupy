"""Thin wrapper of CuPy Thrust wrapper."""
# NOTE: This wrapper does not cover all APIs of Thrust.


###############################################################################
# Extern
###############################################################################

cdef extern from "cupy_thrust.h" namespace "cupy::thrust":
     void stable_sort(float *first, float *last)

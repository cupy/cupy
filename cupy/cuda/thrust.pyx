"""Thin wrapper of CuPy Thrust wrapper."""
# NOTE: This wrapper does not cover all APIs of Thrust.

###############################################################################
# Extern
###############################################################################

cdef extern from "../cuda/cupy_thrust.h" namespace "cupy::thrust":
    void stable_sort[T](void *start, ssize_t num)

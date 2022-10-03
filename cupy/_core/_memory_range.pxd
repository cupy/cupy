from cupy._core.core cimport _ndarray_base

from libcpp.pair cimport pair


cpdef pair[Py_ssize_t, Py_ssize_t] get_bound(_ndarray_base array)
cpdef bint may_share_bounds(_ndarray_base a, _ndarray_base b)

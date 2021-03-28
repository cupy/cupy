from cupy._core.core cimport ndarray

from libcpp.pair cimport pair


cpdef pair[Py_ssize_t, Py_ssize_t] get_bound(ndarray array)
cpdef bint may_share_bounds(ndarray a, ndarray b)

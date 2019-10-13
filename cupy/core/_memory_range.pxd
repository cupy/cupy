from cupy.core.core cimport ndarray


cpdef bint may_share_bounds(ndarray a, ndarray b)

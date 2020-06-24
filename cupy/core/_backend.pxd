cdef list _reduction_backends

cdef list _routine_backends

cpdef enum backend_type:
    BACKEND_CUB = 1
    # BACKEND_CUTENSOR = 2

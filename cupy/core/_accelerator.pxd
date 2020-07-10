cdef list _reduction_accelerators

cdef list _routine_accelerators

cpdef enum accelerator_type:
    ACCELERATOR_CUB = 1
    ACCELERATOR_CUTENSOR = 2

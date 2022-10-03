cdef list _elementwise_accelerators

cdef list _reduction_accelerators

cdef list _routine_accelerators

cpdef enum accelerator_type:
    ACCELERATOR_CUB = 1
    ACCELERATOR_CUTENSOR = 2
    ACCELERATOR_CUTENSORNET = 3

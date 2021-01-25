from libc.stdint cimport intptr_t


cdef class Graph:
    cdef:
        intptr_t graph  # cudaGraph_t
        intptr_t graphExec  # cudaGraphExec_t

    cdef void _init(self, intptr_t g, intptr_t ge)

    @staticmethod
    cdef Graph from_stream(intptr_t g)

    cpdef launch(self)
    cpdef upload(self)

from libc.stdint cimport intptr_t


cdef class Graph:
    cdef:
        intptr_t graph  # cudaGraph_t
        intptr_t graphExec  # cudaGraphExec_t
        intptr_t stream

    cdef _init(self, intptr_t g, intptr_t ge, intptr_t s)

    @staticmethod
    cdef Graph from_stream(intptr_t s)

    cpdef launch(self)

from libc.stdint cimport intptr_t


cdef class Graph:
    cdef:
        readonly intptr_t graph  # cudaGraph_t
        readonly intptr_t graphExec  # cudaGraphExec_t

    cdef void _init(self, intptr_t g, intptr_t ge) except*

    @staticmethod
    cdef Graph from_stream(intptr_t g)

    cpdef launch(self, stream=*)
    cpdef upload(self, stream=*)

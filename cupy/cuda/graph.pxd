from libc.stdint cimport intptr_t


cdef class Graph:
    cdef:
        intptr_t graph  # cudaGraph_t
        intptr_t graphExec  # cudaGraphExec_t
        intptr_t stream_ptr
        readonly object stream

    cdef void _init(self, intptr_t g, intptr_t ge, s)

    @staticmethod
    cdef Graph from_stream(intptr_t g, stream)

    cpdef launch(self)
    cpdef upload(self)

from libc.stdint cimport intptr_t


cdef class Graph:
    cdef:
        readonly intptr_t graph  # cudaGraph_t
        readonly intptr_t graphExec  # cudaGraphExec_t
        readonly object host_funcargs

    cdef void _init(
        self, intptr_t g, intptr_t ge, object host_funcargs) except*

    @staticmethod
    cdef Graph from_stream(intptr_t g, object host_funcargs)

    cpdef launch(self, stream=*)
    cpdef upload(self, stream=*)
    cpdef debug_dot_str(self, flags=*)

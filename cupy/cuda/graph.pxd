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
    cpdef add_conditional_node(self, str node_type, GraphNodeDependencies deps)
    cpdef debug_dot_str(self, flags=*)

cdef class GraphNodeDependencies:
    cdef:
        readonly intptr_t dependency_nodes
        readonly size_t num_dependencies

    cdef void _init(self, intptr_t dependency_nodes, size_t num_dependencies)

    @staticmethod
    cdef GraphNodeDependencies _new(intptr_t dependency_nodes, size_t num_dependencies)

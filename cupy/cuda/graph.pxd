from libc.stdint cimport intptr_t


cdef class Graph:
    cdef:
        readonly intptr_t graph  # cudaGraph_t
        readonly intptr_t graphExec  # cudaGraphExec_t
        readonly bint _is_child
        readonly list _refs

    cdef void _init(self, intptr_t g, intptr_t ge, bint is_child=*) except*

    @staticmethod
    cdef Graph from_stream(intptr_t g, bint is_child=*)

    cpdef _ensure_instantiate(self)

    cpdef launch(self, stream=*)
    cpdef upload(self, stream=*)
    cpdef debug_dot_str(self, flags=*)
    cpdef _add_ref(self, ref)

cpdef int _create_conditional_handle_from_stream(
        stream,
        default_value=*,
        flags=*
    )

cpdef Graph _append_conditional_node_to_stream(
        stream, node_type, handle
    )

from libc.stdint cimport intptr_t


cdef class Graph:
    cdef:
        readonly intptr_t graph  # cudaGraph_t
        readonly intptr_t graphExec  # cudaGraphExec_t

        # If a graph is owned, the responsibility for its memory management
        # lies with the CuPy runtime;
        # otherwise, it falls under the responsibility of the CUDA runtime.
        #
        # A graph is not owned when the graph is a child graph or created by
        # cudaGraphCreate and is passed to cudaStreamBeginCaptureToGraph
        readonly bint _owned
        readonly list _refs

    cdef void _init(self, intptr_t g, intptr_t ge, bint owned=*) except*

    @staticmethod
    cdef Graph from_stream(intptr_t g, bint owned=*)

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

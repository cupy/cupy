import os
import tempfile

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.string cimport memset as c_memset

from cupy_backends.cuda.api cimport runtime
from cupy_backends.cuda cimport stream as stream_module

cdef extern from '../../cupy_backends/cupy_backend_runtime.h':
    pass

cdef class Graph:
    """The CUDA graph object.

    Currently this class cannot be initiated by the user and must be created
    via stream capture. See :meth:`~cupy.cuda.Stream.begin_capture` for detail.

    """

    cdef void _init(
        self, intptr_t graph, intptr_t graphExec, bint owned=True
    ) except*:
        self.graph = graph
        self.graphExec = graphExec
        self._owned = owned
        self._refs = list()

    def __dealloc__(self):
        if not self._owned:
            # Freeing unowned graph is not CuPy's responsibility
            return
        if self.graph > 0:
            runtime.graphDestroy(self.graph)
        if self.graphExec > 0:
            runtime.graphExecDestroy(self.graphExec)

    def __init__(self, bint owned=False):
        raw_graph = runtime.graphCreate()
        self._init(
            raw_graph,
            0,  # graphExec
            owned,
        )

    @staticmethod
    cdef Graph from_stream(intptr_t g, bint owned=True):
        # TODO(leofang): optionally print out the error log?
        cdef intptr_t ge = 0
        cdef Graph graph = Graph.__new__(Graph)
        graph._init(g, ge, owned)
        return graph

    cpdef _ensure_instantiate(self):
        if not self._owned:
            raise RuntimeError("Unowned graph instantiation is not allowed")
        if self.graphExec == 0:
            self.graphExec = runtime.graphInstantiate(self.graph)

    cpdef launch(self, stream=None):
        """Launch the CUDA graph on the given stream.

        Args:
            stream (:class:`~cupy.cuda.Stream`): A CuPy stream object. If not
                specified (using the default value `None`), the graph is
                launched on the current stream.

        .. seealso:: `cudaGraphLaunch()`_

        .. _cudaGraphLaunch():
            https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1g1accfe1da0c605a577c22d9751a09597

        """
        self._ensure_instantiate()
        cdef intptr_t stream_ptr
        if stream is None:
            stream_ptr = stream_module.get_current_stream_ptr()
        else:
            stream_ptr = stream.ptr
        runtime.graphLaunch(self.graphExec, stream_ptr)

    cpdef upload(self, stream=None):
        """Upload the CUDA graph to the given stream.

        Args:
            stream (:class:`~cupy.cuda.Stream`): A CuPy stream object. If not
                specified (using the default value `None`), the graph is
                uploaded the current stream.

        .. seealso:: `cudaGraphUpload()`_

        .. _cudaGraphUpload():
            https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1ge546432e411b4495b93bdcbf2fc0b2bd

        """
        self._ensure_instantiate()
        cdef intptr_t stream_ptr
        if stream is None:
            stream_ptr = stream_module.get_current_stream_ptr()
        else:
            stream_ptr = stream.ptr
        runtime.graphUpload(self.graphExec, stream_ptr)

    cpdef debug_dot_str(self, flags=0):
        """Make DOT formatted string of CUDA graph definition for debugging.

        Args:
            flags (:class:`unsigned int`): Flags to specify information to be
                included.

        .. seealso:: `cudaGraphDebugDotPrint()`_

        .. _cudaGraphDebugDotPrint():
            https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1gbec177c250000405c570dc8c4bde20db

        """
        f = tempfile.NamedTemporaryFile(delete=False)
        try:
            f.close()
            runtime.graphDebugDotPrint(self.graph, f.name, flags)
            with open(f.name) as f2:
                return f2.read()
        finally:
            os.remove(f.name)

    cpdef _add_ref(self, ref):
        self._refs.append(ref)

cpdef int _create_conditional_handle_from_stream(
    stream,
    default_value=False,
    flags=runtime.cudaGraphCondAssignDefault
):
    '''
    Returns conditional handle body value (int)
    '''
    graph, _ = stream._capturing_graph_info

    handle = runtime.graphConditionalHandleCreate(
        graph.graph,
        defaultLaunchValue=default_value,
        flags=flags
    )
    return handle

cpdef Graph _append_conditional_node_to_stream(
    stream, node_type, handle
):
    '''
    Returns conditional node's body graph
    '''
    status, _id, main_graph_ptr, deps_ptr, n_deps = \
        runtime.streamGetCaptureInfo(stream.ptr)
    if status != runtime.streamCaptureStatusActive:
        raise RuntimeError(
            "Conditional node can be added only to capturing stream")

    cdef runtime.GraphConditionalNodeType node_type_enum
    if node_type == "if":
        node_type_enum = <runtime.GraphConditionalNodeType>(
            runtime.cudaGraphCondTypeIf
        )
    elif node_type == "while":
        node_type_enum = <runtime.GraphConditionalNodeType>(
            runtime.cudaGraphCondTypeWhile
        )
    else:
        raise ValueError("`node_type` must be 'if' or 'while'")

    # Allocate node params struct's memory via `malloc` to avoid
    # the use of deleted constructor
    cdef runtime.GraphNodeParams* params = \
        <runtime.GraphNodeParams*>(
            PyMem_Malloc(sizeof(runtime.GraphNodeParams)))
    if not params:
        raise MemoryError()
    cdef runtime.Graph[1] body_graphs
    cdef runtime.GraphNode[1] nodes
    try:
        c_memset(params, 0, sizeof(runtime.GraphNodeParams))
        params.type = <runtime.GraphNodeType>(
            runtime.cudaGraphNodeTypeConditional)
        params.conditional.handle = <unsigned long long>(handle)
        params.conditional.type = node_type_enum
        params.conditional.size = <size_t>(1)

        nodes[0] = <runtime.GraphNode>(runtime.graphAddNode(
            main_graph_ptr,
            deps_ptr,
            n_deps,
            <intptr_t>(params)
        ))

        body_graphs[0] = params.conditional.phGraph_out[0]
    finally:
        PyMem_Free(params)

    runtime.streamUpdateCaptureDependencies(
        stream.ptr,
        <intptr_t>nodes,
        1,  # number of dependency nodes
        runtime.cudaStreamSetCaptureDependencies
    )

    return Graph.from_stream(<intptr_t>(body_graphs[0]), owned=False)

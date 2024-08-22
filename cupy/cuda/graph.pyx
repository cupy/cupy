import os
import tempfile

from cupy_backends.cuda.api cimport runtime
from cupy_backends.cuda cimport stream as stream_module
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.string cimport memset as c_memset


cdef extern from '../../cupy_backends/cupy_backend_runtime.h':
    pass

cdef class Graph:
    """The CUDA graph object.

    Currently this class cannot be initiated by the user and must be created
    via stream capture. See :meth:`~cupy.cuda.Stream.begin_capture` for detail.

    """

    cdef void _init(self, intptr_t graph, intptr_t graphExec) except*:
        self.graph = graph
        self.graphExec = graphExec

    def __dealloc__(self):
        if self.graph > 0:
            runtime.graphDestroy(self.graph)
        if self.graphExec > 0:
            runtime.graphExecDestroy(self.graphExec)

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            'currently this class cannot be initiated by the user and must '
            'be created via stream capture')

    @staticmethod
    cdef Graph from_stream(intptr_t g):
        # TODO(leofang): optionally print out the error log?
        cdef intptr_t ge = runtime.graphInstantiate(g)
        cdef Graph graph = Graph.__new__(Graph)
        graph._init(g, ge)
        return graph

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
        cdef intptr_t stream_ptr
        if stream is None:
            stream_ptr = stream_module.get_current_stream_ptr()
        else:
            stream_ptr = stream.ptr
        runtime.graphUpload(self.graphExec, stream_ptr)

    cpdef add_conditional_node(self, str node_type, GraphNodeDependencies deps):
        """
        Return:
            handle: cudaGrapnConditionalHandle
            body_graph: Graph
            dependency: GraphNodeDependencies
        """
        # Create handle
        cdef unsigned long long handle
        handle = runtime.graphConditionalHandleCreate(
            self.graph,
            1, # defaultLaunchValue
            runtime.cudaGraphCondAssignDefault
        )

        # Allocate node params memory via malloc to avoid `deleted function` error
        cdef runtime.GraphNodeParams* cparams = \
            <runtime.GraphNodeParams*>(PyMem_Malloc(sizeof(runtime.GraphNodeParams)))
        if not cparams:
            raise MemoryError()
        c_memset(cparams, 0, sizeof(runtime.GraphNodeParams))
        try:
            cparams.conditional.size = 1
            if node_type == "while":
                cparams.conditional.type = <runtime.GraphConditionalNodeType>(runtime.cudaGraphCondTypeWhile)
            elif node_type == "if":
                cparams.conditional.type = <runtime.GraphConditionalNodeType>(runtime.cudaGraphCondTypeIf)
            else:
                raise ValueError("`node_type` must be 'if' or 'while'")
            cparams.conditional.handle = handle

            # Add node to graph
            conditional_node = runtime.graphAddNode(
                self.graph, deps.dependency_nodes, deps.num_dependencies, <intptr_t>(cparams))
            body_graph = \
                Graph.from_stream(<intptr_t>(cparams.conditional.phGraph_out[0]))
            return (
                handle,
                body_graph,
                GraphNodeDependencies._new(conditional_node, 1)
            )
        finally:
            PyMem_Free(cparams)


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

cdef class GraphNodeDependencies:
    cdef void _init(self, intptr_t dependency_nodes, size_t num_dependencies):
        self.dependency_nodes = dependency_nodes
        self.num_dependencies = num_dependencies

    @staticmethod
    cdef GraphNodeDependencies _new(intptr_t dependency_nodes, size_t num_dependencies):
        cdef GraphNodeDependencies deps = GraphNodeDependencies.__new__(GraphNodeDependencies)
        deps._init(dependency_nodes, num_dependencies)

        return deps

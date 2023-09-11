from cupy_backends.cuda.api cimport runtime
from cupy_backends.cuda cimport stream as stream_module


cdef class Graph:
    """The CUDA graph object.

    Currently this class cannot be initiated by the user and must be created
    via stream capture. See :meth:`~cupy.cuda.Stream.begin_capture` for detail.

    """

    cdef void _init(self, intptr_t graph, intptr_t graphExec) except*:
        if graph > 0:
            # at this point cudaGraphExec_t has been instantiated, so we no
            # longer need to hold the cudaGraph_t
            runtime.graphDestroy(graph)
        self.graph = 0
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

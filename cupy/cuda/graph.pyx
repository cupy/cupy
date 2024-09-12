import os
import tempfile

from cupy_backends.cuda.api cimport runtime
from cupy_backends.cuda cimport stream as stream_module


cdef class Graph:
    """The CUDA graph object.

    Currently this class cannot be initiated by the user and must be created
    via stream capture. See :meth:`~cupy.cuda.Stream.begin_capture` for detail.

    """

    cdef void _init(self, intptr_t graph, intptr_t graphExec, bint is_child=False) except*:
        self.graph = graph
        self.graphExec = graphExec
        self.is_child = is_child
        self._refs = list()

    def __dealloc__(self):
        if self.is_child:
            # Do not call graphDestroy for child graph because
            # graphDestroy function destroys graph recursively
            return
        if self.graph > 0:
            runtime.graphDestroy(self.graph)
        if self.graphExec > 0:
            runtime.graphExecDestroy(self.graphExec)

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            'currently this class cannot be initiated by the user and must '
            'be created via stream capture')

    @staticmethod
    cdef Graph from_stream(intptr_t g, bint is_child=False):
        # TODO(leofang): optionally print out the error log?
        cdef intptr_t ge = 0
        cdef Graph graph = Graph.__new__(Graph)
        graph._init(g, ge, is_child)
        return graph

    cpdef _ensure_instantiate(self):
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
        if self.is_child:
            raise RuntimeError("Cannnot launch child graph")
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
        if self.is_child:
            raise RuntimeError("Cannnot upload child graph")
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

    cpdef add_ref(self, ref):
        self._refs.append(ref)

from cupy_backends.cuda.api cimport runtime
from cupy_backends.cuda cimport stream as stream_module


cdef class Graph:

    cdef void _init(self, intptr_t graph, intptr_t graphExec):
        self.graph = graph
        self.graphExec = graphExec

    def __dealloc__(self):
        if self.graph > 0:
            runtime.graphDestroy(self.graph)
        if self.graphExec > 0:
            runtime.graphExecDestroy(self.graphExec)

    @staticmethod
    cdef Graph from_stream(intptr_t g):
        # TODO(leofang): optionally print out the error log?
        cdef intptr_t ge = runtime.graphInstantiate(g)
        cdef Graph graph = Graph.__new__(Graph)
        graph._init(g, ge)
        return graph

    cpdef launch(self, stream=None):
        cdef intptr_t stream_ptr
        if stream is None:
            stream_ptr = stream_module.get_current_stream_ptr()
        else:
            stream_ptr = stream.ptr
        runtime.graphLaunch(self.graphExec, stream_ptr)

    cpdef upload(self, stream=None):
        cdef intptr_t stream_ptr
        if stream is None:
            stream_ptr = stream_module.get_current_stream_ptr()
        else:
            stream_ptr = stream.ptr
        runtime.graphUpload(self.graphExec, stream_ptr)

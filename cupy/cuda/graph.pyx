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

    cpdef launch(self):
        cdef intptr_t stream = stream_module.get_current_stream_ptr()
        runtime.graphLaunch(self.graphExec, stream)

    cpdef upload(self):
        # TODO(leofang): I actually don't understand the purpose of this API
        # and did not find a meaningful way to test it, so let's disable it.
        raise NotImplementedError('this function is currently disabled')
        # runtime.graphUpload(self.graphExec, self.stream_ptr)

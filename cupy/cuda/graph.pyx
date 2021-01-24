from cupy_backends.cuda.api cimport runtime


cdef class Graph:

    cdef void _init(self, intptr_t g, intptr_t ge, stream):
        self.graph = g
        self.graphExec = ge
        self.stream_ptr = <intptr_t>(stream.ptr)
        self.stream = stream  # hold the reference to keep it alive

    def __dealloc__(self):
        if self.graph > 0:
            runtime.graphDestroy(self.graph)
        if self.graphExec > 0:
            runtime.graphExecDestroy(self.graphExec)

    @staticmethod
    cdef Graph from_stream(intptr_t g, stream):
        # TODO(leofang): optionally print out the error log?
        cdef intptr_t ge = runtime.graphInstantiate(g)
        cdef Graph graph = Graph.__new__(Graph)
        graph._init(g, ge, stream)
        return graph

    cpdef launch(self):
        # TODO(leofang): can we take a different stream here?
        runtime.graphLaunch(self.graphExec, self.stream_ptr)

    cpdef upload(self):
        # TODO(leofang): I actually don't understand the purpose of this API
        # and did not find a meaningful way to test it, so let's disable it.
        raise NotImplementedError('this function is currently disabled')
        # TODO(leofang): can we take a different stream here?
        # runtime.graphUpload(self.graphExec, self.stream_ptr)

from cupy_backends.cuda.api cimport runtime


cdef class Graph:

    cdef _init(self, intptr_t g, intptr_t ge, intptr_t s):
        self.graph = g
        self.graphExec = ge
        self.stream = s

    def __dealloc__(self):
        if self.graph > 0:
            runtime.graphDestroy(self.graph)
        if self.graphExec > 0:
            runtime.graphExecDestroy(self.graphExec)

    @staticmethod
    cdef Graph from_stream(intptr_t s):
        # TODO(leofang): hold a weak ref of s?
        cdef intptr_t g = runtime.streamEndCapture(s)
        cdef intptr_t ge = runtime.graphInstantiate(g)
        cdef Graph graph = Graph.__new__(Graph)
        graph._init(g, ge, s)
        return graph

    cpdef launch(self):
        # TODO(leofang): can we take a different stream here?
        # TODO(leofang): optionally print out the error log?
        runtime.graphLaunch(self.graphExec, self.stream)

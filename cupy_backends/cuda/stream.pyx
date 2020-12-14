import threading


cdef object _thread_local = threading.local()


cdef class _ThreadLocal:
    cdef intptr_t current_stream

    @staticmethod
    cdef _ThreadLocal get():
        try:
            tls = _thread_local.tls
        except AttributeError:
            tls = _thread_local.tls = _ThreadLocal()
        return <_ThreadLocal>tls

    cdef set_current_stream_ptr(self, intptr_t ptr):
        self.current_stream = ptr

    cdef intptr_t get_current_stream_ptr(self):
        # Returns nullptr if not set, which is equivalent to the default
        # stream.
        return self.current_stream


cdef intptr_t get_current_stream_ptr():
    """C API to get current CUDA stream pointer.

    Returns:
        intptr_t: The current CUDA stream pointer.
    """
    tls = _ThreadLocal.get()
    return <intptr_t>tls.get_current_stream_ptr()


cdef set_current_stream_ptr(intptr_t ptr):
    """C API to set current CUDA stream pointer.

    Args:
        ptr (intptr_t): CUDA stream pointer.
    """
    global enable_current_stream
    enable_current_stream = True
    tls = _ThreadLocal.get()
    tls.set_current_stream_ptr(ptr)

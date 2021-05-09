import os
import threading

from cupy_backends.cuda.api cimport runtime


cdef object _thread_local = threading.local()

cdef bint _ptds = bool(int(
    os.environ.get('CUPY_CUDA_PER_THREAD_DEFAULT_STREAM', '0')) != 0)


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
        # Returns the stream previously set, otherwise returns
        # nullptr or runtime.streamPerThread when
        # CUPY_CUDA_PER_THREAD_DEFAULT_STREAM=1.
        if self.current_stream == 0 and is_ptds_enabled():
            return runtime.streamPerThread

        return self.current_stream

    cdef intptr_t get_default_stream_ptr(self):
        if is_ptds_enabled():
            return runtime.streamPerThread
        else:  # we don't return 0 here
            return runtime.streamLegacy


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

    .. warning::

        This method is intended to be called from `cupy.cuda.stream` module.
        Do not call this method from somewhere else; this method only changes
        the default stream for `cupy_backends.*`, so the stream used will be
        inconsistent with the default one for `cupy.*`.

    """
    tls = _ThreadLocal.get()
    tls.set_current_stream_ptr(ptr)


# cpdef for unit testing
cpdef intptr_t get_default_stream_ptr():
    """Get the CUDA default stream pointer.

    Args:
        ptr (intptr_t): CUDA stream pointer.
    """
    tls = _ThreadLocal.get()
    return <intptr_t>tls.get_default_stream_ptr()


cdef bint is_ptds_enabled():
    if runtime._is_hip_environment:
        # HIP does not support PTDS, just ignore the env var
        return False
    return _ptds

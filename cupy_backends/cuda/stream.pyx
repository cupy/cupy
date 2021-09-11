import os as _os
import threading as _threading

from cupy_backends.cuda.api cimport runtime


cdef object _thread_local = _threading.local()


cdef bint _ptds = bool(int(
    _os.environ.get('CUPY_CUDA_PER_THREAD_DEFAULT_STREAM', '0')) != 0)


cdef class _ThreadLocal:
    cdef list current_stream  # list of intptr_t

    def __init__(self):
        cdef int i, num_devices = runtime.getDeviceCount()
        self.current_stream = [0 for i in range(num_devices)]

    @staticmethod
    cdef _ThreadLocal get():
        try:
            tls = _thread_local.tls
        except AttributeError:
            tls = _thread_local.tls = _ThreadLocal()
        return <_ThreadLocal>tls

    cdef set_current_stream_ptr(self, intptr_t ptr, int device_id=-1):
        if device_id == -1:
            device_id = runtime.getDevice()
        self.current_stream[device_id] = ptr

    cdef intptr_t get_current_stream_ptr(self, int device_id=-1):
        # Returns the stream previously set, otherwise returns
        # nullptr or runtime.streamPerThread when
        # CUPY_CUDA_PER_THREAD_DEFAULT_STREAM=1.
        if device_id == -1:
            device_id = runtime.getDevice()
        cdef intptr_t curr_stream = self.current_stream[device_id]
        if curr_stream == 0 and is_ptds_enabled():
            return runtime.streamPerThread
        return curr_stream


cdef intptr_t get_current_stream_ptr():
    """C API to get current CUDA stream pointer.

    Returns:
        intptr_t: The current CUDA stream pointer.
    """
    tls = _ThreadLocal.get()
    return <intptr_t>tls.get_current_stream_ptr()


cdef set_current_stream_ptr(intptr_t ptr, int device_id=-1):
    """C API to set current CUDA stream pointer.

    Args:
        ptr (intptr_t): CUDA stream pointer.
        device_id (int): device ID. Look up the current device if -1.

    .. warning::

        This method is intended to be called from `cupy.cuda.stream` module.
        Do not call this method from somewhere else; this method only changes
        the default stream for `cupy_backends.*`, so the stream used will be
        inconsistent with the default one for `cupy.*`.

    """
    tls = _ThreadLocal.get()
    tls.set_current_stream_ptr(ptr, device_id)


# cpdef for unit testing
cpdef intptr_t get_default_stream_ptr():
    """Get the CUDA default stream pointer.

    Returns:
        intptr_t: CUDA stream pointer.
    """
    if is_ptds_enabled():
        return runtime.streamPerThread
    else:  # we don't return 0 here
        return runtime.streamLegacy


cdef bint is_ptds_enabled():
    if runtime._is_hip_environment:
        # HIP does not support PTDS, just ignore the env var
        return False
    return _ptds

from cupy_backends.cuda.api cimport runtime
from cupy_backends.cuda cimport stream as stream_module

import os
import threading
import weakref

from cupy import _util


cdef object _thread_local = threading.local()


cdef class _ThreadLocal:
    cdef list current_stream  # list of intptr_t
    cdef list current_stream_ref  # list of object
    cdef list prev_stream_ref_stack  # list of list

    def __init__(self):
        cdef int i, num_devices = runtime.getDeviceCount()
        self.current_stream = [0 for i in range(num_devices)]
        self.current_stream_ref = [None for i in range(num_devices)]
        self.prev_stream_ref_stack = [None for i in range(num_devices)]

    @staticmethod
    cdef _ThreadLocal get():
        try:
            tls = _thread_local.tls
        except AttributeError:
            tls = _thread_local.tls = _ThreadLocal()
        return <_ThreadLocal>tls

    cdef set_current_stream(self, stream):
        cdef intptr_t ptr = <intptr_t>stream.ptr
        cdef int dev = stream.dev
        if dev == -1:
            dev = runtime.getDevice()
        stream_module.set_current_stream_ptr(ptr, dev)
        self.current_stream[dev] = <intptr_t>ptr
        self.current_stream_ref[dev] = weakref.ref(stream)

    cdef set_current_stream_ref(self, stream_ref):
        cdef intptr_t ptr = <intptr_t>stream_ref().ptr
        cdef int dev = stream_ref().dev
        if dev == -1:
            dev = runtime.getDevice()
        stream_module.set_current_stream_ptr(ptr, dev)
        self.current_stream[dev] = <intptr_t>ptr
        self.current_stream_ref[dev] = stream_ref

    cdef get_current_stream(self, int dev=-1):
        if dev == -1:
            dev = runtime.getDevice()
        if self.current_stream_ref[dev] is None:
            if stream_module.is_ptds_enabled():
                self.current_stream_ref[dev] = weakref.ref(Stream.ptds)
            else:
                self.current_stream_ref[dev] = weakref.ref(Stream.null)
        return self.current_stream_ref[dev]()

    cdef get_current_stream_ref(self, int dev=-1):
        if dev == -1:
            dev = runtime.getDevice()
        if self.current_stream_ref[dev] is None:
            if stream_module.is_ptds_enabled():
                self.current_stream_ref[dev] = weakref.ref(Stream.ptds)
            else:
                self.current_stream_ref[dev] = weakref.ref(Stream.null)
        return self.current_stream_ref[dev]

    cdef intptr_t get_current_stream_ptr(self, int dev=-1):
        # Returns the stream previously set, otherwise returns
        # nullptr or runtime.streamPerThread when
        # CUPY_CUDA_PER_THREAD_DEFAULT_STREAM=1.
        if dev == -1:
            dev = runtime.getDevice()
        cdef intptr_t curr_stream = self.current_stream[dev]
        if stream_module.is_ptds_enabled() and curr_stream == 0:
            return <intptr_t>runtime.streamPerThread
        return curr_stream


cdef intptr_t get_current_stream_ptr():
    """C API to get current CUDA stream pointer.

    Returns:
        intptr_t: The current CUDA stream pointer.
    """
    tls = _ThreadLocal.get()
    return <intptr_t>tls.get_current_stream_ptr()


cpdef get_current_stream():
    """Gets current CUDA stream.

    Returns:
        cupy.cuda.Stream: The current CUDA stream.
    """
    tls = _ThreadLocal.get()
    return tls.get_current_stream()


class Event(object):

    """CUDA event, a synchronization point of CUDA streams.

    This class handles the CUDA event handle in RAII way, i.e., when an Event
    instance is destroyed by the GC, its handle is also destroyed.

    Args:
        block (bool): If ``True``, the event blocks on the
            :meth:`~cupy.cuda.Event.synchronize` method.
        disable_timing (bool): If ``True``, the event does not prepare the
            timing data.
        interprocess (bool): If ``True``, the event can be passed to other
            processes.

    Attributes:
        ~Event.ptr (intptr_t): Raw event handle.

    """

    def __init__(self, block=False, disable_timing=False, interprocess=False):
        self.ptr = 0

        if interprocess and not disable_timing:
            raise ValueError('Timing must be disabled for interprocess events')
        flag = ((block and runtime.eventBlockingSync) |
                (disable_timing and runtime.eventDisableTiming) |
                (interprocess and runtime.eventInterprocess))
        self.ptr = runtime.eventCreateWithFlags(flag)

    def __del__(self, is_shutting_down=_util.is_shutting_down):
        if is_shutting_down():
            return
        if self.ptr:
            runtime.eventDestroy(self.ptr)

    @property
    def done(self):
        """True if the event is done."""
        return runtime.eventQuery(self.ptr) == 0  # cudaSuccess

    def record(self, stream=None):
        """Records the event to a stream.

        Args:
            stream (cupy.cuda.Stream): CUDA stream to record event. The null
                stream is used by default.

        .. seealso:: :meth:`cupy.cuda.Stream.record`

        """
        if stream is None:
            stream_ptr = get_current_stream_ptr()
        else:
            stream_ptr = stream.ptr
        runtime.eventRecord(self.ptr, stream_ptr)

    def synchronize(self):
        """Synchronizes all device work to the event.

        If the event is created as a blocking event, it also blocks the CPU
        thread until the event is done.

        """
        runtime.eventSynchronize(self.ptr)


def get_elapsed_time(start_event, end_event):
    """Gets the elapsed time between two events.

    Args:
        start_event (Event): Earlier event.
        end_event (Event): Later event.

    Returns:
        float: Elapsed time in milliseconds.

    """
    return runtime.eventElapsedTime(start_event.ptr, end_event.ptr)


cdef void check_stream_device_match(int dev) except*:
    cdef int curr_dev = runtime.getDevice()
    if dev == -1:
        dev = curr_dev
    if dev != curr_dev:
        raise RuntimeError(
            f'This stream was not created on device {curr_dev}')


class BaseStream(object):

    """CUDA stream.

    Attributes:
        ~Stream.ptr (intptr_t): Raw stream handle.
        ~Stream.dev (int): The ID of the device that the stream was created on.

    """

    null = None

    def __eq__(self, other):
        # This operator is implemented to compare the singleton instance
        # of null stream (Stream.null) can safely be compared with null
        # stream instance created by a user.
        return self.ptr == other.ptr

    def __enter__(self):
        tls = _ThreadLocal.get()
        cdef int dev = self.dev
        check_stream_device_match(dev)
        # to prevent from popping the wrong stream at exit
        self._curr_dev = dev
        if tls.prev_stream_ref_stack[dev] is None:
            tls.prev_stream_ref_stack[dev] = []
        prev_stream_ref = tls.get_current_stream_ref(dev)
        tls.prev_stream_ref_stack[dev].append(prev_stream_ref)
        tls.set_current_stream(self)
        return self

    def __exit__(self, *args):
        tls = _ThreadLocal.get()
        dev = self._curr_dev
        self._curr_dev = -1
        prev_stream_ref = tls.prev_stream_ref_stack[dev].pop()
        tls.set_current_stream_ref(prev_stream_ref)

    def __repr__(self):
        return '<{} {}>'.format(type(self).__name__, self.ptr)

    def use(self):
        """Makes this stream current.

        If you want to switch a stream temporarily, use the *with* statement.
        """
        tls = _ThreadLocal.get()
        check_stream_device_match(self.dev)
        tls.set_current_stream(self)
        return self

    @property
    def done(self):
        """True if all work on this stream has been done."""
        return runtime.streamQuery(self.ptr) == 0  # cudaSuccess

    def synchronize(self):
        """Waits for the stream completing all queued work."""
        runtime.streamSynchronize(self.ptr)

    def add_callback(self, callback, arg):
        """Adds a callback that is called when all queued work is done.

        Args:
            callback (function): Callback function. It must take three
                arguments (Stream object, int error status, and user data
                object), and returns nothing.
            arg (object): Argument to the callback.

        .. note::
            Whenever possible, use the :meth:`launch_host_func` method
            instead of this one, as it may be deprecated and removed from
            CUDA at some point.

        """
        def f(stream, status, dummy):
            callback(self, status, arg)

        runtime.streamAddCallback(self.ptr, f, 0)

    def launch_host_func(self, callback, arg):
        """Launch a callback on host when all queued work is done.

        Args:
            callback (function): Callback function. It must take only one
                argument (user data object), and returns nothing.
            arg (object): Argument to the callback.

        .. note::
            Whenever possible, this method is recommended over
            :meth:`add_callback`, which may be deprecated and removed from
            CUDA at some point.

        .. seealso:: `cudaLaunchHostFunc()`_

        .. _cudaLaunchHostFunc():
            https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g05841eaa5f90f27124241baafb3e856f

        """
        def f(dummy):
            callback(arg)

        runtime.launchHostFunc(self.ptr, f, 0)

    def record(self, event=None):
        """Records an event on the stream.

        Args:
            event (None or cupy.cuda.Event): CUDA event. If ``None``, then a
                new plain event is created and used.

        Returns:
            cupy.cuda.Event: The recorded event.

        .. seealso:: :meth:`cupy.cuda.Event.record`

        """
        if event is None:
            event = Event()
        runtime.eventRecord(event.ptr, self.ptr)
        return event

    def wait_event(self, event):
        """Makes the stream wait for an event.

        The future work on this stream will be done after the event.

        Args:
            event (cupy.cuda.Event): CUDA event.

        """
        runtime.streamWaitEvent(self.ptr, event.ptr)


class Stream(BaseStream):

    """CUDA stream.

    This class handles the CUDA stream handle in RAII way, i.e., when an Stream
    instance is destroyed by the GC, its handle is also destroyed.

    Note that if both ``null`` and ``ptds`` are ``False``, a plain new
    stream is created.

    Args:
        null (bool): If ``True``, the stream is a null stream (i.e. the default
            stream that synchronizes with all streams). Note that you can also
            use the ``Stream.null`` singleton object instead of creating a new
            null stream object.
        ptds (bool): If ``True`` and ``null`` is ``False``, the per-thread
            default stream is used. Note that you can also use the
            ``Stream.ptds`` singleton object instead of creating a new
            per-thread default stream object.
        non_blocking (bool): If ``True`` and both ``null`` and ``ptds`` are
            ``False``, the stream does not synchronize with the NULL stream.

    Attributes:
        ~Stream.ptr (intptr_t): Raw stream handle.
        ~Stream.dev (int): The ID of the device that the stream was created on.
            The value ``-1`` is used for the singleton stream objects.

    """

    def __init__(self, null=False, non_blocking=False, ptds=False):
        if null:
            # TODO(pentschev): move to streamLegacy. This wasn't possible
            # because of a NCCL bug that should be fixed in the version
            # following 2.8.3-1.
            self.ptr = 0
            self.dev = -1
        elif ptds:
            if runtime._is_hip_environment:
                raise ValueError('HIP does not support per-thread '
                                 'default stream (ptds)')
            self.ptr = runtime.streamPerThread
            self.dev = -1
        elif non_blocking:
            self.ptr = runtime.streamCreateWithFlags(
                runtime.streamNonBlocking)
            self.dev = runtime.getDevice()
        else:
            self.ptr = runtime.streamCreate()
            self.dev = runtime.getDevice()

    def __del__(self, is_shutting_down=_util.is_shutting_down):
        cdef intptr_t current_ptr
        if is_shutting_down():
            return
        tls = _ThreadLocal.get()
        if self.ptr:
            current_ptr = <intptr_t>tls.get_current_stream_ptr()
            if <intptr_t>self.ptr == current_ptr:
                tls.set_current_stream(self.null)
            runtime.streamDestroy(self.ptr)
        else:
            current_stream = tls.get_current_stream()
            if current_stream == self:
                tls.set_current_stream(self.null)
        # Note that we can not release memory pool of the stream held in CPU
        # because the memory would still be used in kernels executed in GPU.


class ExternalStream(BaseStream):

    """CUDA stream.

    This class allows to use external streams in CuPy by providing the
    stream pointer obtained from the CUDA runtime call.
    The user is in charge of managing the life-cycle of the stream.

    Args:
        ptr (intptr_t): Address of the `cudaStream_t` object.
        dev (int): The ID of the device that the stream was created on. Default
            is ``-1``, indicating it is unknown.

    Attributes:
        ~Stream.ptr (intptr_t): Raw stream handle.
        ~Stream.dev (int): The ID of the device that the stream was created on.
            The value ``-1`` is used to indicate it is unknown.

    .. warning::
        If ``dev`` is not specified, the user is required to ensure legal
        operations of the stream. Specifically, the stream must be used on the
        device that it was created on.

    """

    def __init__(self, ptr, dev=-1):
        self.ptr = ptr
        # It is in theory unsafe to just call runtime.getDevice() here, as the
        # stream pointer could come from a different device (although
        # unlikely). While we could use driver API combos cuStreamGetCtx ->
        # cuCtxSetCurrent -> cuCtxGetDevice -> ... to retrieve the device ID
        # associated with the stream, it is way too complicated. Let us keep
        # this as thin as possible.
        self.dev = dev


Stream.null = Stream(null=True)
if not runtime._is_hip_environment:
    Stream.ptds = Stream(ptds=True)

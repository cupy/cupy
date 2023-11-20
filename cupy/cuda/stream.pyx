import threading

from cupy_backends.cuda.api cimport runtime
from cupy_backends.cuda cimport stream as backends_stream
from cupy.cuda cimport graph

from cupy import _util


cdef object _thread_local = threading.local()


cdef class _ThreadLocal:
    # We keep both current_stream and current_stream_stack because the
    # former is also used when calling stream.use(). This bookkeeping enables
    # correct rewinding when "with" blocks are mixed with ".use()" (though
    # this is considered an anti-pattern).

    cdef list current_stream  # list of object
    cdef list current_device_id_stack  # list of int
    cdef list current_stream_stack  # list of list

    def __init__(self):
        cdef int i, num_devices = runtime.getDeviceCount()
        self.current_stream = []
        self.current_device_id_stack = []
        self.current_stream_stack = []
        for i in range(num_devices):
            default_stream = get_default_stream()
            self.current_stream.append(default_stream)
            self.current_stream_stack.append([default_stream])

    @staticmethod
    cdef _ThreadLocal get():
        try:
            tls = _thread_local.tls
        except AttributeError:
            tls = _thread_local.tls = _ThreadLocal()
        return <_ThreadLocal>tls

    cdef void push_stream(self, stream, int device_id) except*:
        assert device_id >= 0
        self.current_stream_stack[device_id].append(stream)
        # record device_id to prevent from popping the wrong stream at exit
        self.current_device_id_stack.append(device_id)
        self.set_current_stream(stream)

    cdef void pop_stream(self) except*:
        cdef int device_id = self.current_device_id_stack.pop()
        self.current_stream_stack[device_id].pop()
        prev_stream = self.current_stream_stack[device_id][-1]
        self.set_current_stream(prev_stream)
        assert len(self.current_stream_stack[device_id]) >= 1

    cdef set_current_stream(self, stream):
        cdef intptr_t ptr = <intptr_t>stream.ptr
        cdef int device_id = stream.device_id
        if device_id == -1:
            device_id = runtime.getDevice()
        backends_stream.set_current_stream_ptr(ptr, device_id)
        self.current_stream[device_id] = stream

    cdef get_current_stream(self, int device_id=-1):
        if device_id == -1:
            device_id = runtime.getDevice()
        stream_ref = self.current_stream[device_id]
        return stream_ref

    cdef intptr_t get_current_stream_ptr(self):
        return backends_stream.get_current_stream_ptr()


cdef get_default_stream():
    return Stream.ptds if backends_stream.is_ptds_enabled() else Stream.null


cdef intptr_t get_current_stream_ptr():
    """C API to get current CUDA stream pointer.

    Returns:
        intptr_t: The current CUDA stream pointer.
    """
    tls = _ThreadLocal.get()
    return <intptr_t>tls.get_current_stream_ptr()


cpdef get_current_stream(int device_id=-1):
    """Gets the current CUDA stream for the specified CUDA device.

    Args:
        device_id (int, optional): Index of the device to check for the current
            stream. The currently active device is selected by default.
    Returns:
        cupy.cuda.Stream: The current CUDA stream.
    """
    tls = _ThreadLocal.get()
    return tls.get_current_stream(device_id)


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


cdef int check_stream_device_match(int device_id) except? -1:
    """Check if the stream was created on the current device."""
    cdef int curr_dev = runtime.getDevice()
    if device_id == -1:
        device_id = curr_dev
    if device_id != curr_dev:
        raise RuntimeError(
            f'The device that the stream is created on ({device_id})'
            f' does not match with the current device ({curr_dev})')
    return device_id


class _BaseStream:

    """CUDA stream.

    Attributes:
        ~Stream.ptr (intptr_t): Raw stream handle.
        ~Stream.device_id (int): The ID of the device that the stream was
            created on.

    """

    def __init__(self, ptr, device_id):
        self.ptr = ptr
        self.device_id = device_id

    def __eq__(self, other):
        # This operator needed as the ptr may be shared between multiple Stream
        # instances (e.g, `Stream.null` singleton and `Stream(null=True)` or
        # `ExternalStream`s).
        if isinstance(other, _BaseStream):
            return self.ptr == other.ptr
        return False

    def __enter__(self):
        # N.B. for maintainers: do not use this context manager in CuPy
        # codebase. See #5943 and #5963.
        tls = _ThreadLocal.get()
        cdef int device_id = self.device_id
        device_id = check_stream_device_match(device_id)
        tls.push_stream(self, device_id)
        return self

    def __exit__(self, *args):
        tls = _ThreadLocal.get()
        tls.pop_stream()

    def __repr__(self):
        return '<{} {} (device {})>'.format(
            type(self).__name__, self.ptr, self.device_id)

    def __hash__(self):
        return self.ptr

    def use(self):
        """Makes this stream current.

        If you want to switch a stream temporarily, use the *with* statement.
        """
        tls = _ThreadLocal.get()
        cdef int device_id = self.device_id
        check_stream_device_match(device_id)
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

    def begin_capture(self, mode=None):
        """Begin stream capture to construct a CUDA graph.

        A call to this function must be paired with a call to
        :meth:`end_capture` to complete the capture.

        .. code-block:: python

            # create a non-blocking stream for the purpose of capturing
            s1 = cp.cuda.Stream(non_blocking=True)
            with s1:
                s1.begin_capture()
                # ... perform operations to construct a graph ...
                g = s1.end_capture()

            # the returned graph can be launched on any stream (including s1)
            g.launch(stream=s1)
            s1.synchronize()

            s2 = cp.cuda.Stream()
            with s2:
                g.launch()
            s2.synchronize()

        Args:
            mode (int): The stream capture mode. Default is
                :data:`~cupy.cuda.runtime.streamCaptureModeRelaxed`.

        .. note:: During the stream capture, synchronous device-host transfers
            are not allowed. This has a particular implication for CuPy APIs,
            as some functions that internally require synchronous transfer
            would not work as expected and an exception would be raised. For
            further constraints of CUDA stream capture, please refer to the
            CUDA Programming Guide.

        .. note:: Currently this capability is not supported on HIP.

        .. seealso:: `cudaStreamBeginCapture()`_

        .. _cudaStreamBeginCapture():
            https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g793d7d4e474388ddfda531603dc34aa3

        """
        if runtime._is_hip_environment:
            raise RuntimeError('This function is not supported on HIP')
        if self.ptr == 0 or self.ptr == 1:
            raise RuntimeError('cannot capture on the default (legacy) stream')
        if mode is None:
            # We default to the relaxed mode for the following reason: During
            # the capture the memory pool might need to increase size. If it's
            # backed by cudaMalloc, which is an "unsafe" API, other modes would
            # not permit such a (legitimate) operation. See the API doc for
            # "cudaThreadExchangeStreamCaptureMode" for further detail.
            # (Though, ideally stream capture should be used together with
            # the async APIs, such as cudaMallocAsync.)
            mode = runtime.streamCaptureModeRelaxed
        runtime.streamBeginCapture(self.ptr, mode)

    def end_capture(self):
        """End stream capture and retrieve the constructed CUDA graph.

        Returns:
            cupy.cuda.Graph:
                A CUDA graph object that encapsulates the captured work.

        .. note:: Currently this capability is not supported on HIP.

        .. seealso:: `cudaStreamEndCapture()`_

        .. _cudaStreamEndCapture():
            https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gf5a0efebc818054ceecd1e3e5e76d93e

        """
        if runtime._is_hip_environment:
            raise RuntimeError('This function is not supported on HIP')
        cdef intptr_t g = runtime.streamEndCapture(self.ptr)
        return graph.Graph.from_stream(g)

    def is_capturing(self):
        """Check if the stream is capturing.

        Returns:
            bool:
                If the capturing status is successfully queried, the returned
                value indicates the capturing status. An exception could be
                raised if such a query is illegal, please refer to the CUDA
                Programming Guide for detail.

        """
        # TODO(leofang): is it better to be a property?
        if runtime._is_hip_environment:
            raise RuntimeError('This function is not supported on HIP')
        try:
            return runtime.streamIsCapturing(self.ptr)
        except RuntimeError:  # can be RuntimeError or CUDARuntimeError
            raise


class Stream(_BaseStream):

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
        ~Stream.device_id (int): The ID of the device that the stream was
            created on. The value ``-1`` is used for the singleton stream
            objects.

    """

    null = None
    ptds = None

    def __init__(self, null=False, non_blocking=False, ptds=False):
        if null:
            # TODO(pentschev): move to streamLegacy. This wasn't possible
            # because of a NCCL bug that should be fixed in the version
            # following 2.8.3-1.
            ptr = 0
            device_id = -1
        elif ptds:
            if runtime._is_hip_environment:
                raise ValueError('HIP does not support per-thread '
                                 'default stream (ptds)')
            ptr = runtime.streamPerThread
            device_id = -1
        elif non_blocking:
            ptr = runtime.streamCreateWithFlags(runtime.streamNonBlocking)
            device_id = runtime.getDevice()
        else:
            ptr = runtime.streamCreate()
            device_id = runtime.getDevice()
        super().__init__(ptr, device_id)

    def __del__(self, is_shutting_down=_util.is_shutting_down):
        if is_shutting_down():
            return
        if self.ptr not in (0, runtime.streamLegacy, runtime.streamPerThread):
            runtime.streamDestroy(self.ptr)
        # Note that we can not release memory pool of the stream held in CPU
        # because the memory would still be used in kernels executed in GPU.


class ExternalStream(_BaseStream):

    """CUDA stream not managed by CuPy.

    This class allows to use external streams in CuPy by providing the
    stream pointer obtained from the CUDA runtime call.
    The user is in charge of managing the life-cycle of the stream.

    Args:
        ptr (intptr_t): Address of the `cudaStream_t` object.
        device_id (int): The ID of the device that the stream was created on.
            Default is ``-1``, indicating it is unknown.

    Attributes:
        ~Stream.ptr (intptr_t): Raw stream handle.
        ~Stream.device_id (int): The ID of the device that the stream was
            created on. The value ``-1`` is used to indicate it is unknown.

    .. warning::
        If ``device_id`` is not specified, the user is required to ensure legal
        operations of the stream. Specifically, the stream must be used on the
        device that it was created on.

    """

    def __init__(self, ptr, device_id=-1):
        # It is in theory unsafe to just call runtime.getDevice() here, as the
        # stream pointer could come from a different device (although
        # unlikely). While we could use driver API combos cuStreamGetCtx ->
        # cuCtxSetCurrent -> cuCtxGetDevice -> ... to retrieve the device ID
        # associated with the stream, it is way too complicated and does not
        # work with HIP. Let us keep this as thin as possible.
        super().__init__(ptr, device_id)


Stream.null = Stream(null=True)
if not runtime._is_hip_environment:
    Stream.ptds = Stream(ptds=True)

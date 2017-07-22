from cupy.cuda import runtime
import threading


thread_local = threading.local()


def get_current_stream():
    if not hasattr(thread_local, 'current_stream'):
        thread_local.current_stream = Stream.null
    stream = thread_local.current_stream
    device = runtime.getDevice()
    if stream != Stream.null and stream.device != device:
        raise ValueError('Current stream device (%d) is different '
                         'with the current device (%d)' %
                         (stream.device, device))
    return stream


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
        ptr (cupy.cuda.runtime.Stream): Raw stream handle. It can be passed to
            the CUDA Runtime API via ctypes.

    """

    def __init__(self, block=False, disable_timing=False, interprocess=False):
        self.ptr = 0

        if interprocess and not disable_timing:
            raise ValueError('Timing must be disabled for interprocess events')
        flag = ((block and runtime.eventBlockingSync) |
                (disable_timing and runtime.eventDisableTiming) |
                (interprocess and runtime.eventInterprocess))
        self.ptr = runtime.eventCreateWithFlags(flag)

    def __del__(self):
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
            stream = get_current_stream()
        runtime.eventRecord(self.ptr, stream.ptr)

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


class Stream(object):

    """CUDA stream.

    This class handles the CUDA stream handle in RAII way, i.e., when an Stream
    instance is destroyed by the GC, its handle is also destroyed.

    Args:
        null (bool): If ``True``, the stream is a null stream (i.e. the default
            stream that synchronizes with all streams). Otherwise, a plain new
            stream is created. Users must not use this parameter, instead, use
            ``Stream.null`` object to use the default stream.
        non_blocking (bool): If ``True``, the stream does not synchronize with
            the NULL stream.

    Attributes:
        ptr (cupy.cuda.runtime.Stream): Raw stream handle. It can be passed to
            the CUDA Runtime API via ctypes.
        device (int): CUDA Device ID

    """

    null = None

    def __init__(self, null=False, non_blocking=False):
        if null and Stream.null:
            self.ptr = 0  # to avoid AttributeError on __del__
            raise ValueError('Use cupy.cuda.Stream.null instead of creating '
                             'a new cupy.cuda.Stream(null=True) object')
        if null:
            self.ptr = 0
            self.device = None  # any devices
        elif non_blocking:
            self.ptr = runtime.streamCreateWithFlags(runtime.streamNonBlocking)
            self.device = runtime.getDevice()
        else:
            self.ptr = runtime.streamCreate()
            self.device = runtime.getDevice()

    def __del__(self):
        if self.ptr:
            runtime.streamDestroy(self.ptr)

    def __enter__(self):
        if not hasattr(thread_local, 'prev_stream_stack'):
            thread_local.prev_stream_stack = []
        thread_local.prev_stream_stack.append(get_current_stream())
        thread_local.current_stream = self
        return self

    def __exit__(self, *args):
        thread_local.current_stream = thread_local.prev_stream_stack.pop()

    def use(self):
        """Makes this stream current.

        If you want to switch a stream temporarily, use the *with* statement.
        """
        thread_local.current_stream = self
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

        """
        def f(stream, status, dummy):
            callback(self, status, arg)
        runtime.streamAddCallback(self.ptr, f, 0)

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


Stream.null = Stream(null=True)

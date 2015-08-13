from cupy.cuda import runtime


class Event(object):

    """CUDA event, a synchronization point of CUDA streams.

    This class handles the CUDA event handle in RAII way, i.e., when an Event
    instance is destroyed by the GC, its handle is also destroyed.

    Args:
        block (bool): If True, the event blocks on the
            :meth:`~cupy.cuda.Event.synchronize` method.
        disable_timing (bool): If True, the event does not prepare the timing
            data.
        interprocess (bool): If True, the event can be passed to other
            processes.

    Attributes:
        ptr (cupy.cuda.runtime.Stream): Raw stream handle. It can be passed to
            the CUDA Runtime API via ctypes.

    """
    def __init__(self, block=False, disable_timing=False, interprocess=False):
        self.ptr = runtime.Event()

        if interprocess and not disable_timing:
            raise ValueError('Timing must be disabled for interprocess events')
        flag = ((block and runtime.eventBlockingSync) |
                (disable_timing and runtime.eventDisableTiming) |
                (interprocess and runtime.eventInterprocess))
        self.ptr = runtime.eventCreateWithFlags(flag)

    def __del__(self):
        if self.ptr:
            runtime.eventDestroy(self.ptr)
            self.ptr = runtime.Event()

    @property
    def done(self):
        """True if the event is done."""
        return bool(runtime.eventQuery(self.ptr))

    def record(self, stream=None):
        """Records the event to a stream.

        Args:
            stream (cupy.cuda.Stream): CUDA stream to record event. The null
                stream is used by default.

        .. seealso:: :meth:`cupy.cuda.Stream.record`

        """
        if stream is None:
            stream = Stream(null=True)
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
        null (bool): If True, the stream is a null stream (i.e. the default
            stream that synchronizes with all streams). Otherwise, a plain new
            stream is created.
        non_blocking (bool): If True, the stream does not synchronize with the
            NULL stream.

    Attributes:
        ptr (cupy.cuda.runtime.Stream): Raw stream handle. It can be passed to
            the CUDA Runtime API via ctypes.

    """
    def __init__(self, null=False, non_blocking=False):
        if null:
            self.ptr = runtime.Stream()
        elif non_blocking:
            self.ptr = runtime.streamCreateWithFlags(runtime.streamNonBlocking)
        else:
            self.ptr = runtime.streamCreate()

    def __del__(self):
        if self.ptr:
            runtime.streamDestroy(self.ptr)
            self.ptr = runtime.Stream()

    @property
    def done(self):
        """True if all work on this stream has been done."""
        return bool(runtime.streamQuery(self.ptr))

    def synchronize(self):
        """Waits for the stream completing all queued work."""
        runtime.streamSynchronize(self.ptr)

    def add_callback(self, callback, arg):
        """Adds a callback that is called when all queued work is done.

        Args:
            callback (function): Callback function. It must take three
                arguments (Stream object, int error status, and user data of
                type ctypes.c_void_p), and returns nothing.
            arg (ctypes.c_void_p): Argument to the callback.

        """
        runtime.streamAddCallback(self.ptr, callback, arg)

    def record(self, event=None):
        """Records an event on the stream.

        Args:
            event (None or cupy.cuda.Event): CUDA event. If None, then a new
                plain event is created and used.

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
        runtime.streamWaitEvent(self.ptr, event)

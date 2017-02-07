import contextlib
import functools

from cupy.cuda import nvtx  # NOQA


@contextlib.contextmanager
def timerange(message, id_color=-1):
    """A context manager to describe the enclosed block as a nested range

    >>> with cupy.prof.timerange('some range in green', 0):
    ...    # do something you want to measure
    ...    pass

    Args:
        message (str): Name of a range.
        id_color (int): ID of color for a range.

    .. seealso:: :func:`cupy.cuda.nvtx.RangePush`
        :func:`cupy.cuda.nvtx.RangePop`
    """
    nvtx.RangePush(message, id_color)
    try:
        yield
    finally:
        nvtx.RangePop()


@contextlib.contextmanager
def timerangeC(message, color=0):
    """A context manager to describe the enclosed block as a nested range

    >>> with cupy.prof.timerangeC('some range in green', 0xFF00FF00):
    ...    # do something you want to measure
    ...    pass

    Args:
        message (str): Name of a range.
        color (uint32): ARGB color for a range.

    .. seealso:: :func:`cupy.cuda.nvtx.RangePushC`
        :func:`cupy.cuda.nvtx.RangePop`
    """
    nvtx.RangePushC(message, color)
    try:
        yield
    finally:
        nvtx.RangePop()


class TimeRangeDecorator(object):
    """Decorator to mark function calls with range in NVIDIA profiler

    Decorated function calls are marked as ranges in NVIDIA profiler timeline.

    >>> @cupy.prof.TimeRangeDecorator()
    ... def function_to_profile():
    ...     pass

    Args:
        message (str): Name of a range, default use ``func.__name__``.
        id_color (int): ID of color for a range.

    .. seealso:: :func:`cupy.nvtx.range`
        :func:`cupy.cuda.nvtx.RangePush`
        :func:`cupy.cuda.nvtx.RangePop`
    """

    def __init__(self, message=None, id_color=0):
        self.message = message
        self.id_color = id_color

    def __enter__(self):
        nvtx.RangePush(self.message, self.id_color)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        nvtx.RangePop()

    def _recreate_cm(self, message):
        if self.message is None:
            self.message = message
        return self

    def __call__(self, func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            with self._recreate_cm(func.__name__):
                return func(*args, **kwargs)
        return inner

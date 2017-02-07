import contextlib
import functools

from cupy.cuda import nvtx  # NOQA


@contextlib.contextmanager
def time_range(message, color=-1, use_ARGB=False):
    """A context manager to describe the enclosed block as a nested range

    >>> with cupy.prof.time_range('some range in green', 0):
    ...    # do something you want to measure
    ...    pass

    Args:
        message: Name of a range.
        color: range color ID (int) or ARGB integer (uint32)
        use_ARGB: use ARGB color (e.g. 0xFF00FF00 for green) or not,
            default: ``False``, use color ID

    .. seealso:: :func:`cupy.cuda.nvtx.RangePush`
        :func:`cupy.cuda.nvtx.RangePop`
    """
    if use_ARGB:
        nvtx.RangePushC(message, color)
    else:
        nvtx.RangePush(message, color)
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
        color: range color ID (int) or ARGB integer (uint32)
        use_ARGB: use ARGB color (e.g. 0xFF00FF00 for green) or not,
            default: ``False``, use color ID

    .. seealso:: :func:`cupy.nvtx.range`
        :func:`cupy.cuda.nvtx.RangePush`
        :func:`cupy.cuda.nvtx.RangePop`
    """

    def __init__(self, message=None, color=0, use_ARGB=False):
        self.message = message
        self.color = color
        self.use_ARGB = use_ARGB

    def __enter__(self):
        if self.use_ARGB:
            nvtx.RangePushC(self.message, self.color)
        else:
            nvtx.RangePush(self.message, self.color)
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

import contextlib
import functools

from cupy import cuda
from cupy_backends.cuda.api import runtime


@contextlib.contextmanager
def time_range(message, color_id=None, argb_color=None, sync=False):
    """A context manager to describe the enclosed block as a nested range

    >>> from cupy import prof
    >>> with cupy.prof.time_range('some range in green', color_id=0):
    ...    # do something you want to measure
    ...    pass

    Args:
        message: Name of a range.
        color_id: range color ID
        argb_color: range color in ARGB (e.g. 0xFF00FF00 for green)
        sync (bool): If ``True``, waits for completion of all outstanding
            processing on GPU before calling :func:`cupy.cuda.nvtx.RangePush()`
            or :func:`cupy.cuda.nvtx.RangePop()`

    .. seealso:: :func:`cupy.cuda.nvtx.RangePush`
        :func:`cupy.cuda.nvtx.RangePop`
    """
    if not cuda.nvtx.available:
        raise RuntimeError('nvtx is not installed')

    if color_id is not None and argb_color is not None:
        raise ValueError('Only either color_id or argb_color can be specified')

    if sync:
        runtime.deviceSynchronize()
    if argb_color is not None:
        cuda.nvtx.RangePushC(message, argb_color)
    else:
        if color_id is None:
            color_id = -1
        cuda.nvtx.RangePush(message, color_id)
    try:
        yield
    finally:
        if sync:
            runtime.deviceSynchronize()
        cuda.nvtx.RangePop()


class TimeRangeDecorator(object):
    """Decorator to mark function calls with range in NVIDIA profiler

    Decorated function calls are marked as ranges in NVIDIA profiler timeline.

    >>> from cupy import prof
    >>> @cupy.prof.TimeRangeDecorator()
    ... def function_to_profile():
    ...     pass

    Args:
        message (str): Name of a range, default use ``func.__name__``.
        color_id: range color ID
        argb_color: range color in ARGB (e.g. 0xFF00FF00 for green)
        sync (bool): If ``True``, waits for completion of all outstanding
            processing on GPU before calling :func:`cupy.cuda.nvtx.RangePush()`
            or :func:`cupy.cuda.nvtx.RangePop()`

    .. seealso:: :func:`cupy.cuda.nvtx.RangePush`
        :func:`cupy.cuda.nvtx.RangePop`
    """

    def __init__(self, message=None, color_id=None, argb_color=None,
                 sync=False):
        if not cuda.nvtx.available:
            raise RuntimeError('nvtx is not installed')

        if color_id is not None and argb_color is not None:
            raise ValueError(
                'Only either color_id or argb_color can be specified'
            )
        self.message = message
        self.color_id = color_id if color_id is not None else -1
        self.argb_color = argb_color
        self.sync = sync

    def __enter__(self):
        if self.sync:
            runtime.deviceSynchronize()
        if self.argb_color is not None:
            cuda.nvtx.RangePushC(self.message, self.argb_color)
        else:
            cuda.nvtx.RangePush(self.message, self.color_id)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.sync:
            runtime.deviceSynchronize()
        cuda.nvtx.RangePop()

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

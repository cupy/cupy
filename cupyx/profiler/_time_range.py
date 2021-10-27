import functools

from cupy import cuda
from cupy_backends.cuda.api import runtime


# Note: We use an (old-fashioned) custom object instead of
# @contextlib.contextmanager because for backward compatibility
# when used as a decorator this object needs to fetch the target
# function name.
class time_range:
    """Mark function calls with ranges using NVTX/rocTX. This object can be
    used either as a decorator or a context manager.

    When used as a decorator, the decorated function calls are marked as
    ranges:

    >>> from cupyx.profiler import time_range
    >>> @time_range()
    ... def function_to_profile():
    ...     pass

    When used as a context manager, it describes the enclosed block as a nested
    range:

    >>> from cupyx.profiler import time_range
    >>> with time_range('some range in green', color_id=0):
    ...    # do something you want to measure
    ...    pass

    The marked ranges are visible in the profiler (such as nvvp, nsys-ui, etc)
    timeline.

    Args:
        message (str): Name of a range. When used as a decorator, the default
            is ``func.__name__``.
        color_id: range color ID
        argb_color: range color in ARGB (e.g. 0xFF00FF00 for green)
        sync (bool): If ``True``, waits for completion of all outstanding
            processing on GPU before calling :func:`cupy.cuda.nvtx.RangePush()`
            or :func:`cupy.cuda.nvtx.RangePop()`

    .. seealso:: :func:`cupy.cuda.nvtx.RangePush`,
        :func:`cupy.cuda.nvtx.RangePop`
    """

    def __init__(
            self, message=None, color_id=None, argb_color=None, sync=False):
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
        if self.message is None:
            raise ValueError(
                'when used as a context manager, the message argument cannot '
                'be None')
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

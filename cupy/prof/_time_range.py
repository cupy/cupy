import contextlib
import warnings

from cupyx.profiler import time_range as _time_range


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

    .. warning:: This context manager is deprecated. Please use
        :class:`cupyx.profiler.time_range` instead.
    """
    warnings.warn(
        'cupy.prof.time_range has been deprecated since CuPy v10 '
        'and will be removed in the future. Use cupyx.profiler.time_range '
        'instead.')

    with _time_range(message, color_id, argb_color, sync):
        yield


class TimeRangeDecorator:
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

    .. warning:: This decorator is deprecated. Please use
        :class:`cupyx.profiler.time_range` instead.
    """

    _init = _time_range.__init__
    __enter__ = _time_range.__enter__
    __exit__ = _time_range.__exit__
    __call__ = _time_range.__call__
    _recreate_cm = _time_range._recreate_cm

    def __init__(self, message=None, color_id=None, argb_color=None,
                 sync=False):

        warnings.warn(
            'cupy.prof.TimeRangeDecorator has been deprecated since CuPy v10 '
            'and will be removed in the future. Use cupyx.profiler.time_range '
            'instead.')

        self._init(message, color_id, argb_color, sync)

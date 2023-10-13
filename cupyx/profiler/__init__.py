import contextlib as _contextlib
from cupy.cuda import runtime as _runtime
from cupyx.profiler._time import benchmark  # NOQA
from cupyx.profiler._time_range import time_range  # NOQA


@_contextlib.contextmanager
def profile():
    """Enable CUDA profiling during with statement.

    This function enables profiling on entering a with statement, and disables
    profiling on leaving the statement.

    >>> with cupyx.profiler.profile():
    ...    # do something you want to measure
    ...    pass

    .. note::
        When starting ``nvprof`` from the command line, manually setting
        ``--profile-from-start off`` may be required for the desired behavior.
        Likewise, when using ``nsys profile`` setting ``-c cudaProfilerApi``
        may be required.

    .. seealso:: :func:`cupy.cuda.runtime.profilerStart`,
        :func:`cupy.cuda.runtime.profilerStop`
    """
    _runtime.profilerStart()
    try:
        yield
    finally:
        _runtime.profilerStop()

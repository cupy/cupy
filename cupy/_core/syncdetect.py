import contextlib
import threading

from cupy import _util


_thread_local = threading.local()


class DeviceSynchronized(RuntimeError):
    """Raised when device synchronization is detected while disallowed.

    .. seealso:: :func:`cupyx.allow_synchronize`

    """

    def __init__(self, message=None):
        if message is None:
            message = 'Device synchronization was detected while disallowed.'
        super().__init__(message)


def _is_allowed():
    # Returns whether device synchronization is allowed in the current thread.
    try:
        return _thread_local.allowed
    except AttributeError:
        _thread_local.allowed = True
        return True


def _declare_synchronize():
    # Raises DeviceSynchronized if device synchronization is disallowed in
    # the current thread.
    if not _is_allowed():
        raise DeviceSynchronized()


@contextlib.contextmanager
def allow_synchronize(allow):
    """Allows or disallows device synchronization temporarily in the current \
thread.

    If device synchronization is detected, :class:`cupyx.DeviceSynchronized`
    will be raised.

    Note that there can be false negatives and positives.
    Device synchronization outside CuPy will not be detected.
    """
    _util.experimental('cupyx.allow_synchronize')
    old = _is_allowed()
    _thread_local.allowed = allow
    try:
        yield
    finally:
        _thread_local.allowed = old

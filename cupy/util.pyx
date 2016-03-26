import atexit
import functools

from cupy.cuda cimport device


cdef list _memos = []


def memoize(bint for_each_device=False):
    """Makes a function memoizing the result for each argument and device.

    This decorator provides automatic memoization of the function result.

    Args:
        for_each_device (bool): If ``True``, it memoizes the results for each
            device. Otherwise, it memoizes the results only based on the
            arguments.

    """
    def decorator(f):
        memo = {}
        _memos.append(memo)
        none = object()

        @functools.wraps(f)
        def ret(*args, **kwargs):
            cdef int id = -1
            cdef dict m = memo
            if for_each_device:
                id = device.get_device_id()
            arg_key = (id, args, frozenset(kwargs.items()))
            if arg_key in m:
                result = m[arg_key]
            else:
                result = f(*args, **kwargs)
                m[arg_key] = result
            return result

        return ret

    return decorator


@atexit.register
def clear_memo():
    """Clears the memoized results for all functions decorated by memoize."""
    for memo in _memos:
        memo.clear()

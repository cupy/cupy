import atexit
import functools

from cupy cimport cuda


_memos = []


def memoize(for_each_device=False):
    """Makes a function memoizing the result for each argument and device.

    This decorator provides automatic memoization of the function result.

    Args:
        for_each_device (bool): If True, it memoizes the results for each
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
            if for_each_device:
                id = cuda.Device().id
            arg_key = (id, args, frozenset(kwargs.items()))
            result = memo.get(arg_key, none)
            if result is none:
                result = f(*args, **kwargs)
                memo[arg_key] = result
            return result

        return ret

    return decorator


@atexit.register
def clear_memo():
    """Clears the memoized results for all functions decorated by memoize."""
    for memo in _memos:
        memo.clear()

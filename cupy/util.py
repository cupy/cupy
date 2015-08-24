import atexit
import functools

from cupy import cuda


_memoized_funcs = []


def memoize(for_each_device=False):
    """Makes a function memoizing the result for each argument and device.

    This decorator provides automatic memoization of the function result.

    Args:
        for_each_device (bool): If True, it memoizes the results for each
            device. Otherwise, it memoizes the results only based on the
            arguments.

    """
    def decorator(f):
        global _memoized_funcs
        f._cupy_memo = {}
        _memoized_funcs.append(f)

        @functools.wraps(f)
        def ret(*args, **kwargs):
            arg_key = (args, frozenset(kwargs.items()))
            if for_each_device:
                arg_key = (cuda.Device().id, arg_key)

            memo = f._cupy_memo
            result = memo.get(arg_key, None)
            if result is None:
                result = f(*args, **kwargs)
                memo[arg_key] = result
            return result

        return ret

    return decorator


@atexit.register
def clear_memo():
    """Clears the memoized results for all functions decorated by memoize."""
    global _memoized_funcs
    for f in _memoized_funcs:
        del f._cupy_memo
    _memoized_funcs = []

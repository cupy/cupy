import atexit


_memoized_funcs = []


def memoize(f):
    """Makes a function memoizing the result for each argument.

    This decorator provides global memoizing of the function result.

    """
    def func(*args, **kwargs):
        # TODO(okuta): Improve keyword arguments.
        global _memoized_funcs

        if not hasattr(f, '_cupy_memo'):
            _memoized_funcs.append(f)
            f._cupy_memo = {}

        memo = f._cupy_memo
        arg_key = (args, frozenset(kwargs.items()))
        result = memo.get(arg_key, None)
        if result is None:
            result = f(*args, **kwargs)
            memo[arg_key] = result
        return result

    return func


@atexit.register
def clear_memo():
    """Clears the memoized results for all functions decorated by memoize."""
    global _memoized_funcs
    for func in _memoized_funcs:
        del func._cupy_memo
    _memoized_funcs = []

# distutils: language = c++

import atexit
import functools
import warnings

import cupy
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


def experimental(api_name):
    """Declares that user is using an experimental feature.

    The developer of an API can mark it as *experimental* by calling
    this function. When users call experimental APIs, :class:`FutureWarning`
    is issued.
    The presentation of :class:`FutureWarning` is disabled by setting
    ``cupy.disable_experimental_warning`` to ``True``,
    which is ``False`` by default.

    The basic usage is to call it in the function or method we want to
    mark as experimental along with the API name.

    .. testsetup::

        import sys
        import warnings

        warnings.simplefilter('always')

        def wrapper(message, category, filename, lineno, file=None, line=None):
            sys.stdout.write(warnings.formatwarning(
                message, category, filename, lineno))

        showwarning_orig = warnings.showwarning
        warnings.showwarning = wrapper

    .. testcleanup::

        warnings.showwarning = showwarning_orig

    .. testcode::

        from cupy import util

        def f(x):
            util.experimental('cupy.foo.bar.f')
            # concrete implementation of f follows

        f(1)

    .. testoutput::
        :options: +ELLIPSIS, +NORMALIZE_WHITESPACE

        ... FutureWarning: cupy.foo.bar.f is experimental. \
The interface can change in the future. ...

    We can also make a whole class experimental. In that case,
    we should call this function in its ``__init__`` method.

    .. testcode::

        class C():
            def __init__(self):
              util.experimental('cupy.foo.C')

        C()

    .. testoutput::
        :options: +ELLIPSIS, +NORMALIZE_WHITESPACE

        ... FutureWarning: cupy.foo.C is experimental. \
The interface can change in the future. ...

    If we want to mark ``__init__`` method only, rather than class itself,
    it is recommended that we explicitly feed its API name.

    .. testcode::

        class D():
            def __init__(self):
                util.experimental('D.__init__')

        D()

    .. testoutput::
        :options: +ELLIPSIS, +NORMALIZE_WHITESPACE

        ...  FutureWarning: D.__init__ is experimental. \
The interface can change in the future. ...

    Currently, we do not have any sophisticated way to mark some usage of
    non-experimental function as experimental.
    But we can support such usage by explicitly branching it.

    .. testcode::

        def g(x, experimental_arg=None):
            if experimental_arg is not None:
                util.experimental('experimental_arg of cupy.foo.g')

    Args:
        api_name(str): The name of an API marked as experimental.
    """

    if not cupy.disable_experimental_feature_warning:
        warnings.warn('{} is experimental. '
                      'The interface can change in the future.'.format(
                          api_name),
                      FutureWarning)

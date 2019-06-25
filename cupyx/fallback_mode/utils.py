"""
Utilities needed for fallback_mode.
"""

import logging
import warnings
import threading

from cupyx.fallback_mode import data_transfer


def _call_cupy(func, args, kwargs):
    """
    Calls cupy function with *args and **kwargs.

    Args:
        func: A cupy function that needs to be called.
        args (tuple): Arguments.
        kwargs (dict): Keyword arguments.

    Returns:
        Result after calling func.
    """

    return func(*args, **kwargs)


def _call_numpy(func, args, kwargs):
    """
    Calls numpy function with *args and **kwargs.

    Args:
        func: A numpy function that needs to be called.
        args (tuple): Arguments.
        kwargs (dict): Keyword arguments.

    Returns:
        Result after calling func.
    """

    numpy_args, numpy_kwargs = data_transfer._get_numpy_args(args, kwargs)

    numpy_res = func(*numpy_args, **numpy_kwargs)

    cupy_res = data_transfer._get_cupy_result(numpy_res)

    return cupy_res


_thread_locals = threading.local()
_thread_locals.dispatch_type = 'warn'

FallbackWarning = type('FallbackWarning', (Warning,), {})
warnings.simplefilter(action='always', category=FallbackWarning)


def geterr():
    return _thread_locals.dispatch_type


def seterr(new_dispatch):

    old = _thread_locals.dispatch_type

    if new_dispatch in ['print', 'warn', 'log', 'ignore']:
        _thread_locals.dispatch_type = new_dispatch
        return old

    raise ValueError('{} is not valid dispatch type'.format(new_dispatch))


def dispatch_notification(func):

    msg = "Requested method not in cupy, falling back to '{}.{}'".format(
        func.__module__, func.__name__)

    if _thread_locals.dispatch_type == 'print':
        print("Warning: {}".format(msg))

    elif _thread_locals.dispatch_type == 'warn':
        warnings.warn(msg, FallbackWarning)

    elif _thread_locals.dispatch_type == 'log':
        logging.warning(msg)

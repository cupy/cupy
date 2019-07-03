"""
Methods related to notifications.
"""

import warnings
import threading

_thread_locals = threading.local()
_thread_locals.dispatch_type = 'warn'
_thread_locals.callback = None

FallbackWarning = type('FallbackWarning', (Warning,), {})
warnings.simplefilter(action='always', category=FallbackWarning)


def geterr():
    return _thread_locals.dispatch_type


def seterr(new_dispatch):

    old = _thread_locals.dispatch_type

    if new_dispatch in ['print', 'warn', 'log', 'ignore', 'raise', 'call']:
        _thread_locals.dispatch_type = new_dispatch
        return old

    raise ValueError('{} is not valid dispatch type'.format(new_dispatch))


def geterrcall():
    return _thread_locals.callback


def seterrcall(func):

    if not callable(func):
        if not hasattr(func, 'write') or not callable(func.write):
            raise ValueError('Only callable can be used as Callback')

    old = _thread_locals.callback
    _thread_locals.callback = func
    return old


def dispatch_notification(func):

    msg = "'{}' method not in cupy, falling back to '{}.{}'".format(
        func.__name__, func.__module__, func.__name__)

    raise_msg = "'{}' method not found in cupy".format(
        func.__name__)

    if _thread_locals.dispatch_type == 'print':
        print("Warning: {}".format(msg))

    elif _thread_locals.dispatch_type == 'warn':
        warnings.warn(msg, FallbackWarning, stacklevel=3)

    elif _thread_locals.dispatch_type == 'ignore':
        pass

    elif _thread_locals.dispatch_type == 'log':

        callback_func = _thread_locals.callback

        if hasattr(callback_func, 'write') and callable(callback_func.write):
            callback_func.write(msg)
        else:
            raise ValueError(
                "Callback object must have a callable 'write' method, " +
                "if it is to be used for 'log'")

    elif _thread_locals.dispatch_type == 'raise':
        raise AttributeError(raise_msg)

    elif _thread_locals.dispatch_type == 'call':

        callback_func = _thread_locals.callback

        if callable(callback_func):
            callback_func(func)

        else:
            raise ValueError(
                "Callback method must be callable, " +
                "if it is to be used for 'call'")

    else:
        assert False


class errstate:

    def __init__(self, new_dispatch):
        self.old = None
        self.new = new_dispatch

    def __enter__(self):
        self.old = _thread_locals.dispatch_type
        seterr(self.new)

    def __exit__(self, *exc_info):
        seterr(self.old)

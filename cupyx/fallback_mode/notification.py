"""
Methods related to notifications.
"""

import warnings
import threading

_thread_locals = threading.local()
_thread_locals.dispatch_type = 'warn'
_thread_locals.callback = None


def _init_warnings():
    FallbackWarning = type('FallbackWarning', (Warning,), {})
    warnings.simplefilter(action='always', category=FallbackWarning)
    return FallbackWarning


def geterr():
    """
    Returns current notification dispatch type.
    """
    return _thread_locals.dispatch_type


def seterr(new_dispatch):
    """
    Sets notification dispatch type. These are different dispatch types:
    'warn': Print warning using `warnings` module.
    'print': Prints warning to stdout.
    'log': Record message in log object specified by `seterrcall`.
    'ignore': No action is taken.
    'raise': Raise a AttributeError.
    'call': Call function which was specified by `seterrcall`.
    'warn' is the default dispatch type. If not a correct dispatch type
    raises ValueError.

    Args:
        new_dispatch (str): Notification dispatch type to be set.

    Returns:
        old (str): Old notification dispatch type.
    """

    old = _thread_locals.dispatch_type

    if new_dispatch in ['print', 'warn', 'log', 'ignore', 'raise', 'call']:
        _thread_locals.dispatch_type = new_dispatch
        return old

    raise ValueError('{} is not valid dispatch type'.format(new_dispatch))


def geterrcall():
    """
    Returns the current callback object.
    """
    return _thread_locals.callback


def seterrcall(func):
    """
    Sets a callback function or log object for fallback notifications.
    There are two ways to set an object for callback.

    First is to set the error-handler to 'call', using `seterr`. Then use
    this function to set the callback function. Callback function accepts
    numpy function to which execution falls back.

    Second is to set the error-handler to 'log', using `seterr`. Then use
    this function to set the log object. Log object must have defined a
    'write' method, which needs to be triggered at fallback.

    If these conditions are not met, ValueError is raised.

    Args:
        func: Callback function or log object with 'write' method.

    Returns:
        old: Old callback function or log object.
    """
    if not callable(func):
        if not hasattr(func, 'write') or not callable(func.write):
            raise ValueError('Only callable can be used as Callback')

    old = _thread_locals.callback
    _thread_locals.callback = func
    return old


def _dispatch_notification(func):
    """
    Dispatch notifications using appropriate dispatch type.
    """

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
    """
    Context manager for handling fallback notifications.

    Sets to new dispatch type using `seterr` upon entering the context,
    and upon exiting it is set to old dispatch type.

    Args:
        new_dispatch (str): Notification dispatch type to be set.
    """
    def __init__(self, new_dispatch):
        self.old = None
        self.new = new_dispatch

    def __enter__(self):
        self.old = _thread_locals.dispatch_type
        seterr(self.new)

    def __exit__(self, *exc_info):
        seterr(self.old)


FallbackWarning = _init_warnings()

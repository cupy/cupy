"""
Methods related to notifications.
"""

import warnings

from cupyx import _ufunc_config


def _init_warnings():
    FallbackWarning = type('FallbackWarning', (Warning,), {})
    warnings.simplefilter(action='always', category=FallbackWarning)
    return FallbackWarning


def _dispatch_notification(func):
    """
    Dispatch notifications using appropriate dispatch type.
    """

    dispatch_type = _ufunc_config.get_config_fallback_mode()
    callback_func = _ufunc_config.get_fallback_mode_callback()

    msg = "'{}' method not in cupy, falling back to '{}.{}'".format(
        func.__name__, func.__module__, func.__name__)

    raise_msg = "'{}' method not found in cupy".format(
        func.__name__)

    if dispatch_type == 'print':
        print("Warning: {}".format(msg))

    elif dispatch_type == 'warn':
        warnings.warn(msg, FallbackWarning, stacklevel=3)

    elif dispatch_type == 'ignore':
        pass

    elif dispatch_type == 'log':
        if hasattr(callback_func, 'write') and callable(callback_func.write):
            callback_func.write(msg)
        else:
            raise ValueError(
                "Callback object must have a callable 'write' method, " +
                "if it is to be used for 'log'")

    elif dispatch_type == 'raise':
        raise AttributeError(raise_msg)

    elif dispatch_type == 'call':
        if callable(callback_func):
            callback_func(func)
        else:
            raise ValueError(
                "Callback method must be callable, " +
                "if it is to be used for 'call'")

    else:
        assert False


FallbackWarning = _init_warnings()

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

    _module = hasattr(func, '__module__')
    _name = hasattr(func, '__name__')

    if _name and _module:
        msg = "'{}' method not in cupy, falling back to '{}.{}'".format(
            func.__name__, func.__module__, func.__name__)
    elif _name:
        msg = "'{}' method not in cupy, ".format(func.__name__)
        msg += "falling back to numpy implementation of '{}'".format(
            func.__name__)
    else:
        msg = "This method is not available in cupy, "
        msg += "falling back to numpy"

    if _name:
        raise_msg = "'{}' method not found in cupy".format(func.__name__)
    else:
        raise_msg = "This method is not available in cupy"

    if dispatch_type == 'print':
        print("Warning: {}".format(msg))

    elif dispatch_type == 'warn':
        warnings.warn(msg, FallbackWarning, stacklevel=3)

    elif dispatch_type == 'ignore':
        pass

    elif dispatch_type == 'raise':
        raise AttributeError(raise_msg)

    else:
        assert False


FallbackWarning = _init_warnings()

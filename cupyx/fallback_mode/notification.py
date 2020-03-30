"""
Methods related to notifications.
"""

import warnings

from cupyx import _ufunc_config


def _init_warnings():
    FallbackWarning = type('FallbackWarning', (Warning,), {})
    warnings.simplefilter(action='always', category=FallbackWarning)
    return FallbackWarning


def _dispatch_notification(func, cupy_support=False):
    """
    Dispatch notifications using appropriate dispatch type.
    """

    dispatch_type = _ufunc_config.get_config_fallback_mode()

    _module = getattr(func, '__module__', None)
    _name = getattr(func, '__name__', None)

    if not cupy_support:
        if _name and _module:
            msg = "'{}' method not in cupy, falling back to '{}.{}'".format(
                _name, _module, _name)
        elif _name:
            msg = "'{}' method not in cupy, ".format(_name)
            msg += "falling back to its numpy implementation"
        else:
            msg = "This method is not available in cupy, "
            msg += "falling back to numpy"

        if _name:
            raise_msg = "'{}' method not found in cupy".format(_name)
        else:
            raise_msg = "This method is not available in cupy"
    else:
        if _name and _module:
            msg = "'{}' method is available in cupy but ".format(_name)
            msg += "cannot be used, falling back to '{}.{}'".format(
                _module, _name)
        elif _name:
            msg = "'{}' method is available in cupy but ".format(_name)
            msg += "cannot be used, falling back to its numpy implementation"
        else:
            msg = "This method is available in cupy, but cannot be used"
            msg += "falling back to numpy"

        if _name:
            raise_msg = "'{}' method is available in cupy ".format(_name)
            raise_msg += "but cannot be used"
        else:
            raise_msg = "This method is available in cupy but cannot be used"

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

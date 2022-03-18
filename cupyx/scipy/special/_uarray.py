import numpy as np
import cupy


try:
    import scipy.special as _scipy_special
except ImportError:
    _scipy_special = None


# Backend support for scipy.special

__ua_domain__ = 'numpy.scipy.special'
_implemented = {}  # type: ignore
_notfound = []  # for test


def __ua_convert__(dispatchables, coerce):
    if coerce:
        try:
            replaced = [
                cupy.asarray(d.value) if d.coercible and d.type is np.ndarray
                else d.value for d in dispatchables]
        except TypeError:
            return NotImplemented
    else:
        replaced = [d.value for d in dispatchables]

    if not all(d.type is not np.ndarray or isinstance(r, cupy.ndarray)
               for r, d in zip(replaced, dispatchables)):
        return NotImplemented

    return replaced


def __ua_function__(method, args, kwargs):
    fn = _implemented.get(method, None)
    if fn is None:
        return NotImplemented
    return fn(*args, **kwargs)


def implements(scipy_func_name):
    """Decorator adds function to the dictionary of implemented functions"""
    def inner(func):
        scipy_func = (
            _scipy_special and getattr(_scipy_special, scipy_func_name, None))
        if scipy_func:
            _implemented[scipy_func] = func
        else:
            _notfound.append(scipy_func_name)
        return func

    return inner


def implements_ufuncs(cp_func, func_name):
    """This adds function to the dictionary of implemented functions"""
    scipy_func = getattr(_scipy_special, func_name)
    _implemented[scipy_func] = cp_func

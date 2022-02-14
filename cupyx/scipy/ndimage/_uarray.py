import numpy as np
import cupy


try:
    import scipy.ndimage as _scipy_ndimage
except ImportError:
    class _DummyModule:
        def __getattr__(self, name):
            return None

    _scipy_ndimage = _DummyModule()


# Backend support for scipy.ndimage

__ua_domain__ = 'numpy.scipy.ndimage'
_implemented = {}


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
        scipy_func = getattr(_scipy_ndimage, scipy_func_name)
        _implemented[scipy_func] = func
        return func

    return inner

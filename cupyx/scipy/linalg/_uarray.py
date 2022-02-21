import numpy as np
import cupy
import cupy.linalg as _cp_linalg


try:
    import scipy.linalg as _scipy_linalg
except ImportError:
    class _DummyModule:
        def __getattr__(self, name):
            return None

    _scipy_linalg = _DummyModule()


# Backend support for scipy.linalg

__ua_domain__ = 'numpy.scipy.linalg'
_implemented = {}  # type: ignore


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
        scipy_func = getattr(_scipy_linalg, scipy_func_name)
        _implemented[scipy_func] = func
        return func

    return inner


# cupy linalg functions

_cp_linalg_functions = [
    # cupy.linalg._eigenvalue
    'eigh', 'eigvalsh',
    # cupy.linalg._decomposition
    'cholesky', 'qr', 'svd',
    # cupy.linalg._norms
    'norm', 'det',
    # cupy.linalg._solve
    'solve', 'lstsq', 'inv', 'pinv'
]

for func_name in _cp_linalg_functions:
    cp_func = getattr(_cp_linalg, func_name)
    scipy_func = getattr(_scipy_linalg, func_name)
    _implemented[scipy_func] = cp_func

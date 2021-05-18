import sys as _sys

from cupy._core import ndarray as _ndarray
from cupyx.scipy.sparse.base import spmatrix as _spmatrix


try:
    import scipy as _scipy
    _scipy_available = True
except ImportError:
    _scipy_available = False


_cupyx_scipy = _sys.modules[__name__]


def get_array_module(*args):
    """Returns the array module for arguments.

    This function is used to implement CPU/GPU generic code. If at least one of
    the arguments is a :class:`cupy.ndarray` object, the :mod:`cupyx.scipy`
    module is returned.

    Args:
        args: Values to determine whether NumPy or CuPy should be used.

    Returns:
        module: :mod:`cupyx.scipy` or :mod:`scipy` is returned based on the
        types of the arguments.

    """
    for arg in args:
        if isinstance(arg, (_ndarray, _spmatrix)):
            return _cupyx_scipy
    return _scipy

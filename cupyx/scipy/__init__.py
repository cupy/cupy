import sys

from cupy.core import ndarray
from cupyx.scipy.sparse.base import spmatrix


try:
    import scipy
    _scipy_available = True
except ImportError:
    _scipy_available = False


_cupyx_scipy = sys.modules[__name__]


def get_array_module(*args):
    """Returns the array module for arguments.

    This function is used to implement CPU/GPU generic code. If at least one of
    the arguments is a :class:`cupy.ndarray` object, the :mod:`cupy` module is
    returned.

    Args:
        args: Values to determine whether NumPy or CuPy should be used.

    Returns:
        module: :mod:`cupyx.scipy` or :mod:`scipy` is returned based on the
        types of the arguments.

    """
    for arg in args:
        if isinstance(arg, (ndarray, spmatrix)):
            return _cupyx_scipy
    return scipy

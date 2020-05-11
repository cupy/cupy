import cupy

from cupy import core
from cupy.core.core import ndarray
from cupy.core._reduction import ReductionKernel

try:
    from collections.abc import Callable
except ImportError:
    from collections import Callable

_piecewise_krnl = ReductionKernel(
    'S x1, T x2',
    'U y',
    'x1 ? x2 : NULL',
    'b == NULL? a : b',
    'y = a',
    'NULL',
    'piecewise'
)


def piecewise(x, condlist, funclist):
    """
    Evaluate a piecewise-defined function.

    Args:
    :param x: input domain
    :param condlist: conditions list ( boolean arrays or boolean scalars)
                     Each boolean array/ scalar corresponds to a function
                     in funclist. Length of functionlist is equal to that
                     of condlist. If one extra function is given, it is used
                     as otherwise condition
    :param funclist: list of scalar functions.

    Returns:
        cupy.ndarray: the result of calling the functions in funclist
        on portions of x defined by condlist.

    .. warning::

        This function currently doesn't support callable functions
        args and kw parameters are not supported

    .. seealso:: :func:`numpy.piecewise`
    """

    if any(isinstance(item, Callable) for item in funclist):
        raise ValueError('Callable functions are not supported')
    if cupy.isscalar(x):
        x = cupy.asarray(x)
    scalar = 0
    if cupy.isscalar(condlist):
        scalar = 1
    if cupy.isscalar(condlist) or (
            (not isinstance(condlist[0], (list, ndarray))) and x.ndim != 0):
        condlist = [condlist]
    condlist = cupy.array(condlist, dtype=bool)
    condlen = len(condlist)
    funclen = len(funclist)
    if condlen + 1 == funclen:  # o.w
        condelse = ~ cupy.any(condlist, axis=0, keepdims=True)
        condlist = core.concatenate_method([condlist, condelse], 0)
        condlen += 1
    elif condlen != funclen:
        raise ValueError('with {} condition(s), either {} or {} functions'
                         ' are expected'.format(condlen, condlen, condlen + 1))

    funclist = cupy.asarray(funclist)
    if scalar:
        funclist = funclist[condlist]
        condlist = cupy.ones(shape=(1, x.size), dtype=bool)

    y = cupy.zeros(x.shape, x.dtype)
    if not x.ndim:
        _piecewise_krnl(condlist, funclist, y)
    else:
        for i in range(x.size):
            _piecewise_krnl(condlist[:, i], funclist, y[i])
    return y

import cupy

from cupy import core
from cupy.core.core import ndarray

_piecewise_krnl = core.ElementwiseKernel(
    'U condlist, S funclist',
    'raw T y',
    ' if (condlist) y[0] = funclist',
    'piecewise_kernel'
)


def piecewise(x, condlist, funclist):
    """Evaluate a piecewise-defined function.

        Args:
            x: input domain
            condlist: conditions list ( boolean arrays or boolean scalars).
                Each boolean array/ scalar corresponds to a function
                in funclist. Length of functionlist is equal to that
                of condlist. If one extra function is given, it is used
                as otherwise condition
            funclist: list of scalar functions.

        Returns:
            cupy.ndarray: the result of calling the functions in funclist
                on portions of x defined by condlist.

        .. warning::

            This function currently doesn't support callable functions
            args and kw parameters are not supported

        .. seealso:: :func:`numpy.piecewise`
        """

    if any(callable(item) for item in funclist):
        raise NotImplementedError(
            'Callable functions are not supported currently')
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
        y = cupy.full(shape=x.size, fill_value=funclist[-1], dtype=x.dtype)
        funclist = funclist[:-1]
        funclen -= 1
    elif condlen != funclen:
        raise ValueError('with {} condition(s), either {} or {} functions'
                         ' are expected'.format(condlen, condlen, condlen + 1))
    else:
        y = cupy.zeros(shape=x.shape, dtype=x.dtype)
    funclist = cupy.asarray(funclist)
    if scalar:
        funclist = funclist[condlist]
        condlist = cupy.ones(shape=(1, x.size), dtype=bool)
    if not x.ndim:
        _piecewise_krnl(condlist, funclist, y)
    else:
        for i in range(x.size):
            _piecewise_krnl(condlist.T[i], funclist, y[i])
    return y

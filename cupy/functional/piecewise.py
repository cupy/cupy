import cupy

from cupy import core

_piecewise_krnl = core.ElementwiseKernel(
    'U condlist, S funclist',
    'T y',
    ' if(condlist) y = funclist',
    'piecewise_kernel'
)


def piecewise(x, condlist, funclist):
    """Evaluate a piecewise-defined function.

        Args:
            x (cupy.ndarray): input domain
            condlist (cupy.ndarray or list):
                conditions list ( boolean arrays or boolean scalars).
                Each boolean array/ scalar corresponds to a function
                in funclist. Length of funclist is equal to that
                of condlist. If one extra function is given, it is used
                as otherwise condition
            funclist (cupy.ndarray or list): list of scalar functions.

        Returns:
            cupy.ndarray: the result of calling the functions in funclist
                on portions of x defined by condlist.

        .. warning::

            This function currently doesn't support callable functions,
            args and kw parameters.

        .. seealso:: :func:`numpy.piecewise`
        """

    if any(callable(item) for item in funclist):
        raise NotImplementedError(
            'Callable functions are not supported currently')
    if cupy.isscalar(condlist) or (not isinstance(
            condlist[0], (list, cupy.ndarray)) and x.ndim != 0):
        condlist = [condlist]
    condlist = cupy.array(condlist, dtype=bool)
    condlen = len(condlist)
    funclen = len(funclist)
    if condlen == funclen:
        y = cupy.zeros(shape=x.shape, dtype=x.dtype)
    elif condlen + 1 == funclen:  # o.w
        y = cupy.full(shape=x.shape, fill_value=funclist[-1], dtype=x.dtype)
        funclist = funclist[:-1]
        funclen -= 1
    else:
        raise ValueError('with {} condition(s), either {} or {} functions'
                         ' are expected'.format(condlen, condlen, condlen + 1))
    funclist = cupy.asarray(funclist)
    for i in range(funclen):
        _piecewise_krnl(condlist[i], funclist[i], y)
    return y

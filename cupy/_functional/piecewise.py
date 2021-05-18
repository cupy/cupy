import cupy

from cupy import _core

_piecewise_krnl = _core.ElementwiseKernel(
    'bool cond, T value',
    'T y',
    'if (cond) y = value',
    'piecewise_kernel'
)


def piecewise(x, condlist, funclist):
    """Evaluate a piecewise-defined function.

        Args:
            x (cupy.ndarray): input domain
            condlist (list of cupy.ndarray):
                Each boolean array/ scalar corresponds to a function
                in funclist. Length of funclist is equal to that of
                condlist. If one extra function is given, it is used
                as the default value when the otherwise condition is met
            funclist (list of scalars): list of scalar functions.

        Returns:
            cupy.ndarray: the scalar values in funclist on portions of x
                defined by condlist.

        .. warning::

            This function currently doesn't support callable functions,
            args and kw parameters.

        .. seealso:: :func:`numpy.piecewise`
        """
    if cupy.isscalar(condlist):
        condlist = [condlist]

    condlen = len(condlist)
    funclen = len(funclist)
    if condlen == funclen:
        out = cupy.zeros(x.shape, x.dtype)
    elif condlen + 1 == funclen:
        func = funclist[-1]
        funclist = funclist[:-1]
        if callable(func):
            raise NotImplementedError(
                'Callable functions are not supported currently')
        out = cupy.empty(x.shape, x.dtype)
        out[...] = func
    else:
        raise ValueError('with {} condition(s), either {} or {} functions'
                         ' are expected'.format(condlen, condlen, condlen + 1))

    for condition, func in zip(condlist, funclist):
        if callable(func):
            raise NotImplementedError(
                'Callable functions are not supported currently')
        if isinstance(func, cupy.ndarray):
            func = func.astype(x.dtype)
        _piecewise_krnl(condition, func, out)
    return out

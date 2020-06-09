import cupy

from cupy import core

_piecewise_krnl = core.ElementwiseKernel(
    'U condlist, T funclist',
    'T y',
    'if(condlist) y = funclist',
    'piecewise_kernel'
)


def piecewise(x, condlist, funclist):
    """Evaluate a piecewise-defined function.

        Args:
            x (cupy.ndarray): input domain
            condlist (list of cupy.ndarray or bool scalars):
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
    if not isinstance(condlist[0], cupy.ndarray):
        condlist = cupy.asarray(condlist)
    if isinstance(funclist, cupy.ndarray):
        funclist = funclist.tolist()
    condlen = len(condlist)
    funclen = len(funclist)
    if condlen == funclen:
        y = cupy.zeros(shape=x.shape, dtype=x.dtype)
    elif condlen + 1 == funclen:
        if callable(funclist[-1]):
            raise NotImplementedError(
                'Callable functions are not supported currently')
        y = cupy.full(shape=x.shape,
                      fill_value=x.dtype.type(funclist[-1]), dtype=x.dtype)
        funclist = funclist[:-1]
    else:
        raise ValueError('with {} condition(s), either {} or {} functions'
                         ' are expected'.format(condlen, condlen, condlen + 1))
    for condition, func in zip(condlist, funclist):
        if callable(func):
            raise NotImplementedError(
                'Callable functions are not supported currently')
        _piecewise_krnl(condition, func, y)
    return y

import cupy


def piecewise(x, condlist, funclist):
    """Evaluate a piecewise-defined function.

        Args:
            x (cupy.ndarray): input domain
            condlist (list of cupy.ndarray or bool scalars):
                Each boolean array/ scalar corresponds to a function
                in funclist. Length of funclist is equal to that of
                condlist. If one extra function is given, it is used
                as the default value when the otherwise condition is met
            funclist (cupy.ndarray or list): list of scalar functions.

        Returns:
            cupy.ndarray: the scalar values in funclist on portions of x
                defined by condlist.

        .. warning::

            This function currently doesn't support callable functions,
            args and kw parameters.

        .. seealso:: :func:`numpy.piecewise`
        """
    if any(callable(item) for item in funclist):
        raise NotImplementedError(
            'Callable functions are not supported currently')
    diffshape = 0
    if cupy.isscalar(condlist):
        condlist = [condlist]
    elif isinstance(condlist[0], tuple) and x.ndim != 0:
        diffshape = 2
    elif cupy.isscalar(condlist[0]) and x.ndim != 0:
        diffshape = 1
    condlist = cupy.array(condlist, dtype=bool)
    condlen = len(condlist)
    if diffshape:
        if x.ndim == condlist.ndim:
            for i in range(x.ndim - 1, -1, -1):
                if condlist.shape[i] != x.shape[i]:
                    raise ValueError('boolean index did not match indexed '
                                     'array along dimension {}; dimension is '
                                     '{} but corresponding boolean dimension '
                                     'is {}'.format(i, x.shape[i],
                                                    condlist.shape[i]))
        if condlen == x.shape[0]:
            condlen = 1
        else:
            raise ValueError('boolean index did not match indexed array along '
                             'dimension 0; dimension is {} but corresponding '
                             'boolean dimension is {}'
                             .format(x.shape[0], condlen))
    funclist = cupy.asarray(funclist, dtype=x.dtype)
    funclen = len(funclist)
    if condlen == funclen:
        y = cupy.zeros(shape=x.shape, dtype=x.dtype)
    elif condlen + 1 == funclen:  # o.w
        y = cupy.full(shape=x.shape, fill_value=funclist[-1], dtype=x.dtype)
        funclist = funclist[:-1]
    else:
        raise ValueError('with {} condition(s), either {} or {} functions'
                         ' are expected'.format(condlen, condlen, condlen + 1))
    if diffshape == 1:
        cupy._sorting.search._where_ufunc(condlist, funclist[0], y.T, y.T)
    elif diffshape == 2:
        cupy._sorting.search._where_ufunc(condlist, funclist[0], y, y)
    else:
        for condition, func in zip(condlist, funclist):
            cupy._sorting.search._where_ufunc(condition, func, y, y)
    return y

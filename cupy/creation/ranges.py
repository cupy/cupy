import numpy

import cupy
from cupy import core


def arange(start, stop=None, step=1, dtype=None):
    """Returns an array with evenly spaced values within a given interval.

    Values are generated within the half-open interval [start, stop). The first
    three arguments are mapped like the ``range`` built-in function, i.e. start
    and step are optional.

    Args:
        start: Start of the interval.
        stop: End of the interval.
        step: Step width between each pair of consecutive values.
        dtype: Data type specifier. It is inferred from other arguments by
            default.

    Returns:
        cupy.ndarray: The 1-D array of range values.

    .. seealso:: :func:`numpy.arange`

    """
    if dtype is None:
        if any(numpy.dtype(type(val)).kind == 'f'
               for val in (start, stop, step)):
            dtype = float
        else:
            dtype = int

    if stop is None:
        stop = start
        start = 0
    size = int(numpy.ceil((stop - start) / step))
    if size <= 0:
        return cupy.empty((0,), dtype=dtype)

    ret = cupy.empty((size,), dtype=dtype)
    typ = numpy.dtype(dtype).type
    _arange_ufunc(typ(start), typ(step), ret, dtype=dtype)
    return ret


def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None):
    """Returns an array with evenly-spaced values within a given interval.

    Instead of specifying the step width like :func:`cupy.arange`, this
    function requires the total number of elements specified.

    Args:
        start: Start of the interval.
        stop: End of the interval.
        num: Number of elements.
        endpoint (bool): If ``True``, the stop value is included as the last
            element. Otherwise, the stop value is omitted.
        retstep (bool): If ``True``, this function returns (array, step).
            Otherwise, it returns only the array.
        dtype: Data type specifier. It is inferred from the start and stop
            arguments by default.

    Returns:
        cupy.ndarray: The 1-D array of ranged values.

    """
    if num < 0:
        raise ValueError('linspace with num<0 is not supported')

    if dtype is None:
        # In actual implementation, only float is used
        dtype = float

    ret = cupy.empty((num,), dtype=dtype)
    if num == 0:
        step = float('nan')
    elif num == 1:
        ret.fill(start)
        step = float('nan')
    else:
        div = (num - 1) if endpoint else num
        step = float(stop - start) / div
        stop = float(stop)

        if step == 0.0:
            # for underflow
            _linspace_ufunc_underflow(start, stop - start, div, ret,
                                      casting='unsafe')
        else:
            _linspace_ufunc(start, step, ret, casting='unsafe')

        if endpoint:
            ret[-1] = stop

    if retstep:
        return ret, step
    else:
        return ret


def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None):
    """Returns an array with evenly-spaced values on a log-scale.

    Instead of specifying the step width like :func:`cupy.arange`, this
    function requires the total number of elements specified.

    Args:
        start: Start of the interval.
        stop: End of the interval.
        num: Number of elements.
        endpoint (bool): If ``True``, the stop value is included as the last
            element. Otherwise, the stop value is omitted.
        base (float): Base of the log space. The step sizes between the
            elements on a log-scale are the same as ``base``.
        dtype: Data type specifier. It is inferred from the start and stop
            arguments by default.

    Returns:
        cupy.ndarray: The 1-D array of ranged values.

    """
    y = linspace(start, stop, num=num, endpoint=endpoint)
    if dtype is None:
        return core.power(base, y)
    return core.power(base, y).astype(dtype)


def meshgrid(*xi, **kwargs):
    """Return coordinate matrices from coordinate vectors.

    Given one-dimensional coordinate arrays x1, x2, ..., xn, this function
    makes N-D grids.

    For one-dimensional arrays x1, x2, ..., xn with lengths ``Ni = len(xi)``,
    this function returns ``(N1, N2, N3, ..., Nn)`` shaped arrays
    if indexing='ij' or ``(N2, N1, N3, ..., Nn)`` shaped arrays
    if indexing='xy'.

    Unlike NumPy, CuPy currently only supports 1-D arrays as inputs.
    Also, CuPy does not support ``sparse`` option yet.

    Args:
        xi (tuple of ndarrays): 1-D arrays representing the coordinates
            of a grid.
        indexing ({'xy', 'ij'}, optional): Cartesian ('xy', default) or
            matrix ('ij') indexing of output.
        copy (bool, optional): If ``False``, a view
            into the original arrays are returned. Default is True.

    Returns:
        list of cupy.ndarray

    .. seealso:: :func:`numpy.meshgrid`

    """

    indexing = kwargs.pop('indexing', 'xy')
    copy = bool(kwargs.pop('copy', True))
    if kwargs:
        raise TypeError(
            'meshgrid() got an unexpected keyword argument \'{}\''.format(
                list(kwargs)[0]))
    if indexing not in ['xy', 'ij']:
        raise ValueError('Valid values for `indexing` are \'xy\' and \'ij\'.')

    for x in xi:
        if x.ndim != 1:
            raise ValueError('input has to be 1d')
        if not isinstance(x, cupy.ndarray):
            raise ValueError('input has to be cupy.ndarray')
    if len(xi) <= 1:
        return list(xi)

    meshes = []
    for i, x in enumerate(xi):
        if indexing == 'xy' and i == 0:
            left_none = 1
        elif indexing == 'xy' and i == 1:
            left_none = 0
        else:
            left_none = i

        expand_slices = ((None,) * left_none +
                         (slice(None),) +
                         (None,) * (len(xi) - (left_none + 1)))
        meshes.append(x[expand_slices])
    meshes_br = list(cupy.broadcast_arrays(*meshes))

    if copy:
        for i in range(len(meshes_br)):
            meshes_br[i] = meshes_br[i].copy()
    return meshes_br


# mgrid
# ogrid


_arange_ufunc = core.create_ufunc(
    'cupy_arange',
    ('bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
     'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d'),
    'out0 = in0 + i * in1')


_linspace_ufunc = core.create_ufunc(
    'cupy_linspace',
    ('dd->d',),
    'out0 = in0 + i * in1')

_linspace_ufunc_underflow = core.create_ufunc(
    'cupy_linspace',
    ('ddd->d',),
    'out0 = in0 + i * in1 / in2')

import math

import numpy

import cupy
from cupy import _core


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

    if step is None:
        step = 1

    size = int(numpy.ceil((stop - start) / step))
    if size <= 0:
        return cupy.empty((0,), dtype=dtype)

    if numpy.dtype(dtype).type == numpy.bool_:
        if size > 2:
            raise ValueError('no fill-function for data-type.')
        if size == 2:
            return cupy.array([start, start - step], dtype=numpy.bool_)
        else:
            return cupy.array([start], dtype=numpy.bool_)

    ret = cupy.empty((size,), dtype=dtype)
    typ = numpy.dtype(dtype).type
    _arange_ufunc(typ(start), typ(step), ret, dtype=dtype)
    return ret


def _linspace_scalar(start, stop, num=50, endpoint=True, retstep=False,
                     dtype=None):
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
    dt = cupy.result_type(start, stop, float(num))
    if dtype is None:
        # In actual implementation, only float is used
        dtype = dt

    ret = cupy.empty((num,), dtype=dt)
    div = (num - 1) if endpoint else num
    if div <= 0:
        if num > 0:
            ret.fill(start)
        step = float('nan')
    else:
        step = float(stop - start) / div
        stop = float(stop)

        if step == 0.0:
            # for underflow
            _linspace_ufunc_underflow(start, stop - start, div, ret)
        else:
            _linspace_ufunc(start, step, ret)

        if endpoint:
            # Here num == div + 1 > 1 is ensured.
            ret[-1] = stop

    if cupy.issubdtype(dtype, cupy.integer):
        cupy.floor(ret, out=ret)

    ret = ret.astype(dtype, copy=False)

    if retstep:
        return ret, step
    else:
        return ret


def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None,
             axis=0):
    """Returns an array with evenly-spaced values within a given interval.

    Instead of specifying the step width like :func:`cupy.arange`, this
    function requires the total number of elements specified.

    Args:
        start (scalar or array_like): Starting value(s) of the sequence.
        stop (scalar or array_like): Ending value(s) of the sequence, unless
            ``endpoint`` is set to ``False``. In that case, the sequence
            consists of all but the last of ``num + 1`` evenly spaced samples,
            so that ``stop`` is excluded.  Note that the step size changes when
            ``endpoint`` is ``False``.
        num: Number of elements.
        endpoint (bool): If ``True``, the stop value is included as the last
            element. Otherwise, the stop value is omitted.
        retstep (bool): If ``True``, this function returns (array, step).
            Otherwise, it returns only the array.
        dtype: Data type specifier. It is inferred from the start and stop
            arguments by default.
        axis (int):  The axis in the result to store the samples.  Relevant
            only if start or stop are array-like.  By default ``0``, the
            samples will be along a new axis inserted at the beginning.
            Use ``-1`` to get an axis at the end.

    Returns:
        cupy.ndarray: The 1-D array of ranged values.

    .. seealso:: :func:`numpy.linspace`

    """
    if num < 0:
        raise ValueError('linspace with num<0 is not supported')
    div = (num - 1) if endpoint else num

    scalar_start = cupy.isscalar(start)
    scalar_stop = cupy.isscalar(stop)
    if scalar_start and scalar_stop:
        return _linspace_scalar(start, stop, num, endpoint, retstep, dtype)

    if not scalar_start:
        if not (isinstance(start, cupy.ndarray) and start.dtype.kind == 'f'):
            start = cupy.asarray(start) * 1.0

    if not scalar_stop:
        if not (isinstance(stop, cupy.ndarray) and stop.dtype.kind == 'f'):
            stop = cupy.asarray(stop) * 1.0

    dt = cupy.result_type(start, stop, float(num))
    if dtype is None:
        # In actual implementation, only float is used
        dtype = dt

    delta = stop - start

    # ret = cupy.arange(0, num, dtype=dt).reshape((-1,) + (1,) * delta.ndim)
    ret = cupy.empty((num,), dtype=dt)
    _arange_ufunc(0.0, 1.0, ret, dtype=dt)
    ret = ret.reshape((-1,) + (1,) * delta.ndim)

    # In-place multiplication y *= delta/div is faster, but prevents the
    # multiplicant from overriding what class is produced, and thus prevents,
    # e.g. use of Quantities, see numpy#7142. Hence, we multiply in place only
    # for standard scalar types.
    if num > 1:
        step = delta / div
        if cupy.any(step == 0):
            # Special handling for denormal numbers, numpy#5437
            ret /= div
            ret = ret * delta
        else:
            ret = ret * step
    else:
        # 0 and 1 item long sequences have an undefined step
        step = float('nan')
        # Multiply with delta to allow possible override of output class.
        ret = ret * delta

    ret += start
    if endpoint and num > 1:
        ret[-1] = stop

    if axis != 0:
        ret = cupy.moveaxis(ret, 0, axis)

    if cupy.issubdtype(dtype, cupy.integer):
        cupy.floor(ret, out=ret)

    ret = ret.astype(dtype, copy=False)

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

    .. seealso:: :func:`numpy.logspace`

    """
    y = linspace(start, stop, num=num, endpoint=endpoint)
    if dtype is None:
        return _core.power(base, y)
    return _core.power(base, y).astype(dtype)


def meshgrid(*xi, **kwargs):
    """Return coordinate matrices from coordinate vectors.

    Given one-dimensional coordinate arrays ``x1, x2, ... , xn`` this
    function makes N-D grids.

    For one-dimensional arrays ``x1, x2, ... , xn`` with lengths
    ``Ni = len(xi)``, this function returns ``(N1, N2, N3, ..., Nn)`` shaped
    arrays if indexing='ij' or ``(N2, N1, N3, ..., Nn)`` shaped arrays
    if indexing='xy'.

    Unlike NumPy, CuPy currently only supports 1-D arrays as inputs.

    Args:
        xi (tuple of ndarrays): 1-D arrays representing the coordinates
            of a grid.
        indexing ({'xy', 'ij'}, optional): Cartesian ('xy', default) or
            matrix ('ij') indexing of output.
        sparse (bool, optional): If ``True``, a sparse grid is returned in
            order to conserve memory. Default is ``False``.
        copy (bool, optional): If ``False``, a view
            into the original arrays are returned. Default is ``True``.

    Returns:
        list of cupy.ndarray

    .. seealso:: :func:`numpy.meshgrid`

    """
    indexing = kwargs.pop('indexing', 'xy')
    copy = bool(kwargs.pop('copy', True))
    sparse = bool(kwargs.pop('sparse', False))
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

    if sparse:
        meshes_br = meshes
    else:
        meshes_br = list(cupy.broadcast_arrays(*meshes))

    if copy:
        for i in range(len(meshes_br)):
            meshes_br[i] = meshes_br[i].copy()
    return meshes_br


class nd_grid(object):
    """Construct a multi-dimensional "meshgrid".

    ``grid = nd_grid()`` creates an instance which will return a mesh-grid
    when indexed.  The dimension and number of the output arrays are equal
    to the number of indexing dimensions.  If the step length is not a
    complex number, then the stop is not inclusive.

    However, if the step length is a **complex number** (e.g. 5j), then the
    integer part of its magnitude is interpreted as specifying the
    number of points to create between the start and stop values, where
    the stop value **is inclusive**.

    If instantiated with an argument of ``sparse=True``, the mesh-grid is
    open (or not fleshed out) so that only one-dimension of each returned
    argument is greater than 1.

    Args:
        sparse (bool, optional): Whether the grid is sparse or not.
            Default is False.

    .. seealso:: :data:`numpy.mgrid` and :data:`numpy.ogrid`

    """

    def __init__(self, sparse=False):
        self.sparse = sparse

    def __getitem__(self, key):
        if isinstance(key, slice):
            step = key.step
            stop = key.stop
            start = key.start
            if start is None:
                start = 0
            if isinstance(step, complex):
                step = abs(step)
                length = int(step)
                if step != 1:
                    step = (key.stop - start) / float(step - 1)
                stop = key.stop + step
                return cupy.arange(0, length, 1, float) * step + start
            else:
                return cupy.arange(start, stop, step)

        size = []
        typ = int
        for k in range(len(key)):
            step = key[k].step
            start = key[k].start
            if start is None:
                start = 0
            if step is None:
                step = 1
            if isinstance(step, complex):
                size.append(int(abs(step)))
                typ = float
            else:
                size.append(
                    int(math.ceil((key[k].stop - start) / (step * 1.0))))
            if (isinstance(step, float) or
                    isinstance(start, float) or
                    isinstance(key[k].stop, float)):
                typ = float
        if self.sparse:
            nn = [cupy.arange(_x, dtype=_t)
                  for _x, _t in zip(size, (typ,) * len(size))]
        else:
            nn = cupy.indices(size, typ)
        for k in range(len(size)):
            step = key[k].step
            start = key[k].start
            if start is None:
                start = 0
            if step is None:
                step = 1
            if isinstance(step, complex):
                step = int(abs(step))
                if step != 1:
                    step = (key[k].stop - start) / float(step - 1)
            nn[k] = (nn[k] * step + start)
        if self.sparse:
            slobj = [cupy.newaxis] * len(size)
            for k in range(len(size)):
                slobj[k] = slice(None, None)
                nn[k] = nn[k][tuple(slobj)]
                slobj[k] = cupy.newaxis
        return nn

    def __len__(self):
        return 0


mgrid = nd_grid(sparse=False)
ogrid = nd_grid(sparse=True)


_arange_ufunc = _core.create_ufunc(
    'cupy_arange',
    ('bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
     'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d',
     ('FF->F', 'out0 = in0 + float(i) * in1'),
     ('DD->D', 'out0 = in0 + double(i) * in1')),
    'out0 = in0 + i * in1')

_linspace_ufunc = _core.create_ufunc(
    'cupy_linspace',
    ('dd->d',),
    'out0 = in0 + i * in1')

_linspace_ufunc_underflow = _core.create_ufunc(
    'cupy_linspace',
    ('ddd->d',),
    'out0 = in0 + i * in1 / in2')

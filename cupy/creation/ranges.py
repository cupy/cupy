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


# TODO(okuta): Implement logspace


# TODO(okuta): Implement meshgrid


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

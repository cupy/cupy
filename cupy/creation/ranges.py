import numpy

import cupy
from cupy import elementwise


def arange(start, stop=None, step=1, dtype=None):
    """Rerurns an array with evenly spaced values within a given interval.

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
        endpoint (bool): If True, the stop value is included as the last
            element. Otherwise, the stop value is omitted.
        retstep (bool): If True, this function returns (array, step).
            Otherwise, it returns only the array.
        dtype: Data type specifier. It is inferred from the start and stop
            arguments by default.

    Returns:
        cupy.ndarray: The 1-D array of ranged values.

    """
    if num <= 0:
        # TODO(beam2d): Return zero-sized array
        raise ValueError('linspace with num<=0 is not supported')

    if dtype is None:
        if any(numpy.dtype(type(val)).kind == 'f' for val in (start, stop)):
            dtype = float
        else:
            dtype = int

    ret = cupy.empty((num,), dtype=dtype)
    if num == 0:
        return ret
    elif num == 1:
        ret.fill(start)
        return ret

    if endpoint:
        step = (stop - start) / (num - 1)
    else:
        step = (stop - start) / num
        stop = start + step * (num - 1)

    typ = numpy.dtype(dtype).type
    _linspace_ufunc(typ(start), stop - start, num - 1, ret)
    if retstep:
        return ret, step
    else:
        return ret


# TODO(okuta): Implement logspace


# TODO(okuta): Implement meshgrid


# mgrid
# ogrid


_arange_ufunc = elementwise.create_ufunc(
    'cupy_arange',
    ('bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
     'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d'),
    'out0 = in0 + i * in1')


_float_linspace = 'out0 = in0 + i * in1 / in2'
_linspace_ufunc = elementwise.create_ufunc(
    'cupy_linspace',
    ('bbb->b', 'Bbb->B', 'hhh->h', 'Hhh->H', 'iii->i', 'Iii->I', 'lll->l',
     'Lll->L', 'qqq->q', 'Qqq->Q', ('eel->e', _float_linspace),
     ('ffl->f', _float_linspace), ('ddl->d', _float_linspace)),
    'out0 = (in0_type)(in0 + _floor_divide(in1_type(i * in1), in2))')

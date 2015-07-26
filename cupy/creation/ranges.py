import numpy

import cupy
from cupy import cuda
from cupy import elementwise


def arange(start, stop=None, step=1, dtype=numpy.int_, allocator=cuda.alloc):
    if stop is None:
        stop = start
        start = 0
    size = int(numpy.ceil((stop - start) / step))
    if size <= 0:
        return cupy.empty((0,), dtype=dtype, allocator=allocator)

    ret = cupy.empty((size,), dtype=dtype, allocator=allocator)
    typ = numpy.dtype(dtype).type
    _arange_ufunc(typ(start), typ(step), ret, dtype=dtype)
    return ret


def linspace(start, stop, num=50, endpoint=True, retstep=False,
             dtype=numpy.int_, allocator=cuda.alloc):
    if num <= 0:
        # TODO(beam2d): Return zero-sized array
        raise ValueError('linspace with num<=0 is not supported')

    ret = cupy.empty((num,), dtype=dtype, allocator=allocator)
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
    _linspace_ufunc(typ(start), typ(stop - start), num - 1, ret)
    if retstep:
        return ret, step
    else:
        return ret


def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None,
             allocator=None):
    # TODO(beam2d): Implement these
    raise NotImplementedError


def meshgrid(*xi, **kwargs):
    # TODO(beam2d): Implement these
    raise NotImplementedError


# TODO(beam2d): Implement these
# mgrid
# ogrid


_arange_ufunc = elementwise.create_ufunc(
    'cupy_arange',
    ['bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
     'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d'],
    'out0 = in0 + i * in1')


_float_linspace = 'out0 = in0 + i * in1 / in2'
_linspace_ufunc = elementwise.create_ufunc(
    'cupy_linspace',
    ['bbb->b', 'BBb->B', 'hhh->h', 'HHh->H', 'iii->i', 'IIi->I', 'lll->l',
     'LLl->L', 'qqq->q', 'QQq->Q', ('eel->e', _float_linspace),
     ('ffl->f', _float_linspace), ('ddl->d', _float_linspace)],
    'out0 = (in0_type)(in0 + _floor_divide((in2_type)(i * in1), in2))')

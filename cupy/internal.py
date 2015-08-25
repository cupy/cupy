import ctypes

import numpy
import six

import cupy
from cupy import cuda


def prod(args, init=1):
    for arg in args:
        init *= arg
    return init


def get_reduced_dims(shape, strides, itemsize):
    if not shape:
        return (), ()
    if 0 in shape:
        return (0,), (itemsize,)

    if len(shape) == 1:
        return tuple(shape), tuple(strides)
    if len(shape) == 2:
        if shape[0] == 1 or strides[0] == shape[1] * strides[1]:
            return (shape[0] * shape[1],), (strides[1],)
        else:
            return tuple(shape), tuple(strides)

    reduced_shape = [shape[0]]
    reduced_strides = [strides[0]]
    for sh, st, prev_st in six.moves.zip(shape[1:], strides[1:], strides):
        if reduced_shape[-1] == 1 or prev_st == sh * st:
            reduced_shape[-1] *= sh
            reduced_strides[-1] = st
        else:
            reduced_shape.append(sh)
            reduced_strides.append(st)

    return tuple(reduced_shape), tuple(reduced_strides)


def get_reduced_dims_from_array(a):
    return get_reduced_dims(a.shape, a.strides, a.itemsize)


def get_strides_for_nocopy_reshape(a, new_shape):
    if a.size != prod(new_shape):
        return None
    a_shape = a.shape
    a_strides = a.strides
    a_itemsize = a.itemsize
    if len(a_shape) == 0:
        return (a_itemsize,) * len(new_shape)

    shape, strides = get_reduced_dims(a_shape, a_strides, a_itemsize)

    ndim = len(shape)
    dim = 0
    sh = shape[0]
    st = strides[0]
    last_stride = sh * st
    new_strides = []
    for size in new_shape:
        if size > 1:
            if sh == 1:
                dim += 1
                if dim >= ndim:
                    return None
                sh = shape[dim]
                st = strides[dim]
            if sh % size != 0:
                return None
            sh //= size
            last_stride = sh * st
        new_strides.append(last_stride)

    return tuple(new_strides)


def get_contiguous_strides(shape, itemsize):
    strides = [itemsize] * len(shape)
    for i in six.moves.range(len(strides) - 1, 0, -1):
        strides[i - 1] = strides[i] * max(1, shape[i])
    return tuple(strides)


def get_ndarray_ptr(a_cpu):
    if a_cpu.dtype.type == numpy.bool_:
        # Boolean array cannot be directly converted to ctypes
        a_cpu = a_cpu.view(dtype=numpy.uint8)
    elif a_cpu.dtype.type == numpy.float16:
        # Float16 array cannot be directly converted to ctypes
        a_cpu = a_cpu.view(dtype=numpy.uint16)
    if a_cpu.shape:
        return ctypes.cast(numpy.ctypeslib.as_ctypes(a_cpu), ctypes.c_void_p)
    else:
        return ctypes.cast(ctypes.pointer(numpy.ctypeslib.as_ctypes(a_cpu)),
                           ctypes.c_void_p)


def complete_slice(slc, dim):
    step = 1 if slc.step is None else slc.step
    if step == 0:
        raise ValueError('Slice step must be nonzero.')
    elif step > 0:
        start = 0 if slc.start is None else max(0, min(dim, slc.start))
        stop = dim if slc.stop is None else max(start, min(dim, slc.stop))
    else:
        start = dim - 1 if slc.start is None else max(0, min(dim, slc.start))
        stop = -1 if slc.stop is None else max(0, min(start, slc.stop))
    return slice(start, stop, step)


def get_c_contiguity(shape, strides, itemsize):
    if 0 in shape:
        return True
    _, strides = get_reduced_dims(shape, strides, itemsize)
    return len(strides) == 0 or (len(strides) == 1 and strides[0] == itemsize)


def infer_unknown_dimension(shape, size):
    cnt = 0
    for dim in shape:
        cnt += dim < 0
    if cnt == 0:
        return shape
    if cnt > 1:
        raise ValueError('can only specify only one unknown dimension')
    p = size // prod(dim for dim in shape if dim >= 0)
    return tuple([dim if dim >= 0 else p for dim in shape])


def check_args_device(args):
    dev = cuda.Device()
    for arg in args:
        if isinstance(arg, cupy.ndarray):
            arg_dev = arg.data.device
            if arg_dev == dev:
                continue
            raise ValueError('Array device must be same as the current '
                             'device: array device = %d while current = %d'
                             % (arg_dev.id, dev.id))

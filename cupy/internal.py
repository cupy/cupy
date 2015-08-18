import ctypes
import operator

import numpy
import six

import cupy
from cupy import cuda


def prod(args, init=1):
    return reduce(operator.mul, args, init)


def get_reduced_dims(shape, strides, itemsize):
    if not shape:
        return (), ()
    elif 0 in shape:
        return (0,), (itemsize,)
    reduced_shape = [shape[0]]
    reduced_strides = [strides[0]]
    for i in six.moves.range(1, len(shape)):
        if strides[i - 1] == shape[i] * strides[i] or \
           reduced_shape[-1] == 1:
            reduced_shape[-1] *= shape[i]
            reduced_strides[-1] = strides[i]
        else:
            reduced_shape.append(shape[i])
            reduced_strides.append(strides[i])

    return tuple(reduced_shape), tuple(reduced_strides)


def get_reduced_dims_from_array(a):
    return get_reduced_dims(a.shape, a.strides, a.itemsize)


def get_strides_for_nocopy_reshape(array, new_shape):
    shape, strides = map(list, get_reduced_dims_from_array(array))
    new_strides = []

    dim = 0
    ndim = len(shape)
    if len(array.shape) == 0:
        last_stride = array.itemsize
    else:
        last_stride = array.strides[0] * array.shape[0]
    for size in new_shape:
        if size <= 1:
            new_strides.append(last_stride)
            continue
        if dim >= ndim or shape[dim] % size != 0:
            return None
        shape[dim] //= size
        last_stride = shape[dim] * strides[dim]
        new_strides.append(last_stride)
        if shape[dim] == 1:
            dim = dim + 1

    return tuple(new_strides)


def get_contiguous_strides(shape, itemsize):
    if not shape:
        return ()
    else:
        s = numpy.array(shape[1:])
        strides = numpy.maximum(1, s[::-1]).cumprod()[::-1] * itemsize
        return tuple(strides) + (itemsize,)


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
    if sum(dim < 0 for dim in shape) > 1:
        raise ValueError('can only specify only one unknown dimension')

    shape = tuple(dim if dim >= 0 else -1 for dim in shape)
    p = prod(shape)
    if p < 0:
        return tuple(dim if dim >= 0 else size // -p for dim in shape)
    else:
        return shape


def check_args_device(args):
    dev = cuda.Device()
    for arg in args:
        if isinstance(arg, cupy.ndarray):
            arg_dev = arg.data.device
            if arg_dev != dev:
                raise ValueError('Array device must be same as the current '
                                 'device: array device = %d while current = %d'
                                 % (arg_dev.id, dev.id))

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
        return shape, strides
    if len(shape) == 2:
        shape0, shape1 = shape
        strides0, strides1 = strides
        if shape0 == 1 or strides0 == shape1 * strides1:
            return (shape0 * shape1,), (strides1,)
        else:
            return shape, strides

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
    a_size = a.size
    if a_size != prod(new_shape):
        return None
    a_shape = a.shape
    a_strides = a.strides
    a_itemsize = a.itemsize
    if a_size == 1:
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
    ndim = len(shape)
    if ndim == 0:
        return ()
    if ndim == 1:
        return itemsize,
    if ndim == 2:
        return shape[1] * itemsize, itemsize

    strides = [0] * ndim
    st = itemsize
    for i in six.moves.range(ndim - 1, -1, -1):
        strides[i] = st
        sh = shape[i]
        if sh > 1:
            st *= sh
    return tuple(strides)


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
    ndim = len(strides)
    return ndim == 0 or (ndim == 1 and strides[0] == itemsize)


def infer_unknown_dimension(shape, size):
    cnt = 0
    for dim in shape:
        cnt += dim < 0
    if cnt == 0:
        return shape
    if cnt > 1:
        raise ValueError('can only specify only one unknown dimension')
    p = size
    for dim in shape:
        if dim > 0:
            p //= dim
    return tuple([dim if dim >= 0 else p for dim in shape])


def check_args_device(args):
    dev = cuda.Device()
    for arg in args:
        if isinstance(arg, cupy.ndarray) and not arg.data.device == dev:
            raise ValueError('Array device must be same as the current '
                             'device: array device = %d while current = %d'
                             % (arg.data.device.id, dev.id))

import numpy

import cupy


def atleast_1d(*arys):
    res = []
    for a in arys:
        if not isinstance(a, cupy.ndarray):
            raise TypeError('Only cupy arrays can be atleast_1d')
        if a.ndim == 0:
            a = a.reshape(1)
        res.append(a)
    if len(res) == 1:
        res = res[0]
    return res


def atleast_2d(*arys):
    res = []
    for a in arys:
        if not isinstance(a, cupy.ndarray):
            raise TypeError('Only cupy arrays can be atleast_2d')
        if a.ndim == 0:
            a = a.reshape(1, 1)
        elif a.ndim == 1:
            a = a[cupy.newaxis, :]
        res.append(a)
    if len(res) == 1:
        res = res[0]
    return res


def atleast_3d(*arys):
    res = []
    for a in arys:
        if not isinstance(a, cupy.ndarray):
            raise TypeError('Only cupy arrays can be atleast_3d')
        if a.ndim == 0:
            a = a.reshape(1, 1, 1)
        elif a.ndim == 1:
            a = a[cupy.newaxis, :, cupy.newaxis]
        elif a.ndim == 2:
            a = a[:, :, cupy.newaxis]
        res.append(a)
    if len(res) == 1:
        res = res[0]
    return res


class Broadcast(object):

    def __init__(self, *arrays):
        ndim = 0
        for array in arrays:
            if isinstance(array, cupy.ndarray):
                ndim = max(ndim, array.ndim)

        shape = [1] * ndim
        for array in arrays:
            if isinstance(array, cupy.ndarray):
                offset = len(shape) - array.ndim
                for i, dim in enumerate(array.shape):
                    if dim != 1 and shape[i + offset] != dim:
                        if shape[i + offset] != 1:
                            raise RuntimeError('Broadcasting failed')
                        else:
                            shape[i + offset] = dim

        self.shape = tuple(shape)
        self.size = numpy.prod(self.shape, dtype=int)
        self.nd = len(shape)

        broadcasted = []
        for array in arrays:
            if not isinstance(array, cupy.ndarray):
                broadcasted.append(array)
                continue
            if array.shape == self.shape:
                broadcasted.append(array)
                continue

            offset = self.nd - array.ndim
            strides = []
            for i, dim in enumerate(shape):
                if i < offset:
                    # TODO(okuta) fix if `dim` == 1
                    strides.append(0)
                elif array.shape[i - offset] != dim:
                    strides.append(0)
                else:
                    strides.append(array._strides[i - offset])

            view = array.view()
            view._shape = self.shape
            view._strides = tuple(strides)
            view._update_contiguity()
            broadcasted.append(view)

        self.values = broadcasted


def broadcast(*arrays):
    # It does not support multi-iterator
    return Broadcast(*arrays)


def broadcast_arrays(*args):
    return Broadcast(*args).values


def expand_dims(a, axis):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def squeeze(a, axis=None):
    if axis is None:
        axis = tuple(i for i, n in enumerate(a._shape) if n == 1)
    elif isinstance(axis, int):
        axis = axis,

    new_shape = []
    new_strides = []
    j = 0
    for i, n in enumerate(a._shape):
        if j < len(axis) and i == axis[j]:
            if n != 1:
                raise RuntimeError('Cannot squeeze dimension of size > 1')
            j += 1
        else:
            new_shape.append(n)
            new_strides.append(a._strides[i])

    v = a.view()
    v._shape = tuple(new_shape)
    v._strides = tuple(new_strides)
    a._update_contiguity()
    return v

from chainer import cuda


def pad(array, pad_width, mode=None, **kwargs):
    if mode != 'constant':
        raise NotImplementedError
    constant_value = kwargs.pop('constant_values', None)
    if len(kwargs) != 0:
        raise NotImplementedError
    if constant_value is None:
        constant_value = 0
    if not isinstance(constant_value, int):
        raise NotImplementedError
    if isinstance(pad_width, int):
        shape = tuple([s + pad_width * 2 for s in array.shape])
        ret = cuda.cupy.full(shape, constant_value, dtype=array.dtype)
        index = tuple([slice(pad_width, pad_width + s) for s in array.shape])
        ret[index] = array
    elif isinstance(pad_width, (list, tuple, cuda.cupy.ndarray)):
        if array.ndim == len(pad_width):
            shape = tuple([s + pad_width[i] * 2
                          for i, s in enumerate(array.shape)])
            ret = cuda.cupy.full(shape, constant_value,
                            dtype=array.dtype)
            index = tuple([slice(pad_width[i], pad_width[i] + s)
                          for i, s in enumerate(array.shape)])
            ret[index] = array
        elif array.ndim == 1 and len(pad_width) == 2:
            front = cuda.cupy.zeros(pad_width[0]) + constant_value
            end = cuda.cupy.zeros(pad_width[1]) + constant_value
            ret = cuda.cupy.hstack((front, array, end))
        else:
            fmt = 'Unable to create correctly shaped tuple from %s'
            raise ValueError(fmt % (shape,))
    else:
        raise TypeError('`pad_width` must be of integral type.')
    return ret

import cupy


def pad(array, pad_width, mode=None, **kwargs):
    if mode != 'constant':
        raise NotImplementedError

    if kwargs.keys() == []:
        fill_value = 0
    elif kwargs.keys() == ['constant_values']:
        if not isinstance(kwargs['constant_values'], int):
            raise NotImplementedError
        fill_value = kwargs['constant_values']
    else:
        raise NotImplementedError

    if isinstance(pad_width, int):
        shape = tuple([s + pad_width * 2 for s in array.shape])
        ret = cupy.full(shape, fill_value, dtype=array.dtype)
        index = tuple([slice(pad_width, pad_width + s) for s in array.shape])
        ret[index] = array
    elif isinstance(pad_width, (list, tuple, cupy.ndarray)):
        if array.ndim == len(pad_width):
            shape = tuple([s + pad_width[i] * 2
                          for i, s in enumerate(array.shape)])
            ret = cupy.full(shape, fill_value,
                            dtype=array.dtype)
            index = tuple([slice(pad_width[i], pad_width[i] + s)
                          for i, s in enumerate(array.shape)])
            ret[index] = array
        elif array.ndim == 1 and len(pad_width) == 2:
            front = cupy.zeros(pad_width[0])  # fix
            end = cupy.zeros(pad_width[1])  # fix
            ret = cupy.hstack((cupy.hstack((front, array)), end))
        else:
            fmt = 'Unable to create correctly shaped tuple from %s'
            raise ValueError(fmt % (shape,))
    else:
        raise TypeError('`pad_width` must be of integral type.')
    return ret

import cupy


def pad(array, pad_width, mode=None, **kwargs):
    if not isinstance(array, (list, tuple, cupy.ndarray)):
        return cupy.array(array)
    if isinstance(array, (list, tuple)):
        array = cupy.array(array)
    if mode != 'constant':
        raise NotImplementedError
    constant_value = kwargs.pop('constant_values', None)
    if len(kwargs) != 0:
        raise NotImplementedError
    if constant_value is None:
        constant_value = 0
    if not isinstance(constant_value, int):
        raise NotImplementedError
    if not cupy.asarray(pad_width).dtype.kind == 'i':
        raise TypeError('`pad_width` must be of integral type.')
    if cupy.asarray(pad_width).ndim > 1:
        raise NotImplementedError
    if isinstance(pad_width, int):
        pad_width_seq = [pad_width, pad_width]
    else:
        if len(pad_width) == 1:
            pad_width_seq = [pad_width[0], pad_width[0]]
        elif len(pad_width) != 2:
            fmt = 'Unable to create correctly shaped tuple'
            raise ValueError(fmt)
        else:
            pad_width_seq = pad_width
    shape = tuple([s + pad_width_seq[0] + pad_width_seq[1]
                  for s in array.shape])
    ret = cupy.full(shape, constant_value, dtype=array.dtype)
    index = tuple([slice(pad_width_seq[0], pad_width_seq[0] + s)
                  for s in array.shape])
    ret[index] = array
    return ret

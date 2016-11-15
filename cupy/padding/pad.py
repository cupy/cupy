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
    if not isinstance(pad_width, (int, list, tuple, cuda.cupy.ndarray)):
        raise TypeError('`pad_width` must be of integral type.')
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
    ret = cuda.cupy.full(shape, constant_value, dtype=array.dtype)
    index = tuple([slice(pad_width_seq[0], pad_width_seq[0] + s)
                  for s in array.shape])
    ret[index] = array
    return ret

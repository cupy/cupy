import cupy


def pad(array, pad_width, mode=None, **kwargs):
    if not isinstance(pad_width, int):
        raise NotImplementedError

    if mode != 'constant':
        raise NotImplementedError

    shape = tuple(s + pad_width * 2 for s in array.shape)
    # TODO(unno): Fix fill value
    ret = cupy.full(shape, 0, dtype=array.dtype)
    index = tuple(slice(pad_width, pad_width + s) for s in array.shape)
    ret[index] = array
    return ret

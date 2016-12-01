import cupy
import six


def _prepend_const(arr, pad_amt, val, axis=-1):
    if pad_amt == 0:
        return arr
    padshape = tuple(x if i != axis else pad_amt
                     for (i, x) in enumerate(arr.shape))
    if val == 0:
        return cupy.concatenate((cupy.zeros(padshape, dtype=arr.dtype), arr),
                                axis=axis)
    else:
        return cupy.concatenate(((cupy.zeros(padshape) + val).astype(
                                 arr.dtype), arr), axis=axis)


def _append_const(arr, pad_amt, val, axis=-1):
    if pad_amt == 0:
        return arr
    padshape = tuple(x if i != axis else pad_amt
                     for (i, x) in enumerate(arr.shape))
    if val == 0:
        return cupy.concatenate((arr, cupy.zeros(padshape, dtype=arr.dtype)),
                                axis=axis)
    else:
        return cupy.concatenate(
            (arr, (cupy.zeros(padshape) + val).astype(arr.dtype)), axis=axis)


def _normalize_shape(ndarray, shape, cast_to_int=True):
    ndims = ndarray.ndim
    if shape is None:
        return ((None, None), ) * ndims
    arr = cupy.asarray(shape)
    if arr.ndim <= 1:
        if arr.shape == () or arr.shape == (1,):
            arr = cupy.ones((ndims, 2), dtype=ndarray.dtype) * arr
        elif arr.shape == (2,):
            arr = arr[cupy.newaxis, :].repeat(ndims, axis=0)
        else:
            fmt = "Unable to create correctly shaped tuple from %s"
            raise ValueError(fmt % (shape,))
    elif arr.ndim == 2:
        if arr.shape[1] == 1 and arr.shape[0] == ndims:
            arr = arr.repeat(2, axis=1)
        elif arr.shape[0] == ndims:
            pass
        else:
            fmt = "Unable to create correctly shaped tuple from %s"
            raise ValueError(fmt % (shape,))
    else:
        fmt = "Unable to create correctly shaped tuple from %s"
        raise ValueError(fmt % (shape,))
    if cast_to_int is True:
        arr = cupy.rint(arr).astype(int)
    return tuple(tuple(axis) for axis in arr.tolist())


def _validate_lengths(narray, number_elements):
    normshp = _normalize_shape(narray, number_elements)
    for i in normshp:
        chk = [1 if x is None else x for x in i]
        chk = [1 if x >= 0 else -1 for x in chk]
        if (chk[0] < 0) or (chk[1] < 0):
            fmt = "%s cannot contain negative values."
            raise ValueError(fmt % (number_elements,))
    return normshp


def pad(array, pad_width, mode, **kwargs):
    if not cupy.asarray(pad_width).dtype.kind == 'i':
        raise TypeError('pad_width must be of integral type.')
    narray = cupy.array(array)
    pad_width = _validate_lengths(narray, pad_width)
    allowedkwargs = {
        'constant': ['constant_values'],
    }
    kwdefaults = {
        'constant_values': 0,
    }
    if mode != 'constant':
        raise NotImplementedError
    for key in kwargs:
        if key not in allowedkwargs[mode]:
            raise ValueError('%s keyword not in allowed keywords %s' %
                             (key, allowedkwargs[mode]))
    for kw in allowedkwargs[mode]:
        kwargs.setdefault(kw, kwdefaults[kw])
    for i in kwargs:
        if i in ['constant_values']:
            kwargs[i] = _normalize_shape(narray, kwargs[i], cast_to_int=False)
    newmat = narray.copy()
    for axis, ((pad_before, pad_after), (before_val, after_val)) \
            in enumerate(six.moves.zip(pad_width, kwargs['constant_values'])):
        newmat = _prepend_const(newmat, pad_before, before_val, axis)
        newmat = _append_const(newmat, pad_after, after_val, axis)
    return newmat

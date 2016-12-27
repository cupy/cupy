import numpy

import cupy
import six


def _prepend_const(narray, pad_amount, value, axis=-1):
    if pad_amount == 0:
        return narray
    padshape = tuple(x if i != axis else pad_amount
                     for i, x in enumerate(narray.shape))
    return cupy.concatenate((cupy.full(padshape, value, narray.dtype),
                             narray), axis=axis)


def _append_const(narray, pad_amount, value, axis=-1):
    if pad_amount == 0:
        return narray
    padshape = tuple(x if i != axis else pad_amount
                     for i, x in enumerate(narray.shape))
    return cupy.concatenate((narray,
                             (cupy.full(padshape, value, narray.dtype))),
                            axis=axis)


def _normalize_shape(ndarray, shape, cast_to_int=True):
    ndims = ndarray.ndim
    if shape is None:
        return ((None, None), ) * ndims
    ndshape = numpy.asarray(shape)
    extend_shape = numpy.repeat(ndshape, 2)
    if len(extend_shape) == 2:
        ndshape = extend_shape
    if ndshape.ndim == 1:
        ndshape = numpy.tile(ndshape, (ndims, 1))
    if ndshape.shape != (ndims, 2):
        message = 'Unable to create correctly shaped tuple from %s' % shape
        raise ValueError(message)
    if cast_to_int:
        ndshape = numpy.rint(ndshape).astype(int)
    return tuple(tuple(axis) for axis in ndshape)


def _validate_lengths(narray, number_elements):
    shape = _normalize_shape(narray, number_elements)
    for axis_shape in shape:
        axis_shape = [1 if x is None else x for x in axis_shape]
        axis_shape = [1 if x >= 0 else -1 for x in axis_shape]
        if axis_shape[0] < 0 or axis_shape[1] < 0:
            message = '%s cannot contain negative values.' % number_elements
            raise ValueError(message)
    return shape


def pad(array, pad_width, mode, **keywords):
    """Returns padded array. You can specify the padded widths and values.

    This function currently supports only ``mode=constant`` .

    Args:
        array (array-like): Input array of rank N.
        pad_width (int or array-like): Number of values padded
            to the edges of each axis.
            ((before_1, after_1), ... (before_N, after_N)) uniquely pad widths
            for each axis.
            ((before, after),) yields same before and after pad for each axis.
            (pad,) or int is a shortcut for before = after = pad width for all
            axes.
            You cannot specify ``cupy.ndarray`` .
        mode (str):
            'constant'
                Pads with a constant values.
        constant_values (int or array-like): Used in
            ``constant``.
            The values are padded for each axis.
            ((before_1, after_1), ... (before_N, after_N)) uniquely pad
            constants for each axis.
            ((before, after),) yields same before and after constants for each
            axis.
            (constant,) or int is a shortcut for before = after = constant for
            all axes.
            Default is 0. You cannot specify ``cupy.ndarray`` .

    Returns:
        cupy.ndarray:
        Padded array of rank equal to ``array`` with shape increased according
        to ``pad_width`` .

    .. seealso:: :func:`numpy.pad`

    """
    if not numpy.asarray(pad_width).dtype.kind == 'i':
        raise TypeError('pad_width must be of integral type.')
    narray = cupy.array(array)
    pad_width = _validate_lengths(narray, pad_width)
    allowed_keywords = {
        'constant': ['constant_values'],
    }
    keyword_defaults = {
        'constant_values': 0,
    }
    if mode != 'constant':
        raise NotImplementedError
    for key in keywords:
        if key not in allowed_keywords[mode]:
            raise ValueError('%s keyword not in allowed keywords %s' %
                             (key, allowed_keywords[mode]))
    for allowed_keyword in allowed_keywords[mode]:
        keywords.setdefault(allowed_keyword, keyword_defaults[allowed_keyword])
    for key in keywords:
        if key == 'constant_values':
            keywords[key] = _normalize_shape(narray, keywords[key],
                                             cast_to_int=False)
    newmatrix = narray.copy()
    for axis, ((pad_before, pad_after), (before_value, after_value)) \
            in enumerate(six.moves.zip(pad_width,
                                       keywords['constant_values'])):
        newmatrix = _prepend_const(newmatrix, pad_before, before_value, axis)
        newmatrix = _append_const(newmatrix, pad_after, after_value, axis)
    return newmatrix

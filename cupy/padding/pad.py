import numpy
import six

import cupy


def _prepend_const(narray, pad_amount, value, axis=-1):
    if pad_amount == 0:
        return narray
    padshape = tuple([x if i != axis else pad_amount
                      for i, x in enumerate(narray.shape)])
    return cupy.concatenate((cupy.full(padshape, value, narray.dtype),
                             narray), axis=axis)


def _append_const(narray, pad_amount, value, axis=-1):
    if pad_amount == 0:
        return narray
    padshape = tuple([x if i != axis else pad_amount
                      for i, x in enumerate(narray.shape)])
    return cupy.concatenate((narray,
                             cupy.full(padshape, value, narray.dtype)),
                            axis=axis)


def _prepend_edge(arr, pad_amt, axis=-1):
    """Prepend `pad_amt` to `arr` along `axis` by extending edge values.

    Parameters
    ----------
    arr : ndarray
        Input array of arbitrary shape.
    pad_amt : int
        Amount of padding to prepend.
    axis : int
        Axis along which to pad `arr`.

    Returns
    -------
    padarr : ndarray
        Output array, extended by `pad_amt` edge values appended along `axis`.

    """
    if pad_amt == 0:
        return arr

    edge_slice = tuple([slice(None) if i != axis else 0
                        for i, x in enumerate(arr.shape)])

    # Shape to restore singleton dimension after slicing
    pad_singleton = tuple([x if i != axis else 1
                           for i, x in enumerate(arr.shape)])
    edge_arr = arr[edge_slice].reshape(pad_singleton)
    return cupy.concatenate((edge_arr.repeat(pad_amt, axis=axis), arr),
                            axis=axis)


def _append_edge(arr, pad_amt, axis=-1):
    """Append `pad_amt` to `arr` along `axis` by extending edge values.

    Parameters
    ----------
    arr : ndarray
        Input array of arbitrary shape.
    pad_amt : int
        Amount of padding to append.
    axis : int
        Axis along which to pad `arr`.

    Returns
    -------
    padarr : ndarray
        Output array, extended by `pad_amt` edge values prepended along
        `axis`.

    """
    if pad_amt == 0:
        return arr

    edge_slice = tuple([slice(None) if i != axis else arr.shape[axis] - 1
                        for i, x in enumerate(arr.shape)])

    # Shape to restore singleton dimension after slicing
    pad_singleton = tuple([x if i != axis else 1
                           for i, x in enumerate(arr.shape)])
    edge_arr = arr[edge_slice].reshape(pad_singleton)
    return cupy.concatenate((arr, edge_arr.repeat(pad_amt, axis=axis)),
                            axis=axis)


def _pad_ref(arr, pad_amt, method, axis=-1):
    """Pad `axis` of `arr` by reflection.

    Parameters
    ----------
    arr : ndarray
        Input array of arbitrary shape.
    pad_amt : tuple of ints, length 2
        Padding to (prepend, append) along `axis`.
    method : str
        Controls method of reflection; options are 'even' or 'odd'.
    axis : int
        Axis along which to pad `arr`.

    Returns
    -------
    padarr : ndarray
        Output array, with `pad_amt[0]` values prepended and `pad_amt[1]`
        values appended along `axis`. Both regions are padded with reflected
        values from the original array.

    Notes
    -----
    This algorithm does not pad with repetition, i.e. the edges are not
    repeated in the reflection. For that behavior, use `mode='symmetric'`.

    The modes 'reflect', 'symmetric', and 'wrap' must be padded with a
    single function, lest the indexing tricks in non-integer multiples of the
    original shape would violate repetition in the final iteration.

    """
    # Implicit booleanness to test for zero (or None) in any scalar type
    if pad_amt[0] == 0 and pad_amt[1] == 0:
        return arr

    ##########################################################################
    # Prepended region

    # Slice off a reverse indexed chunk from near edge to pad `arr` before
    ref_slice = tuple([slice(None) if i != axis else slice(pad_amt[0], 0, -1)
                       for i, x in enumerate(arr.shape)])

    ref_chunk1 = arr[ref_slice]

    # Shape to restore singleton dimension after slicing
    pad_singleton = tuple([x if i != axis else 1
                           for i, x in enumerate(arr.shape)])
    if pad_amt[0] == 1:
        ref_chunk1 = ref_chunk1.reshape(pad_singleton)

    # Memory/computationally more expensive, only do this if `method='odd'`
    if 'odd' in method and pad_amt[0] > 0:
        edge_slice1 = tuple([slice(None) if i != axis else 0
                             for i, x in enumerate(arr.shape)])
        edge_chunk = arr[edge_slice1].reshape(pad_singleton)
        ref_chunk1 = 2 * edge_chunk - ref_chunk1
        del edge_chunk

    ##########################################################################
    # Appended region

    # Slice off a reverse indexed chunk from far edge to pad `arr` after
    start = arr.shape[axis] - pad_amt[1] - 1
    end = arr.shape[axis] - 1
    ref_slice = tuple([slice(None) if i != axis else slice(start, end)
                       for i, x in enumerate(arr.shape)])
    rev_idx = tuple([slice(None) if i != axis else slice(None, None, -1)
                     for i, x in enumerate(arr.shape)])
    ref_chunk2 = arr[ref_slice][rev_idx]

    if pad_amt[1] == 1:
        ref_chunk2 = ref_chunk2.reshape(pad_singleton)

    if 'odd' in method:
        edge_slice2 = tuple([slice(None) if i != axis else -1
                             for i, x in enumerate(arr.shape)])
        edge_chunk = arr[edge_slice2].reshape(pad_singleton)
        ref_chunk2 = 2 * edge_chunk - ref_chunk2
        del edge_chunk

    # Concatenate `arr` with both chunks, extending along `axis`
    return cupy.concatenate((ref_chunk1, arr, ref_chunk2), axis=axis)


def _normalize_shape(ndarray, shape, cast_to_int=True):
    ndims = ndarray.ndim
    if shape is None:
        return ((None, None), ) * ndims
    ndshape = numpy.asarray(shape)
    if ndshape.size == 1:
        ndshape = numpy.repeat(ndshape, 2)
    if ndshape.ndim == 1:
        ndshape = numpy.tile(ndshape, (ndims, 1))
    if ndshape.shape != (ndims, 2):
        message = 'Unable to create correctly shaped tuple from %s' % shape
        raise ValueError(message)
    if cast_to_int:
        ndshape = numpy.rint(ndshape).astype(int)
    return tuple([tuple(axis) for axis in ndshape])


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
            'edge'
                Pads with the edge values of array.
            'reflect'
                Pads with the reflection of the vector mirrored on the first
                and last values of the vector along each axis.
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
        reflect_type : {'even', 'odd'}, optional
            Used in 'reflect', and 'symmetric'.  The 'even' style is the
            default with an unaltered reflection around the edge value.  For
            the 'odd' style, the extented part of the array is created by
            subtracting the reflected values from two times the edge value.

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
        'edge': [],
        'reflect': ['reflect_type'],
    }
    keyword_defaults = {
        'constant_values': 0,
        'reflect_type': 'even',
    }
    if mode not in allowed_keywords:
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
    if mode == 'constant':
        for axis, ((pad_before, pad_after), (before_value, after_value)) \
                in enumerate(six.moves.zip(pad_width,
                                           keywords['constant_values'])):
            newmatrix = _prepend_const(
                newmatrix, pad_before, before_value, axis)
            newmatrix = _append_const(newmatrix, pad_after, after_value, axis)
    elif mode == 'edge':
        for axis, (pad_before, pad_after) in enumerate(pad_width):
            newmatrix = _prepend_edge(newmatrix, pad_before, axis)
            newmatrix = _append_edge(newmatrix, pad_after, axis)
    elif mode == 'reflect':
        for axis, (pad_before, pad_after) in enumerate(pad_width):
            if narray.shape[axis] == 0:
                # Axes with non-zero padding cannot be empty.
                if pad_before > 0 or pad_after > 0:
                    raise ValueError('There aren\'t any elements to reflect'
                                     ' in axis {} of `array`'.format(axis))
                # Skip zero padding on empty axes.
                continue

            # Recursive padding along any axis where `pad_amt` is too large
            # for indexing tricks. We can only safely pad the original axis
            # length, to keep the period of the reflections consistent.
            if ((pad_before > 0) or
                    (pad_after > 0)) and newmatrix.shape[axis] == 1:
                # Extending singleton dimension for 'reflect' is legacy
                # behavior; it really should raise an error.
                newmatrix = _prepend_edge(newmatrix, pad_before, axis)
                newmatrix = _append_edge(newmatrix, pad_after, axis)
                continue

            method = keywords['reflect_type']
            safe_pad = newmatrix.shape[axis] - 1
            while ((pad_before > safe_pad) or (pad_after > safe_pad)):
                pad_iter_b = min(safe_pad,
                                 safe_pad * (pad_before // safe_pad))
                pad_iter_a = min(safe_pad, safe_pad * (pad_after // safe_pad))
                newmatrix = _pad_ref(
                    newmatrix, (pad_iter_b, pad_iter_a), method, axis)
                pad_before -= pad_iter_b
                pad_after -= pad_iter_a
                safe_pad += pad_iter_b + pad_iter_a
            newmatrix = _pad_ref(
                newmatrix, (pad_before, pad_after), method, axis)
    return newmatrix

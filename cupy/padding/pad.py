import numbers

import numpy

import cupy


###############################################################################
# Private utility functions.


def _round_if_needed(arr, dtype):
    """Rounds arr inplace if the destination dtype is an integer.
    """
    if cupy.issubdtype(dtype, cupy.integer):
        arr.round(out=arr)  # bug in round so use rint (cupy/cupy#2330)


def _slice_at_axis(sl, axis):
    """Constructs a tuple of slices to slice an array in the given dimension.

    Args:
      sl(slice): The slice for the given dimension.
      axis(int): The axis to which `sl` is applied. All other dimensions are
          left "unsliced".

    Returns:
      tuple of slices: A tuple with slices matching `shape` in length.
    """
    return (slice(None),) * axis + (sl,) + (Ellipsis,)


def _view_roi(array, original_area_slice, axis):
    """Gets a view of the current region of interest during iterative padding.

    When padding multiple dimensions iteratively corner values are
    unnecessarily overwritten multiple times. This function reduces the
    working area for the first dimensions so that corners are excluded.

    Args:
      array(cupy.ndarray): The array with the region of interest.
      original_area_slice(tuple of slices): Denotes the area with original
          values of the unpadded array.
      axis(int): The currently padded dimension assuming that `axis` is padded
          before `axis` + 1.

    Returns:
    """
    axis += 1
    sl = (slice(None),) * axis + original_area_slice[axis:]
    return array[sl]


def _pad_simple(array, pad_width, fill_value=None):
    """Pads an array on all sides with either a constant or undefined values.

    Args:
      array(cupy.ndarray): Array to grow.
      pad_width(sequence of tuple[int, int]): Pad width on both sides for each
          dimension in `arr`.
      fill_value(scalar, optional): If provided the padded area is
          filled with this value, otherwise the pad area left undefined.
          (Default value = None)
    """
    # Allocate grown array
    new_shape = tuple(
        left + size + right
        for size, (left, right) in zip(array.shape, pad_width)
    )
    order = 'F' if array.flags.fnc else 'C'  # Fortran and not also C-order
    padded = cupy.empty(new_shape, dtype=array.dtype, order=order)

    if fill_value is not None:
        padded.fill(fill_value)

    # Copy old array into correct space
    original_area_slice = tuple(
        slice(left, left + size)
        for size, (left, right) in zip(array.shape, pad_width)
    )
    padded[original_area_slice] = array

    return padded, original_area_slice


def _set_pad_area(padded, axis, width_pair, value_pair):
    """Set an empty-padded area in given dimension.
    """
    left_slice = _slice_at_axis(slice(None, width_pair[0]), axis)
    padded[left_slice] = value_pair[0]

    right_slice = _slice_at_axis(
        slice(padded.shape[axis] - width_pair[1], None), axis
    )
    padded[right_slice] = value_pair[1]


def _get_edges(padded, axis, width_pair):
    """Retrieves edge values from an empty-padded array along a given axis.

    Args:
      padded(cupy.ndarray): Empty-padded array.
      axis(int): Dimension in which the edges are considered.
      width_pair((int, int)): Pair of widths that mark the pad area on both
          sides in the given dimension.
    """
    left_index = width_pair[0]
    left_slice = _slice_at_axis(slice(left_index, left_index + 1), axis)
    left_edge = padded[left_slice]

    right_index = padded.shape[axis] - width_pair[1]
    right_slice = _slice_at_axis(slice(right_index - 1, right_index), axis)
    right_edge = padded[right_slice]

    return left_edge, right_edge


def _get_linear_ramps(padded, axis, width_pair, end_value_pair):
    """Constructs linear ramps for an empty-padded array along a given axis.

    Args:
      padded(cupy.ndarray): Empty-padded array.
      axis(int): Dimension in which the ramps are constructed.
      width_pair((int, int)): Pair of widths that mark the pad area on both
          sides in the given dimension.
      end_value_pair((scalar, scalar)): End values for the linear ramps which
          form the edge of the fully padded array. These values are included in
          the linear ramps.
    """
    edge_pair = _get_edges(padded, axis, width_pair)

    left_ramp = cupy.linspace(
        start=end_value_pair[0],
        # squeeze axis replaced by linspace
        stop=edge_pair[0].squeeze(axis),
        num=width_pair[0],
        endpoint=False,
        dtype=padded.dtype,
        axis=axis,
    )

    right_ramp = cupy.linspace(
        start=end_value_pair[1],
        # squeeze axis replaced by linspace
        stop=edge_pair[1].squeeze(axis),
        num=width_pair[1],
        endpoint=False,
        dtype=padded.dtype,
        axis=axis,
    )
    # Reverse linear space in appropriate dimension
    right_ramp = right_ramp[_slice_at_axis(slice(None, None, -1), axis)]

    return left_ramp, right_ramp


def _get_stats(padded, axis, width_pair, length_pair, stat_func):
    """Calculates a statistic for an empty-padded array along a given axis.

    Args:
      padded(cupy.ndarray): Empty-padded array.
      axis(int): Dimension in which the statistic is calculated.
      width_pair((int, int)): Pair of widths that mark the pad area on both
          sides in the given dimension.
      length_pair(2-element sequence of None or int): Gives the number of
          values in valid area from each side that is taken into account when
          calculating the statistic. If None the entire valid area in `padded`
          is considered.
      stat_func(function): Function to compute statistic. The expected
          signature is
          ``stat_func(x: ndarray, axis: int, keepdims: bool) -> ndarray``.
    """
    # Calculate indices of the edges of the area with original values
    left_index = width_pair[0]
    right_index = padded.shape[axis] - width_pair[1]
    # as well as its length
    max_length = right_index - left_index

    # Limit stat_lengths to max_length
    left_length, right_length = length_pair
    if left_length is None or max_length < left_length:
        left_length = max_length
    if right_length is None or max_length < right_length:
        right_length = max_length

    # Calculate statistic for the left side
    left_slice = _slice_at_axis(
        slice(left_index, left_index + left_length), axis
    )
    left_chunk = padded[left_slice]
    left_stat = stat_func(left_chunk, axis=axis, keepdims=True)
    _round_if_needed(left_stat, padded.dtype)

    if left_length == right_length == max_length:
        # return early as right_stat must be identical to left_stat
        return left_stat, left_stat

    # Calculate statistic for the right side
    right_slice = _slice_at_axis(
        slice(right_index - right_length, right_index), axis
    )
    right_chunk = padded[right_slice]
    right_stat = stat_func(right_chunk, axis=axis, keepdims=True)
    _round_if_needed(right_stat, padded.dtype)
    return left_stat, right_stat


def _set_reflect_both(padded, axis, width_pair, method, include_edge=False):
    """Pads an `axis` of `arr` using reflection.

    Args:
      padded(cupy.ndarray): Input array of arbitrary shape.
      axis(int): Axis along which to pad `arr`.
      width_pair((int, int)): Pair of widths that mark the pad area on both
          sides in the given dimension.
      method(str): Controls method of reflection; options are 'even' or 'odd'.
      include_edge(bool, optional): If true, edge value is included in
          reflection, otherwise the edge value forms the symmetric axis to the
          reflection. (Default value = False)
    """
    left_pad, right_pad = width_pair
    old_length = padded.shape[axis] - right_pad - left_pad

    if include_edge:
        # Edge is included, we need to offset the pad amount by 1
        edge_offset = 1
    else:
        edge_offset = 0  # Edge is not included, no need to offset pad amount
        old_length -= 1  # but must be omitted from the chunk

    if left_pad > 0:
        # Pad with reflected values on left side:
        # First limit chunk size which can't be larger than pad area
        chunk_length = min(old_length, left_pad)
        # Slice right to left, stop on or next to edge, start relative to stop
        stop = left_pad - edge_offset
        start = stop + chunk_length
        left_slice = _slice_at_axis(slice(start, stop, -1), axis)
        left_chunk = padded[left_slice]

        if method == 'odd':
            # Negate chunk and align with edge
            edge_slice = _slice_at_axis(slice(left_pad, left_pad + 1), axis)
            left_chunk = 2 * padded[edge_slice] - left_chunk

        # Insert chunk into padded area
        start = left_pad - chunk_length
        stop = left_pad
        pad_area = _slice_at_axis(slice(start, stop), axis)
        padded[pad_area] = left_chunk
        # Adjust pointer to left edge for next iteration
        left_pad -= chunk_length

    if right_pad > 0:
        # Pad with reflected values on right side:
        # First limit chunk size which can't be larger than pad area
        chunk_length = min(old_length, right_pad)
        # Slice right to left, start on or next to edge, stop relative to start
        start = -right_pad + edge_offset - 2
        stop = start - chunk_length
        right_slice = _slice_at_axis(slice(start, stop, -1), axis)
        right_chunk = padded[right_slice]

        if method == 'odd':
            # Negate chunk and align with edge
            edge_slice = _slice_at_axis(
                slice(-right_pad - 1, -right_pad), axis
            )
            right_chunk = 2 * padded[edge_slice] - right_chunk

        # Insert chunk into padded area
        start = padded.shape[axis] - right_pad
        stop = start + chunk_length
        pad_area = _slice_at_axis(slice(start, stop), axis)
        padded[pad_area] = right_chunk
        # Adjust pointer to right edge for next iteration
        right_pad -= chunk_length

    return left_pad, right_pad


def _set_wrap_both(padded, axis, width_pair):
    """Pads an `axis` of `arr` with wrapped values.

    Args:
      padded(cupy.ndarray): Input array of arbitrary shape.
      axis(int): Axis along which to pad `arr`.
      width_pair((int, int)): Pair of widths that mark the pad area on both
          sides in the given dimension.
    """
    left_pad, right_pad = width_pair
    period = padded.shape[axis] - right_pad - left_pad

    # If the current dimension of `arr` doesn't contain enough valid values
    # (not part of the undefined pad area) we need to pad multiple times.
    # Each time the pad area shrinks on both sides which is communicated with
    # these variables.
    new_left_pad = 0
    new_right_pad = 0

    if left_pad > 0:
        # Pad with wrapped values on left side
        # First slice chunk from right side of the non-pad area.
        # Use min(period, left_pad) to ensure that chunk is not larger than
        # pad area
        right_slice = _slice_at_axis(
            slice(
                -right_pad - min(period, left_pad),
                -right_pad if right_pad != 0 else None,
            ),
            axis,
        )
        right_chunk = padded[right_slice]

        if left_pad > period:
            # Chunk is smaller than pad area
            pad_area = _slice_at_axis(slice(left_pad - period, left_pad), axis)
            new_left_pad = left_pad - period
        else:
            # Chunk matches pad area
            pad_area = _slice_at_axis(slice(None, left_pad), axis)
        padded[pad_area] = right_chunk

    if right_pad > 0:
        # Pad with wrapped values on right side
        # First slice chunk from left side of the non-pad area.
        # Use min(period, right_pad) to ensure that chunk is not larger than
        # pad area
        left_slice = _slice_at_axis(
            slice(left_pad, left_pad + min(period, right_pad)), axis
        )
        left_chunk = padded[left_slice]

        if right_pad > period:
            # Chunk is smaller than pad area
            pad_area = _slice_at_axis(
                slice(-right_pad, -right_pad + period), axis
            )
            new_right_pad = right_pad - period
        else:
            # Chunk matches pad area
            pad_area = _slice_at_axis(slice(-right_pad, None), axis)
        padded[pad_area] = left_chunk

    return new_left_pad, new_right_pad


def _as_pairs(x, ndim, as_index=False):
    """Broadcasts `x` to an array with shape (`ndim`, 2).

    A helper function for `pad` that prepares and validates arguments like
    `pad_width` for iteration in pairs.

    Args:
      x(scalar or array-like, optional): The object to broadcast to the shape
          (`ndim`, 2).
      ndim(int): Number of pairs the broadcasted `x` will have.
      as_index(bool, optional): If `x` is not None, try to round each
          element of `x` to an integer (dtype `cupy.intp`) and ensure every
          element is positive. (Default value = False)

    Returns:
      nested iterables, shape (`ndim`, 2): The broadcasted version of `x`.
    """
    if x is None:
        # Pass through None as a special case, otherwise cupy.round(x) fails
        # with an AttributeError
        return ((None, None),) * ndim
    elif isinstance(x, numbers.Number):
        if as_index:
            x = round(x)
        return ((x, x),) * ndim

    x = numpy.array(x)
    if as_index:
        x = numpy.asarray(numpy.round(x), dtype=numpy.intp)

    if x.ndim < 3:
        # Optimization: Possibly use faster paths for cases where `x` has
        # only 1 or 2 elements. `numpy.broadcast_to` could handle these as well
        # but is currently slower

        if x.size == 1:
            # x was supplied as a single value
            x = x.ravel()  # Ensure x[0] works for x.ndim == 0, 1, 2
            if as_index and x < 0:
                raise ValueError("index can't contain negative values")
            return ((x[0], x[0]),) * ndim

        if x.size == 2 and x.shape != (2, 1):
            # x was supplied with a single value for each side
            # but except case when each dimension has a single value
            # which should be broadcasted to a pair,
            # e.g. [[1], [2]] -> [[1, 1], [2, 2]] not [[1, 2], [1, 2]]
            x = x.ravel()  # Ensure x[0], x[1] works
            if as_index and (x[0] < 0 or x[1] < 0):
                raise ValueError("index can't contain negative values")
            return ((x[0], x[1]),) * ndim

    if as_index and x.min() < 0:
        raise ValueError("index can't contain negative values")

    # Converting the array with `tolist` seems to improve performance
    # when iterating and indexing the result (see usage in `pad`)
    x_view = x.view()
    x_view.shape = (ndim, 2)
    return x_view.tolist()

# def _pad_dispatcher(array, pad_width, mode=None, **kwargs):
#    return (array,)


###############################################################################
# Public functions


# @array_function_dispatch(_pad_dispatcher, module='numpy')
def pad(array, pad_width, mode='constant', **kwargs):
    """Pads an array with specified widths and values.

    Args:
      array(cupy.ndarray): The array to pad.
      pad_width(sequence, array_like or int): Number of values padded to the
          edges of each axis. ((before_1, after_1), ... (before_N, after_N))
          unique pad widths for each axis. ((before, after),) yields same
          before and after pad for each axis. (pad,) or int is a shortcut for
          before = after = pad width for all axes. You cannot specify
          ``cupy.ndarray``.
      mode(str or function, optional): One of the following string values or a
          user supplied function

          'constant' (default)
              Pads with a constant value.
          'edge'
              Pads with the edge values of array.
          'linear_ramp'
              Pads with the linear ramp between end_value and the array edge
              value.
          'maximum'
              Pads with the maximum value of all or part of the vector along
              each axis.
          'mean'
              Pads with the mean value of all or part of the vector along each
              axis.
          'median'
              Pads with the median value of all or part of the vector along
              each axis. (Not Implemented)
          'minimum'
              Pads with the minimum value of all or part of the vector along
              each axis.
          'reflect'
              Pads with the reflection of the vector mirrored on the first and
              last values of the vector along each axis.
          'symmetric'
               Pads with the reflection of the vector mirrored along the edge
               of the array.
          'wrap'
              Pads with the wrap of the vector along the axis. The first
              values are used to pad the end and the end values are used to
              pad the beginning.
          'empty'
              Pads with undefined values.
          <function>
              Padding function, see Notes.
      stat_length(sequence or int, optional): Used in 'maximum', 'mean',
          'median', and 'minimum'.  Number of values at edge of each axis used
          to calculate the statistic value.
          ((before_1, after_1), ... (before_N, after_N)) unique statistic
          lengths for each axis. ((before, after),) yields same before and
          after statistic lengths for each axis. (stat_length,) or int is a
          shortcut for before = after = statistic length for all axes.
          Default is ``None``, to use the entire axis. You cannot specify
          ``cupy.ndarray``.
      constant_values(sequence or scalar, optional): Used in 'constant'. The
          values to set the padded values for each axis.
          ((before_1, after_1), ... (before_N, after_N)) unique pad constants
          for each axis.
          ((before, after),) yields same before and after constants for each
          axis.
          (constant,) or constant is a shortcut for before = after = constant
          for all axes.
          Default is 0. You cannot specify ``cupy.ndarray``.
      end_values(sequence or scalar, optional): Used in 'linear_ramp'. The
          values used for the ending value of the linear_ramp and that will
          form the edge of the padded array.
          ((before_1, after_1), ... (before_N, after_N)) unique end values
          for each axis.
          ((before, after),) yields same before and after end
          values for each axis.
          (constant,) or constant is a shortcut for before = after = constant
          for all axes.
          Default is 0. You cannot specify ``cupy.ndarray``.
      reflect_type({'even', 'odd'}, optional): Used in 'reflect', and
          'symmetric'.  The 'even' style is the default with an unaltered
          reflection around the edge value.  For the 'odd' style, the extended
          part of the array is created by subtracting the reflected values from
          two times the edge value.

    Returns:
      cupy.ndarray: Padded array with shape extended by ``pad_width``.

    .. note::
        For an array with rank greater than 1, some of the padding of later
        axes is calculated from padding of previous axes.  This is easiest to
        think about with a rank 2 array where the corners of the padded array
        are calculated by using padded values from the first axis.

        The padding function, if used, should modify a rank 1 array in-place.
        It has the following signature:

        ``padding_func(vector, iaxis_pad_width, iaxis, kwargs)``

        where

        vector (cupy.ndarray)
            A rank 1 array already padded with zeros.  Padded values are
            ``vector[:iaxis_pad_width[0]]`` and
            ``vector[-iaxis_pad_width[1]:]``.
        iaxis_pad_width (tuple)
            A 2-tuple of ints, ``iaxis_pad_width[0]`` represents the number of
            values padded at the beginning of vector where
            ``iaxis_pad_width[1]`` represents the number of values padded at
            the end of vector.
        iaxis (int)
            The axis currently being calculated.
        kwargs (dict)
            Any keyword arguments the function requires.

    Examples
    --------
    >>> a = cupy.array([1, 2, 3, 4, 5])
    >>> cupy.pad(a, (2, 3), 'constant', constant_values=(4, 6))
    array([4, 4, 1, ..., 6, 6, 6])

    >>> cupy.pad(a, (2, 3), 'edge')
    array([1, 1, 1, ..., 5, 5, 5])

    >>> cupy.pad(a, (2, 3), 'linear_ramp', end_values=(5, -4))
    array([ 5,  3,  1,  2,  3,  4,  5,  2, -1, -4])

    >>> cupy.pad(a, (2,), 'maximum')
    array([5, 5, 1, 2, 3, 4, 5, 5, 5])

    >>> cupy.pad(a, (2,), 'mean')
    array([3, 3, 1, 2, 3, 4, 5, 3, 3])

    >>> a = cupy.array([[1, 2], [3, 4]])
    >>> cupy.pad(a, ((3, 2), (2, 3)), 'minimum')
    array([[1, 1, 1, 2, 1, 1, 1],
           [1, 1, 1, 2, 1, 1, 1],
           [1, 1, 1, 2, 1, 1, 1],
           [1, 1, 1, 2, 1, 1, 1],
           [3, 3, 3, 4, 3, 3, 3],
           [1, 1, 1, 2, 1, 1, 1],
           [1, 1, 1, 2, 1, 1, 1]])

    >>> a = cupy.array([1, 2, 3, 4, 5])
    >>> cupy.pad(a, (2, 3), 'reflect')
    array([3, 2, 1, 2, 3, 4, 5, 4, 3, 2])

    >>> cupy.pad(a, (2, 3), 'reflect', reflect_type='odd')
    array([-1,  0,  1,  2,  3,  4,  5,  6,  7,  8])

    >>> cupy.pad(a, (2, 3), 'symmetric')
    array([2, 1, 1, 2, 3, 4, 5, 5, 4, 3])

    >>> cupy.pad(a, (2, 3), 'symmetric', reflect_type='odd')
    array([0, 1, 1, 2, 3, 4, 5, 5, 6, 7])

    >>> cupy.pad(a, (2, 3), 'wrap')
    array([4, 5, 1, 2, 3, 4, 5, 1, 2, 3])

    >>> def pad_with(vector, pad_width, iaxis, kwargs):
    ...     pad_value = kwargs.get('padder', 10)
    ...     vector[:pad_width[0]] = pad_value
    ...     vector[-pad_width[1]:] = pad_value
    >>> a = cupy.arange(6)
    >>> a = a.reshape((2, 3))
    >>> cupy.pad(a, 2, pad_with)
    array([[10, 10, 10, 10, 10, 10, 10],
           [10, 10, 10, 10, 10, 10, 10],
           [10, 10,  0,  1,  2, 10, 10],
           [10, 10,  3,  4,  5, 10, 10],
           [10, 10, 10, 10, 10, 10, 10],
           [10, 10, 10, 10, 10, 10, 10]])
    >>> cupy.pad(a, 2, pad_with, padder=100)
    array([[100, 100, 100, 100, 100, 100, 100],
           [100, 100, 100, 100, 100, 100, 100],
           [100, 100,   0,   1,   2, 100, 100],
           [100, 100,   3,   4,   5, 100, 100],
           [100, 100, 100, 100, 100, 100, 100],
           [100, 100, 100, 100, 100, 100, 100]])
    """
    if isinstance(pad_width, numbers.Integral):
        pad_width = ((pad_width, pad_width),) * array.ndim
    else:
        pad_width = numpy.asarray(pad_width)

        if not pad_width.dtype.kind == 'i':
            raise TypeError('`pad_width` must be of integral type.')

        # Broadcast to shape (array.ndim, 2)
        pad_width = _as_pairs(pad_width, array.ndim, as_index=True)

    if callable(mode):
        # Old behavior: Use user-supplied function with numpy.apply_along_axis
        function = mode
        # Create a new zero padded array
        padded, _ = _pad_simple(array, pad_width, fill_value=0)
        # And apply along each axis

        for axis in range(padded.ndim):
            # Iterate using ndindex as in apply_along_axis, but assuming that
            # function operates inplace on the padded array.

            # view with the iteration axis at the end
            view = cupy.moveaxis(padded, axis, -1)

            # compute indices for the iteration axes, and append a trailing
            # ellipsis to prevent 0d arrays decaying to scalars (gh-8642)
            inds = numpy.ndindex(view.shape[:-1])
            inds = (ind + (Ellipsis,) for ind in inds)
            for ind in inds:
                function(view[ind], pad_width[axis], axis, kwargs)

        return padded

    # Make sure that no unsupported keywords were passed for the current mode
    allowed_kwargs = {
        'empty': [],
        'edge': [],
        'wrap': [],
        'constant': ['constant_values'],
        'linear_ramp': ['end_values'],
        'maximum': ['stat_length'],
        'mean': ['stat_length'],
        # 'median': ['stat_length'],
        'minimum': ['stat_length'],
        'reflect': ['reflect_type'],
        'symmetric': ['reflect_type'],
    }
    try:
        unsupported_kwargs = set(kwargs) - set(allowed_kwargs[mode])
    except KeyError:
        raise ValueError("mode '{}' is not supported".format(mode))
    if unsupported_kwargs:
        raise ValueError(
            "unsupported keyword arguments for mode '{}': {}".format(
                mode, unsupported_kwargs
            )
        )

    if mode == 'constant':
        values = kwargs.get('constant_values', 0)
        if isinstance(values, numbers.Number) and values == 0 and (
                array.ndim == 1 or array.size < 4e6):
            # faster path for 1d arrays or small n-dimensional arrays
            return _pad_simple(array, pad_width, 0)[0]

    stat_functions = {
        'maximum': cupy.max,
        'minimum': cupy.min,
        'mean': cupy.mean,
        # 'median': cupy.median,
    }

    # Create array with final shape and original values
    # (padded area is undefined)
    padded, original_area_slice = _pad_simple(array, pad_width)
    # And prepare iteration over all dimensions
    # (zipping may be more readable than using enumerate)
    axes = range(padded.ndim)

    if mode == 'constant':
        values = _as_pairs(values, padded.ndim)
        for axis, width_pair, value_pair in zip(axes, pad_width, values):
            roi = _view_roi(padded, original_area_slice, axis)
            _set_pad_area(roi, axis, width_pair, value_pair)

    elif mode == 'empty':
        pass  # Do nothing as _pad_simple already returned the correct result

    elif array.size == 0:
        # Only modes 'constant' and 'empty' can extend empty axes, all other
        # modes depend on `array` not being empty
        # -> ensure every empty axis is only 'padded with 0'
        for axis, width_pair in zip(axes, pad_width):
            if array.shape[axis] == 0 and any(width_pair):
                raise ValueError(
                    "can't extend empty axis {} using modes other than "
                    "'constant' or 'empty'".format(axis)
                )
        # passed, don't need to do anything more as _pad_simple already
        # returned the correct result

    elif mode == 'edge':
        for axis, width_pair in zip(axes, pad_width):
            roi = _view_roi(padded, original_area_slice, axis)
            edge_pair = _get_edges(roi, axis, width_pair)
            _set_pad_area(roi, axis, width_pair, edge_pair)

    elif mode == 'linear_ramp':
        end_values = kwargs.get('end_values', 0)
        end_values = _as_pairs(end_values, padded.ndim)
        for axis, width_pair, value_pair in zip(axes, pad_width, end_values):
            roi = _view_roi(padded, original_area_slice, axis)
            ramp_pair = _get_linear_ramps(roi, axis, width_pair, value_pair)
            _set_pad_area(roi, axis, width_pair, ramp_pair)

    elif mode in stat_functions:
        func = stat_functions[mode]
        length = kwargs.get('stat_length', None)
        length = _as_pairs(length, padded.ndim, as_index=True)
        for axis, width_pair, length_pair in zip(axes, pad_width, length):
            roi = _view_roi(padded, original_area_slice, axis)
            stat_pair = _get_stats(roi, axis, width_pair, length_pair, func)
            _set_pad_area(roi, axis, width_pair, stat_pair)

    elif mode in {'reflect', 'symmetric'}:
        method = kwargs.get('reflect_type', 'even')
        include_edge = True if mode == 'symmetric' else False
        for axis, (left_index, right_index) in zip(axes, pad_width):
            if array.shape[axis] == 1 and (left_index > 0 or right_index > 0):
                # Extending singleton dimension for 'reflect' is legacy
                # behavior; it really should raise an error.
                edge_pair = _get_edges(padded, axis, (left_index, right_index))
                _set_pad_area(
                    padded, axis, (left_index, right_index), edge_pair
                )
                continue

            roi = _view_roi(padded, original_area_slice, axis)
            while left_index > 0 or right_index > 0:
                # Iteratively pad until dimension is filled with reflected
                # values. This is necessary if the pad area is larger than
                # the length of the original values in the current dimension.
                left_index, right_index = _set_reflect_both(
                    roi, axis, (left_index, right_index), method, include_edge
                )

    elif mode == 'wrap':
        for axis, (left_index, right_index) in zip(axes, pad_width):
            roi = _view_roi(padded, original_area_slice, axis)
            while left_index > 0 or right_index > 0:
                # Iteratively pad until dimension is filled with wrapped
                # values. This is necessary if the pad area is larger than
                # the length of the original values in the current dimension.
                left_index, right_index = _set_wrap_both(
                    roi, axis, (left_index, right_index)
                )

    return padded

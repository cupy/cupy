import warnings

import cupy


def correlate(input, weights, output=None, mode='reflect', cval=0.0, origin=0):
    """Multi-dimensional correlate.

    The array is correlated with the given kernel.

    Args:
        input (cupy.ndarray): The input array.
        weights (cupy.ndarray): Array of weights, same number of dimensions as
            input
        output (cupy.ndarray, dtype or None): The array in which to place the
            output.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``constant``. Default is ``0.0``.
        origin (scalar or tuple of scalar): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.

    Returns:
        cupy.ndarray: The result of correlate.

    .. seealso:: :func:`scipy.ndimage.correlate`
    """
    return _correlate_or_convolve(input, weights, output, mode, cval, origin)


def convolve(input, weights, output=None, mode='reflect', cval=0.0, origin=0):
    """Multi-dimensional convolution.

    The array is convolved with the given kernel.

    Args:
        input (cupy.ndarray): The input array.
        weights (cupy.ndarray): Array of weights, same number of dimensions as
            input
        output (cupy.ndarray, dtype or None): The array in which to place the
            output.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``constant``. Default is ``0.0``.
        origin (scalar or tuple of scalar): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.

    Returns:
        cupy.ndarray: The result of convolution.

    .. seealso:: :func:`scipy.ndimage.convolve`
    """
    return _correlate_or_convolve(input, weights, output, mode, cval, origin,
                                  True)


def correlate1d(input, weights, axis=-1, output=None, mode="reflect", cval=0.0,
                origin=0):
    """One-dimensional correlate.

    The array is correlated with the given kernel.

    Args:
        input (cupy.ndarray): The input array.
        weights (cupy.ndarray): One-dimensional array of weights
        axis (int): The axis of input along which to calculate. Default is -1.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int): The origin parameter controls the placement of the
            filter, relative to the center of the current element of the
            input. Default is ``0``.
    Returns:
        cupy.ndarray: The result of the 1D correlation.
    .. seealso:: :func:`scipy.ndimage.correlate1d`
    """
    weights, origins = _convert_1d_args(input.ndim, weights, origin, axis)
    return _correlate_or_convolve(input, weights, output, mode, cval, origins)


def convolve1d(input, weights, axis=-1, output=None, mode="reflect", cval=0.0,
               origin=0):
    """One-dimensional convolution.

    The array is convolved with the given kernel.

    Args:
        input (cupy.ndarray): The input array.
        weights (cupy.ndarray): One-dimensional array of weights
        axis (int): The axis of input along which to calculate. Default is -1.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int): The origin parameter controls the placement of the
            filter, relative to the center of the current element of the
            input. Default is ``0``.
    Returns:
        cupy.ndarray: The result of the 1D convolution.
    .. seealso:: :func:`scipy.ndimage.convolve1d`
    """
    weights = weights[::-1]
    origin = -origin
    if not len(weights) & 1:
        origin -= 1
    weights, origins = _convert_1d_args(input.ndim, weights, origin, axis)
    return _correlate_or_convolve(input, weights, output, mode, cval, origins)


def _correlate_or_convolve(input, weights, output, mode, cval, origin,
                           convolution=False):
    origins, int_type = _check_nd_args(input, weights, mode, origin)
    if weights.size == 0:
        return cupy.zeros_like(input)
    if convolution:
        weights = weights[tuple([slice(None, None, -1)] * weights.ndim)]
        origins = list(origins)
        for i, wsize in enumerate(weights.shape):
            origins[i] = -origins[i]
            if wsize % 2 == 0:
                origins[i] -= 1
        origins = tuple(origins)
    kernel = _get_correlate_kernel(mode, weights.shape, int_type,
                                   origins, cval)
    return _call_kernel(kernel, input, weights, output)


@cupy.util.memoize(for_each_device=True)
def _get_correlate_kernel(mode, wshape, int_type, origins, cval):
    return _generate_nd_kernel(
        'correlate',
        'W sum = (W)0;',
        'sum += (W){value} * wval;',
        'y = (Y)sum;',
        mode, wshape, int_type, origins, cval)


def minimum_filter(input, size=None, footprint=None, output=None,
                   mode="reflect", cval=0.0, origin=0):
    """Multi-dimensional minimum filter.

    Args:
        input (cupy.ndarray): The input array.
        size (int or sequence of int): One of ``size`` or ``footprint`` must be
            provided. If ``footprint`` is given, ``size`` is ignored. Otherwise
            ``footprint = cupy.ones(size)`` with ``size`` automatically made to
            match the number of dimensions in ``input``.
        footprint (cupy.ndarray): a boolean array which specifies which of the
            elements within this shape will get passed to the filter function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int or sequence of int): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.minimum_filter`
    """
    return _min_or_max_filter(input, size, footprint, None, output, mode,
                              cval, origin, 'min')


def maximum_filter(input, size=None, footprint=None, output=None,
                   mode="reflect", cval=0.0, origin=0):
    """Multi-dimensional maximum filter.

    Args:
        input (cupy.ndarray): The input array.
        size (int or sequence of int): One of ``size`` or ``footprint`` must be
            provided. If ``footprint`` is given, ``size`` is ignored. Otherwise
            ``footprint = cupy.ones(size)`` with ``size`` automatically made to
            match the number of dimensions in ``input``.
        footprint (cupy.ndarray): a boolean array which specifies which of the
            elements within this shape will get passed to the filter function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int or sequence of int): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.maximum_filter`
    """
    return _min_or_max_filter(input, size, footprint, None, output, mode,
                              cval, origin, 'max')


def _min_or_max_filter(input, size, footprint, structure, output, mode, cval,
                       origin, func):
    # structure is used by morphology.grey_erosion() and grey_dilation()
    # and not by the regular min/max filters

    sizes, footprint, structure = _check_size_footprint_structure(
        input.ndim, size, footprint, structure)

    if sizes is not None:
        # Seperable filter, run as a series of 1D filters
        fltr = minimum_filter1d if func == 'min' else maximum_filter1d
        output_orig = output
        output = _get_output(output, input)
        sizes = _fix_sequence_arg(sizes, input.ndim, 'size', int)
        modes = _fix_sequence_arg(mode, input.ndim, 'mode', _check_mode)
        origins = _fix_sequence_arg(origin, input.ndim, 'origin', int)
        n_filters = sum(size > 1 for size in sizes)
        if n_filters == 0:
            output[...] = input[...]
            return output
        # We can't operate in-place efficiently, so use a 2-buffer system
        temp = _get_output(output.dtype, input) if n_filters > 1 else None
        first = True
        iterator = zip(sizes, modes, origins)
        for axis, (size, mode, origin) in enumerate(iterator):
            if size <= 1:
                continue
            fltr(input, size, axis, output, mode, cval, origin)
            input, output = output, temp if first else input
        if isinstance(output_orig, cupy.ndarray) and input is not output_orig:
            output_orig[...] = input
            input = output_orig
        return input

    origins, int_type = _check_nd_args(input, footprint, mode, origin,
                                       'footprint')
    if structure is not None and structure.ndim != input.ndim:
        raise RuntimeError('structure array has incorrect shape')

    if footprint.size == 0:
        return cupy.zeros_like(input)
    center = tuple(x//2 + origin
                   for x, origin in zip(footprint.shape, origins))
    kernel = _get_min_or_max_kernel(mode, footprint.shape, func,
                                    origins, float(cval), int_type,
                                    has_structure=structure is not None,
                                    has_central_value=bool(footprint[center]))
    return _call_kernel(kernel, input, footprint, output, structure,
                        weights_dtype=bool)


def minimum_filter1d(input, size, axis=-1, output=None, mode="reflect",
                     cval=0.0, origin=0):
    """Compute the minimum filter along a single axis.

    Args:
        input (cupy.ndarray): The input array.
        size (int): Length of the minimum filter.
        axis (int): The axis of input along which to calculate. Default is -1.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int): The origin parameter controls the placement of the
            filter, relative to the center of the current element of the
            input. Default is ``0``.
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.minimum_filter1d`
    """
    return _min_or_max_1d(input, size, axis, output, mode, cval, origin, 'min')


def maximum_filter1d(input, size, axis=-1, output=None, mode="reflect",
                     cval=0.0, origin=0):
    """Compute the maximum filter along a single axis.

    Args:
        input (cupy.ndarray): The input array.
        size (int): Length of the maximum filter.
        axis (int): The axis of input along which to calculate. Default is -1.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int): The origin parameter controls the placement of the
            filter, relative to the center of the current element of the
            input. Default is ``0``.
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.maximum_filter1d`
    """
    return _min_or_max_1d(input, size, axis, output, mode, cval, origin, 'max')


def _min_or_max_1d(input, size, axis=-1, output=None, mode="reflect", cval=0.0,
                   origin=0, func='min'):
    ftprnt = cupy.ones(size, dtype=bool)
    ftprnt, origins = _convert_1d_args(input.ndim, ftprnt, origin, axis)
    origins, int_type = _check_nd_args(input, ftprnt, mode, origins,
                                       'footprint')
    kernel = _get_min_or_max_kernel(mode, ftprnt.shape, func, origins,
                                    float(cval), int_type, has_weights=False)
    return _call_kernel(kernel, input, None, output, weights_dtype=bool)


@cupy.util.memoize(for_each_device=True)
def _get_min_or_max_kernel(mode, wshape, func, origins, cval, int_type,
                           has_weights=True, has_structure=False,
                           has_central_value=True):
    value = '{value}'
    if has_structure:
        value += ' - (X)sval' if func == 'min' else ' + (X)sval'

    if has_central_value:
        pre = 'X value = x[i];'
        found = 'value = {func}({value}, value);'
    else:
        # If the central pixel is not included in the footprint we cannot
        # assume `x[i]` is not below the min or above the max and thus cannot
        # seed with that value. Instead we keep track of having set `value`.
        pre = 'X value; bool set = false;'
        found = 'value = set ? {func}({value}, value) : {value}; set=true;'
    return _generate_nd_kernel(
        func, pre, found.format(func=func, value=value), 'y = (Y)value;',
        mode, wshape, int_type, origins, cval,
        has_weights=has_weights, has_structure=has_structure)


def rank_filter(input, rank, size=None, footprint=None, output=None,
                mode="reflect", cval=0.0, origin=0):
    """Multi-dimensional rank filter.
    Args:
        input (cupy.ndarray): The input array.
        rank (int): The rank of the element to get. Can be negative to count
            from the largest value, e.g. ``-1`` indicates the largest value.
        size (int or sequence of int): One of ``size`` or ``footprint`` must be
            provided. If ``footprint`` is given, ``size`` is ignored. Otherwise
            ``footprint = cupy.ones(size)`` with ``size`` automatically made to
            match the number of dimensions in ``input``.
        footprint (cupy.ndarray): a boolean array which specifies which of the
            elements within this shape will get passed to the filter function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int or sequence of int): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.rank_filter`
    """
    rank = int(rank)
    return _rank_filter(input, lambda fs: rank+fs if rank < 0 else rank,
                        size, footprint, output, mode, cval, origin)


def median_filter(input, size=None, footprint=None, output=None,
                  mode="reflect", cval=0.0, origin=0):
    """Multi-dimensional median filter.
    Args:
        input (cupy.ndarray): The input array.
        size (int or sequence of int): One of ``size`` or ``footprint`` must be
            provided. If ``footprint`` is given, ``size`` is ignored. Otherwise
            ``footprint = cupy.ones(size)`` with ``size`` automatically made to
            match the number of dimensions in ``input``.
        footprint (cupy.ndarray): a boolean array which specifies which of the
            elements within this shape will get passed to the filter function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int or sequence of int): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.median_filter`
    """
    return _rank_filter(input, lambda fs: fs//2,
                        size, footprint, output, mode, cval, origin)


def percentile_filter(input, percentile, size=None, footprint=None,
                      output=None, mode="reflect", cval=0.0, origin=0):
    """Multi-dimensional percentile filter.
    Args:
        input (cupy.ndarray): The input array.
        percentile (scalar): The percentile of the element to get (from ``0``
            to ``100``). Can be negative, thus ``-20`` equals ``80``.
        size (int or sequence of int): One of ``size`` or ``footprint`` must be
            provided. If ``footprint`` is given, ``size`` is ignored. Otherwise
            ``footprint = cupy.ones(size)`` with ``size`` automatically made to
            match the number of dimensions in ``input``.
        footprint (cupy.ndarray): a boolean array which specifies which of the
            elements within this shape will get passed to the filter function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int or sequence of int): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.percentile_filter`
    """
    percentile = float(percentile)
    if percentile < 0.0:
        percentile += 100.0
    if percentile < 0.0 or percentile > 100.0:
        raise RuntimeError('invalid percentile')
    if percentile == 100.0:
        def get_rank(fs):
            return fs - 1
    else:
        def get_rank(fs):
            return int(float(fs) * percentile / 100.0)
    return _rank_filter(input, get_rank,
                        size, footprint, output, mode, cval, origin)


def _rank_filter(input, get_rank, size=None, footprint=None, output=None,
                 mode="reflect", cval=0.0, origin=0):
    _, footprint, _ = _check_size_footprint_structure(
        input.ndim, size, footprint, None, force_footprint=True)
    origins, int_type = _check_nd_args(input, footprint, mode, origin,
                                       'footprint')
    if footprint.size == 0:
        return cupy.zeros_like(input)
    filter_size = int(footprint.sum())
    rank = get_rank(filter_size)
    if rank < 0 or rank >= filter_size:
        raise RuntimeError('rank not within filter footprint size')
    if rank == 0:
        return _min_or_max_filter(input, None, footprint, None, output, mode,
                                  cval, origins, 'min')
    if rank == filter_size - 1:
        return _min_or_max_filter(input, None, footprint, None, output, mode,
                                  cval, origins, 'max')
    kernel = _get_rank_kernel(filter_size, rank, mode, footprint.shape,
                              origins, float(cval), int_type)
    return _call_kernel(kernel, input, footprint, output, None, bool)


__SHELL_SORT = '''
__device__ void sort(X *array, int size) {{
    int gap = {gap};
    while (gap > 1) {{
        gap /= 3;
        for (int i = gap; i < size; ++i) {{
            X value = array[i];
            int j = i - gap;
            while (j >= 0 && value < array[j]) {{
                array[j + gap] = array[j];
                j -= gap;
            }}
            array[j + gap] = value;
        }}
    }}
}}'''


__SELECTION_SORT = '''
__device__ void sort(X *array, int size) {
    for (int i = 0; i < size-1; ++i) {
        X min_val = array[i];
        int min_idx = i;
        for (int j = i+1; j < size; ++j) {
            X val_j = array[j];
            if (val_j < min_val) {
                min_idx = j;
                min_val = val_j;
            }
        }
        if (i != min_idx) {
            array[min_idx] = array[i];
            array[i] = min_val;
        }
    }
}'''


@cupy.util.memoize()
def _get_shell_gap(filter_size):
    gap = 1
    while gap < filter_size:
        gap = 3*gap+1
    return gap


@cupy.util.memoize(for_each_device=True)
def _get_rank_kernel(filter_size, rank, mode, wshape, origins, cval, int_type):
    # Below 225 (15x15 median filter) selection sort is 1.5-2.5x faster
    # Above, shell sort does progressively better (by 3025 (55x55) it is 9x)
    # Also tried insertion sort, which is always slower than either one
    sorter = __SELECTION_SORT if filter_size <= 255 else \
        __SHELL_SORT.format(gap=_get_shell_gap(filter_size))
    return _generate_nd_kernel(
        'rank_{}_{}'.format(filter_size, rank),
        'int iv = 0;\nX values[{}];'.format(filter_size),
        'values[iv++] = {value};',
        'sort(values, {});\ny = (Y)values[{}];'.format(filter_size, rank),
        mode, wshape, int_type, origins, cval, preamble=sorter)


def _get_output(output, input, shape=None):
    if shape is None:
        shape = input.shape
    if isinstance(output, cupy.ndarray):
        if output.shape != tuple(shape):
            raise ValueError('output shape is not correct')
    else:
        dtype = output
        if dtype is None:
            dtype = input.dtype
        output = cupy.zeros(shape, dtype)
    return output


def _fix_sequence_arg(arg, ndim, name, conv=lambda x: x):
    if isinstance(arg, str):
        return [conv(arg)] * ndim
    try:
        arg = iter(arg)
    except TypeError:
        return [conv(arg)] * ndim
    lst = [conv(x) for x in arg]
    if len(lst) != ndim:
        msg = "{} must have length equal to input rank".format(name)
        raise RuntimeError(msg)
    return lst


def _check_origin(origin, width):
    origin = int(origin)
    if (width // 2 + origin < 0) or (width // 2 + origin >= width):
        raise ValueError('invalid origin')
    return origin


def _check_mode(mode):
    if mode not in ('reflect', 'constant', 'nearest', 'mirror', 'wrap'):
        msg = 'boundary mode not supported (actual: {})'.format(mode)
        raise RuntimeError(msg)
    return mode


def _check_size_footprint_structure(ndim, size, footprint, structure,
                                    stacklevel=3, force_footprint=False):
    if structure is None and footprint is None:
        if size is None:
            raise RuntimeError("no footprint or filter size provided")
        sizes = _fix_sequence_arg(size, ndim, 'size', int)
        if force_footprint:
            return None, cupy.ones(sizes, bool), None
        return sizes, None, None
    if size is not None:
        warnings.warn("ignoring size because {} is set".format(
            'structure' if footprint is None else 'footprint'),
            UserWarning, stacklevel=stacklevel+1)

    if footprint is not None:
        footprint = cupy.array(footprint, bool, True, 'C')
        if not footprint.any():
            raise ValueError("all-zero footprint is not supported")

    if structure is None:
        if not force_footprint and footprint.all():
            return footprint.shape, None, None
        return None, footprint, None

    structure = cupy.ascontiguousarray(structure)
    if footprint is None:
        footprint = cupy.ones(structure.shape, bool)
    return None, footprint, structure


def _convert_1d_args(ndim, weights, origin, axis):
    if weights.ndim != 1 or weights.size < 1:
        raise RuntimeError('incorrect filter size')
    axis = cupy.util._normalize_axis_index(axis, ndim)
    wshape = [1]*ndim
    wshape[axis] = weights.size
    weights = weights.reshape(wshape)
    origins = [0]*ndim
    origins[axis] = _check_origin(origin, weights.size)
    return weights, tuple(origins)


def _check_nd_args(input, weights, mode, origins, wghts_name='filter weights'):
    if input.dtype.kind == 'c':
        raise TypeError('Complex type not supported')
    _check_mode(mode)
    # The integer type to use for indices in the input array
    # The indices actually use byte positions and we can't just use
    # input.nbytes since that won't tell us the number of bytes between the
    # first and last elements when the array is non-contiguous
    nbytes = sum((x-1)*abs(stride) for x, stride in
                 zip(input.shape, input.strides)) + input.dtype.itemsize
    int_type = 'int' if nbytes < (1 << 31) else 'ptrdiff_t'
    # However, weights must always be 2 GiB or less
    if weights.nbytes > (1 << 31):
        raise RuntimeError('weights must be 2 GiB or less, use FFTs instead')
    weight_dims = [x for x in weights.shape if x != 0]
    if len(weight_dims) != input.ndim:
        raise RuntimeError('{} array has incorrect shape'.format(wghts_name))
    origins = _fix_sequence_arg(origins, len(weight_dims), 'origin', int)
    for origin, width in zip(origins, weight_dims):
        _check_origin(origin, width)
    return tuple(origins), int_type


def _call_kernel(kernel, input, weights, output, structure=None,
                 weights_dtype=cupy.float64, structure_dtype=cupy.float64):
    """
    Calls a constructed ElementwiseKernel. The kernel must take an input image,
    an optional array of weights, an optional array for the structure, and an
    output array.

    weights and structure can be given as None (structure defaults to None) in
    which case they are not passed to the kernel at all. If the output is given
    as None then it will be allocated in this function.

    This function deals with making sure that the weights and structure are
    contiguous and float64 (or bool for weights that are footprints)*, that the
    output is allocated and appriopately shaped. This also deals with the
    situation that the input and output arrays overlap in memory.

    * weights is always cast to float64 or bool in order to get an output
    compatible with SciPy, though float32 might be sufficient when input dtype
    is low precision. If weights_dtype is passed as weights.dtype then no
    dtype conversion will occur. The input and output are never converted.
    """
    args = [input]
    if weights is not None:
        weights = cupy.ascontiguousarray(weights, weights_dtype)
        args.append(weights)
    if structure is not None:
        structure = cupy.ascontiguousarray(structure, structure_dtype)
        args.append(structure)
    output = _get_output(output, input)
    needs_temp = cupy.shares_memory(output, input, 'MAY_SHARE_BOUNDS')
    if needs_temp:
        output, temp = _get_output(output.dtype, input), output
    args.append(output)
    kernel(*args)
    if needs_temp:
        temp[...] = output[...]
        output = temp
    return output


def _generate_boundary_condition_ops(mode, ix, xsize):
    if mode == 'reflect':
        ops = '''
        if ({ix} < 0) {{
            {ix} = - 1 - {ix};
        }}
        {ix} %= {xsize} * 2;
        {ix} = min({ix}, 2 * {xsize} - 1 - {ix});'''.format(ix=ix, xsize=xsize)
    elif mode == 'mirror':
        ops = '''
        if ({ix} < 0) {{
            {ix} = - {ix};
        }}
        if ({xsize} == 1) {{
            {ix} = 0;
        }} else {{
            {ix} = 1 + ({ix} - 1) % (({xsize} - 1) * 2);
            {ix} = min({ix}, 2 * {xsize} - 2 - {ix});
        }}'''.format(ix=ix, xsize=xsize)
    elif mode == 'nearest':
        ops = '''
        {ix} = min(max({ix}, 0), {xsize} - 1);'''.format(ix=ix, xsize=xsize)
    elif mode == 'wrap':
        ops = '''
        if ({ix} < 0) {{
            {ix} += (1 - ({ix} / {xsize})) * {xsize};
        }}
        {ix} %= {xsize};'''.format(ix=ix, xsize=xsize)
    elif mode == 'constant':
        ops = '''
        if ({ix} >= {xsize}) {{
            {ix} = -1;
        }}'''.format(ix=ix, xsize=xsize)
    return ops


def _generate_nd_kernel(name, pre, found, post, mode, wshape, int_type,
                        origins, cval, preamble='', options=(),
                        has_weights=True, has_structure=False):
    # Currently this code uses CArray for weights but avoids using CArray for
    # the input data and instead does the indexing itself since it is faster.
    # If CArray becomes faster than follow the comments that start with
    # CArray: to switch over to using CArray for the input data as well.

    ndim = len(wshape)
    in_params = 'raw X x'
    if has_weights:
        in_params += ', raw W w'
    if has_structure:
        in_params += ', raw S s'
    out_params = 'Y y'

    inds = _generate_indices_ops(
        ndim, int_type, 'xsize_{j}',
        [' - {}'.format(wshape[j]//2 + origins[j]) for j in range(ndim)])
    # CArray: remove xstride_{j}=... from string
    sizes = ['{type} xsize_{j}=x.shape()[{j}], xstride_{j}=x.strides()[{j}];'.
             format(j=j, type=int_type) for j in range(ndim)]
    # CArray: remove expr entirely
    expr = ' + '.join(['ix_{0}'.format(j) for j in range(ndim)])

    ws_init = ws_pre = ws_post = ''
    if has_weights or has_structure:
        ws_init = 'int iws = 0;'
        if has_structure:
            ws_pre = 'S sval = s[iws];\n'
        if has_weights:
            ws_pre += 'W wval = w[iws];\nif (wval)'
        ws_post = 'iws++;'

    loops = []
    for j in range(ndim):
        if wshape[j] == 1:
            # CArray: string becomes 'inds[{j}] = ind_{j};', remove (int_)type
            loops.append('{{ {type} ix_{j} = ind_{j} * xstride_{j};'.
                         format(j=j, type=int_type))
        else:
            boundary = _generate_boundary_condition_ops(
                mode, 'ix_{}'.format(j), 'xsize_{}'.format(j))
            # CArray: last line of string becomes inds[{j}] = ix_{j};
            loops.append('''
    for (int iw_{j} = 0; iw_{j} < {wsize}; iw_{j}++)
    {{
        {type} ix_{j} = ind_{j} + iw_{j};
        {boundary}
        ix_{j} *= xstride_{j};
        '''.format(j=j, wsize=wshape[j], boundary=boundary, type=int_type))

    # CArray: string becomes 'x[inds]', no format call needed
    value = '(*(X*)&data[{expr}])'.format(expr=expr)
    if mode == 'constant':
        cond = ' || '.join(['(ix_{0} < 0)'.format(j) for j in range(ndim)])
        value = '(({cond}) ? (X){cval} : {value})'.format(
            cond=cond, cval=cval, value=value)
    found = found.format(value=value)

    # CArray: replace comment and next line in string with
    #   {type} inds[{ndim}] = {{0}};
    # and add ndim=ndim, type=int_type to format call
    operation = '''
    {sizes}
    {inds}
    // don't use a CArray for indexing (faster to deal with indexing ourselves)
    const unsigned char* data = (const unsigned char*)&x[0];
    {ws_init}
    {pre}
    {loops}
        // inner-most loop
        {ws_pre} {{
            {found}
        }}
        {ws_post}
    {end_loops}
    {post}
    '''.format(sizes='\n'.join(sizes), inds=inds, pre=pre, post=post,
               ws_init=ws_init, ws_pre=ws_pre, ws_post=ws_post,
               loops='\n'.join(loops), found=found, end_loops='}'*ndim)

    name = 'cupy_ndimage_{}_{}d_{}_w{}'.format(
        name, ndim, mode, '_'.join(['{}'.format(j) for j in wshape]))
    if int_type == 'ptrdiff_t':
        name += '_i64'
    if has_structure:
        name += '_with_structure'
    return cupy.ElementwiseKernel(in_params, out_params, operation, name,
                                  reduce_dims=False, preamble=preamble,
                                  options=options)


def _generate_indices_ops(ndim, int_type, xsize='x.shape()[{j}]', extras=None):
    if extras is None:
        extras = ('',)*ndim
    code = '{type} ind_{j} = _i % ' + xsize + '{extra}; _i /= ' + xsize + ';'
    body = [code.format(type=int_type, j=j, extra=extras[j])
            for j in range(ndim-1, 0, -1)]
    return '{type} _i = i;\n{body}\n{type} ind_0 = _i{extra};'.format(
        type=int_type, body='\n'.join(body), extra=extras[0])

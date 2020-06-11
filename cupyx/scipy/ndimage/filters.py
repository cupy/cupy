import warnings

import cupy


# ######## Convolutions and Correlations ##########

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
    return _correlate_or_convolve(input, weights, output, mode, cval, origin,
                                  False)


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
    return _correlate_or_convolve(input, weights, output, mode, cval, origins,
                                  False)


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
    return _correlate_or_convolve(input, weights, output, mode, cval, origins,
                                  False)


def _correlate_or_convolve(input, weights, output, mode, cval, origin,
                           convolution):
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


@cupy.util.memoize()
def _get_correlate_kernel(mode, wshape, int_type, origins, cval):
    return _generate_nd_kernel(
        'correlate',
        'W sum = (W)0;',
        'sum += (W){value} * wval;',
        'y = (Y)sum;',
        mode, wshape, int_type, origins, cval)


# ######## Rank-Base Filters ##########

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
    return _min_or_max_filter(input, size, footprint, output, mode, cval,
                              origin, 'min')


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
    return _min_or_max_filter(input, size, footprint, output, mode, cval,
                              origin, 'max')


def _min_or_max_filter(input, size, ftprnt, output, mode, cval, origin, func):
    sizes, ftprnt, sep = \
        _check_size_or_ftprnt(input.ndim, size, ftprnt, 3, True)

    if sep:
        # seperable filter, run as a series of 1D filters
        fltr = minimum_filter1d if func == 'min' else maximum_filter1d
        output_orig = output
        output = _get_output(output, input)
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
        if output_orig is not None and input is not output_orig:
            output_orig[...] = input
            input = output_orig
        return input

    origins, int_type = _check_nd_args(input, ftprnt, mode, origin, 'footprint')
    if ftprnt.size == 0:
        return cupy.zeros_like(input)
    kernel = _get_min_or_max_kernel(mode, ftprnt.shape, func,
                                    origins, float(cval), int_type)
    return _call_kernel(kernel, input, ftprnt, output, bool)


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
    return _max_or_min_1d(input, size, axis, output, mode, cval, origin, 'min')


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
    return _max_or_min_1d(input, size, axis, output, mode, cval, origin, 'max')


def _max_or_min_1d(input, size, axis=-1, output=None, mode="reflect", cval=0.0,
                   origin=0, func='min'):
    ftprnt = cupy.ones(size, dtype=bool)
    ftprnt, origins = _convert_1d_args(input.ndim, ftprnt, origin, axis)
    origins, int_type = _check_nd_args(input, ftprnt, mode, origins, 'footprint')
    kernel = _get_min_or_max_kernel(mode, ftprnt.shape, func, origins,
                                    float(cval), int_type, False)
    return _call_kernel(kernel, input, None, output, bool)


@cupy.util.memoize()
def _get_min_or_max_kernel(mode, wshape, func, origins, cval, int_type, has_weights=True):
    return _generate_nd_kernel(
        func, 'X value = x[i];',
        'value = {func}((X){{value}}, value);'.format(func=func),
        'y = (Y)value;', mode, wshape, int_type, origins, cval,
        has_weights=has_weights)


# ######## Utility Functions ##########

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
        msg = 'boundary mode not supported (actual: {}).'.format(mode)
        raise RuntimeError(msg)
    return mode


def _check_axis(axis, ndim):
    axis = int(axis)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError('invalid axis')
    return axis


def _check_size_or_ftprnt(ndim, size, ftprnt, stacklevel, check_sep=False):
    if (size is not None) and (ftprnt is not None):
        warnings.warn("ignoring size because footprint is set",
                      UserWarning, stacklevel=stacklevel+1)
    if ftprnt is None:
        if size is None:
            raise RuntimeError("no footprint or filter size provided")
        sizes = _fix_sequence_arg(size, ndim, 'size', int)
        if check_sep:
            return sizes, None, True
        ftprnt = cupy.ones(sizes, dtype=bool)
    else:
        ftprnt = cupy.ascontiguousarray(ftprnt, dtype=bool)
        if not ftprnt.any():
            raise ValueError("All-zero footprint is not supported.")
        if check_sep:
            if ftprnt.all():
                return ftprnt.shape, None, True
            return None, ftprnt, False
    return ftprnt


def _convert_1d_args(ndim, weights, origin, axis):
    if weights.ndim != 1 or weights.size < 1:
        raise RuntimeError('incorrect filter size')
    axis = _check_axis(axis, ndim)
    wshape = [1]*ndim
    wshape[axis] = weights.size
    weights = weights.reshape(wshape)
    origins = [0]*ndim
    origins[axis] = _check_origin(origin, weights.size)
    return weights, tuple(origins)


def _check_nd_args(input, weights, mode, origins, wghts_name='filter weights'):
    if input.dtype.kind == 'c':
        raise TypeError('Complex type not supported.')
    _check_mode(mode)
    # The integer type to use for positions in input
    # We will always assume that wsize is int32 however
    int_type = 'size_t' if input.size > 1 << 31 else 'int'
    weight_dims = [x for x in weights.shape if x != 0]
    if len(weight_dims) != input.ndim:
        raise RuntimeError('{} array has incorrect shape'.format(wghts_name))
    origins = _fix_sequence_arg(origins, len(weight_dims), 'origin', int)
    for origin, width in zip(origins, weight_dims):
        _check_origin(origin, width)
    return tuple(origins), int_type


def _call_kernel(kernel, input, weights, output,
                 weight_dtype=cupy.float64):
    """
    Calls a constructed ElementwiseKernel. The kernel must take an input image,
    an array of weights, and an output array.

    The weights are the only optional part and can be passed as None and then
    one less argument is passed to the kernel. If the output is given as None
    then it will be allocated in this function.

    This function deals with making sure that the weights are contiguous and
    float64 or bool*, that the output is allocated and appriopate shaped. This
    also deals with the situation that the input and output arrays overlap in
    memory.

    * weights is always casted to float64 or bool in order to get an output
    compatible with SciPy, though float32 might be sufficient when input dtype
    is low precision.
    """
    if weights is not None:
        weights = cupy.ascontiguousarray(weights, weight_dtype)
    output = _get_output(output, input)
    needs_temp = cupy.shares_memory(output, input, 'MAY_SHARE_BOUNDS')
    if needs_temp:
        output, temp = _get_output(output.dtype, input), output
    if weights is None:
        kernel(input, output)
    else:
        kernel(input, weights, output)
    if needs_temp:
        temp[...] = output[...]
        output = temp
    return output


# ######## Generating Elementwise Kernels ##########

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
                        has_weights=True):
    ndim = len(wshape)
    in_params = 'raw X x, raw W w'
    out_params = 'Y y'

    inds = _generate_indices_ops(
        ndim, int_type, 'xsize_{j}',
        [' - {}'.format(wshape[j]//2 + origins[j]) for j in range(ndim)])
    sizes = ['{type} xsize_{j}=x.shape()[{j}], xstride_{j}=x.strides()[{j}];'.
             format(j=j, type=int_type) for j in range(ndim)]
    cond = ' || '.join(['(ix_{0} < 0)'.format(j) for j in range(ndim)])
    expr = ' + '.join(['ix_{0}'.format(j) for j in range(ndim)])

    if has_weights:
        weights_init = 'const W* weights = (const W*)&w[0];\nint iw = 0;'
        weights_check = 'W wval = weights[iw++];\nif (wval)'
    else:
        in_params = 'raw X x'
        weights_init = weights_check = ''

    loops = []
    for j in range(ndim):
        if wshape[j] == 1:
            loops.append('{{ {type} ix_{j} = ind_{j} * xstride_{j};'.
                         format(j=j, type=int_type))
        else:
            boundary = _generate_boundary_condition_ops(
                mode, 'ix_{}'.format(j), 'xsize_{}'.format(j))
            loops.append('''
    for (int iw_{j} = 0; iw_{j} < {wsize}; iw_{j}++)
    {{
        {type} ix_{j} = ind_{j} + iw_{j};
        {boundary}
        ix_{j} *= xstride_{j};
        '''.format(j=j, wsize=wshape[j], boundary=boundary, type=int_type))

    value = '(*(X*)&data[{expr}])'.format(expr=expr)
    if mode == 'constant':
        value = '(({cond}) ? (X){cval} : {value})'.format(
            cond=cond, cval=cval, value=value)
    found = found.format(value=value)

    operation = '''
    {sizes}
    {inds}
    // don't use a CArray for indexing (faster to deal with indexing ourselves)
    const unsigned char* data = (const unsigned char*)&x[0];
    {weights_init}
    {pre}
    {loops}
        // inner-most loop
        {weights_check} {{
            {found}
        }}
    {end_loops}
    {post}
    '''.format(sizes='\n'.join(sizes), inds=inds, pre=pre, post=post,
               weights_init=weights_init, weights_check=weights_check,
               loops='\n'.join(loops), found=found, end_loops='}'*ndim)

    name = 'cupy_ndimage_{}_{}d_{}_w{}'.format(
        name, ndim, mode, '_'.join(['{}'.format(j) for j in wshape]))
    if int_type == 'size_t':
        name += '_i64'
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

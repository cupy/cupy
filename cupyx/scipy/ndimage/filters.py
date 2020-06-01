import cupy

import warnings


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
    return _get_nd_kernel('correlate',
                          'W sum = (W)0;',
                          'sum += (W){value} * wval;',
                          'y = (Y)sum;',
                          mode, wshape, int_type, origins, cval)


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
    if hasattr(arg, '__iter__') and not isinstance(arg, str):
        lst = [conv(x) for x in arg]
        if len(lst) != ndim:
            msg = "{} must have length equal to input rank".format(name)
            raise RuntimeError(msg)
    else:
        lst = [conv(arg)] * ndim
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


# def _check_axis(axis, ndim):
#     axis = int(axis)
#     if axis < 0:
#         axis += ndim
#     if axis < 0 or axis >= ndim:
#         raise ValueError('invalid axis')
#     return axis


def _convert_1d_args(ndim, weights, origin, axis):
    if weights.ndim != 1 or weights.size < 1:
        raise RuntimeError('incorrect filter size')
    # axis = _check_axis(axis, ndim)
    axis = cupy.util._normalize_axis_index(axis, ndim)
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


def _get_nd_kernel(name, pre, found, post, mode, wshape, int_type,
                   origins, cval, preamble='', options=(), has_weights=True):
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


def _generate_correlete_kernel(ndim, mode, cval, xshape, wshape, origin):
    in_params = 'raw X x, raw W w'
    out_params = 'Y y'

    ops = []
    ops.append('X* x_data = (X*)&(x[0]);')
    ops.append('W* w_data = (W*)&(w[0]);')
    ops.append('const int sx_{} = 1;'.format(ndim-1))
    for j in range(ndim-1, 0, -1):
        ops.append('int sx_{jm} = sx_{j} * {xsize_j};'.
                   format(jm=j-1, j=j, xsize_j=xshape[j]))
    ops.append('int _i = i;')
    for j in range(ndim-1, -1, -1):
        ops.append('int cx_{j} = _i % {xsize} - ({wsize} / 2) - ({origin});'
                   .format(j=j, xsize=xshape[j], wsize=wshape[j],
                           origin=origin[j]))
        if (j > 0):
            ops.append('_i /= {xsize};'.format(xsize=xshape[j]))
    ops.append('W sum = (W)0;')
    ops.append('int iw = 0;')

    for j in range(ndim):
        ops.append('''
    for (int iw_{j} = 0; iw_{j} < {wsize}; iw_{j}++)
    {{
        int ix_{j} = cx_{j} + iw_{j};'''.format(j=j, wsize=wshape[j]))
        ixvar = 'ix_{}'.format(j)
        ops.append(_generate_boundary_condition_ops(mode, ixvar, xshape[j]))
        ops.append('        ix_{j} *= sx_{j};'.format(j=j))

    ops.append('''
        W wval = w_data[iw];
        if (wval == (W)0) {{
            iw += 1;
            continue;
        }}''')
    _cond = ' || '.join(['(ix_{0} < 0)'.format(j) for j in range(ndim)])
    _expr = ' + '.join(['ix_{0}'.format(j) for j in range(ndim)])
    ops.append('''
        if ({cond}) {{
            sum += (W){cval} * wval;
        }} else {{
            int ix = {expr};
            sum += (W)x_data[ix] * wval;
        }}
        iw += 1;'''.format(cond=_cond, expr=_expr, cval=cval))

    ops.append('} ' * ndim)
    ops.append('y = (Y)sum;')
    operation = '\n'.join(ops)

    name = 'cupy_ndimage_correlate_{}d_{}_x{}_w{}'.format(
        ndim, mode, '_'.join(['{}'.format(j) for j in xshape]),
        '_'.join(['{}'.format(j) for j in wshape]))
    return in_params, out_params, operation, name


@cupy.util.memoize()
def _get_correlete_kernel(ndim, mode, cval, xshape, wshape, origin):
    # weights is always casted to float64 in order to get an output compatible
    # with SciPy, thought float32 might be sufficient when input dtype is low
    # precision.
    in_params, out_params, operation, name = _generate_correlete_kernel(
        ndim, mode, cval, xshape, wshape, origin)
    return cupy.ElementwiseKernel(in_params, out_params, operation, name)


def _normalize_sequence(x, ndim):
    if not hasattr(x, '__getitem__') or isinstance(x, str):
        return [x] * ndim
    else:
        return list(x)


def _min_or_max_filter(input, size, footprint, structure, output, mode,
                       cval, origin, minimum):
    if input.dtype.kind == 'c':
        raise TypeError('Complex type not supported.')
    if (size is not None) and (footprint is not None):
        warnings.warn("ignoring size because footprint is set", UserWarning,
                      stacklevel=3)
    if structure is None:
        if footprint is None:
            if size is None:
                raise RuntimeError("no footprint provided")
            else:
                footprint = cupy.ones(_normalize_sequence(size, input.ndim),
                                      bool)
        else:
            footprint = cupy.array(footprint, dtype=bool, order='C')
            if not footprint.any():
                raise ValueError("All-zero footprint is not supported")
    else:
        structure = cupy.array(structure, dtype=cupy.float64, order='C')
        if footprint is None:
            footprint = cupy.ones(structure.shape, bool)
        else:
            footprint = cupy.array(footprint, dtype=bool, order='C')
    modes = _normalize_sequence(mode, input.ndim)
    for mode in modes:
        if mode not in ('reflect', 'constant', 'nearest', 'mirror', 'wrap'):
            msg = 'boundary mode not supported (actual: {}).'.format(mode)
            raise RuntimeError(msg)
    origins = _normalize_sequence(origin, input.ndim)
    for origin, lenfp in zip(origins, footprint.shape):
        if (lenfp // 2 + origin < 0) or (lenfp // 2 + origin >= lenfp):
            raise ValueError('invalid origin')

    kernel = _get_min_or_max_kernel(
        input.ndim, tuple(footprint.shape), structure is not None,
        tuple(modes), cval, tuple(origins), minimum)

    input = cupy.ascontiguousarray(input)
    input_shape = cupy.array(input.shape, dtype=cupy.int32)
    if structure is None:
        structure = cupy.empty(0, dtype=cupy.float64)
    output = _get_output(output, input)
    return kernel(input, input_shape, footprint, structure, output)


def _generate_min_or_max_kernel(ndim, fp_shape, use_structure, modes, cval,
                                origins, minimum):
    in_params = 'raw X x, raw int32 x_shape, raw bool fp, raw S st'
    out_params = 'Y y'

    ops = []
    ops.append('const int sx_{} = 1;'.format(ndim-1))
    for j in range(ndim-1, 0, -1):
        ops.append('int sx_{jm} = sx_{j} * x_shape[{j}];'.
                   format(jm=j-1, j=j))

    ops.append('int remain = i;')
    for j in range(ndim-1, -1, -1):
        ops.append('int cx_{j} = remain % x_shape[{j}] - ({fpsize} / 2)'
                   ' - ({origin});'
                   .format(j=j, fpsize=fp_shape[j], origin=origins[j]))
        if (j > 0):
            ops.append('remain /= x_shape[{j}];'.format(j=j))
    ops.append('S ret = (S)0;')
    ops.append('int first = 1;')
    ops.append('int ifp = 0;')

    for j in range(ndim):
        ops.append('''
    for (int ifp_{j} = 0; ifp_{j} < {fpsize}; ifp_{j}++)
    {{
        int ix_{j} = cx_{j} + ifp_{j};'''.format(j=j,
                                                 fpsize=fp_shape[j]))
        mode = modes[j]
        ixvar = 'ix_{}'.format(j)
        xsize = 'x_shape[{}]'.format(j)
        ops.append(_generate_boundary_condition_ops(mode, ixvar, xsize))
        ops.append('        ix_{j} *= sx_{j};'.format(j=j))

    ops.append('if (fp[ifp]) {')
    cond = ' || '.join(['(ix_{} < 0)'.format(j) for j in range(ndim)])
    ix = ' + '.join(['ix_{}'.format(j) for j in range(ndim)])
    ops.append('''
        S val;
        if ({cond}) {{
            val = (S){cval};
        }} else {{
            val = (S)x[{ix}];
        }}'''.format(cond=cond, ix=ix, cval=cval))
    if use_structure:
        if minimum:
            ops.append('        val -= st[ifp];')
        else:
            ops.append('        val += st[ifp];')
    if minimum:
        filter_type = 'min'
    else:
        filter_type = 'max'
    ops.append('''
        if (first) {{
            first = 0;
            ret = val;
        }} else {{
            ret = {filter_type}(ret, val);
        }}'''.format(filter_type=filter_type))
    ops.append('}')
    ops.append('ifp++;')

    ops.append('} ' * ndim)
    ops.append('y = (Y)ret;')
    operation = '\n'.join(ops)

    name = 'cupyx_nd_{}_{}d_fp_{}_md_{}'.format(
        filter_type, ndim,
        '_'.join(['{}'.format(fp_shape[j]) for j in range(ndim)]),
        '_'.join(['{}'.format(modes[j]) for j in range(ndim)])
    )
    if use_structure:
        name += '_wt_structure'
    else:
        name += '_wo_structure'

    return in_params, out_params, operation, name


@cupy.util.memoize()
def _get_min_or_max_kernel(ndim, fp_shape, use_structure, modes, cval,
                           origins, minimum):
    in_params, out_params, operation, name = _generate_min_or_max_kernel(
        ndim, fp_shape, use_structure, modes, cval, origins, minimum)
    return cupy.ElementwiseKernel(in_params, out_params, operation, name)


def minimum_filter(input, size=None, footprint=None, output=None,
                   mode='reflect', cval=0.0, origin=0):
    """Calculates a multi-dimensional minimum filter.

    Args:
        input (cupy.ndarray): The input array.
        size (tuple of ints): ```size``` specifies the shape that is taken from
            the input array, at every element position, to define the input to
            the filter function.
        footprint (array of ints): ```footprint``` specifies the shape, but
            also which elements within the shape will get passed to the filter
             function.
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
        cupy.ndarray: The result of minimum filter.

    .. seealso:: :func:`scipy.ndimage.minimum_filter`
    """
    return _min_or_max_filter(input, size, footprint, None, output, mode, cval,
                              origin, True)


def maximum_filter(input, size=None, footprint=None, output=None,
                   mode='reflect', cval=0.0, origin=0):
    """Calculates a multi-dimensional maximum filter.

    Args:
        input (cupy.ndarray): The input array.
        size (tuple of ints): ```size``` specifies the shape that is taken from
            the input array, at every element position, to define the input to
            the filter function.
        footprint (array of ints): ```footprint``` specifies the shape, but
            also which elements within the shape will get passed to the filter
             function.
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
        cupy.ndarray: The result of maximum filter.

    .. seealso:: :func:`scipy.ndimage.maximum_filter`
    """
    return _min_or_max_filter(input, size, footprint, None, output, mode, cval,
                              origin, False)

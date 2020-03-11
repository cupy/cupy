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


def _correlate_or_convolve(input, weights, output, mode, cval, origin,
                           convolution):
    origins, int_type = _check_nd_args(input, weights, mode, origin)
    if weights.size == 0:
        return cupy.zeros_like(input)
    if convolution:
        weights = weights[tuple([slice(None, None, -1)] * weights.ndim)]
        for i, wsize in enumerate(weights.shape):
            origins[i] = -origins[i]
            if wsize % 2 == 0:
                origins[i] -= 1
    kernel = _get_correlate_kernel(mode, weights.shape, int_type,
                                   tuple(origins), cval)
    return _call_kernel(kernel, input, weights, output)


@cupy.util.memoize()
def _get_correlate_kernel(mode, wshape, int_type, origins, cval):
    return _get_nd_kernel('correlate',
                          'W sum = (W)0;',
                          'sum += (W){value} * w[iw];',
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


def _check_origins(origins, shape):
    origins = _fix_sequence_arg(origins, len(shape), 'origin', int)
    for origin, width in zip(origins, shape):
        _check_origin(origin, width)
    return origins


def _check_mode(mode):
    if mode not in ('reflect', 'constant', 'nearest', 'mirror', 'wrap'):
        msg = 'boundary mode not supported (actual: {}).'.format(mode)
        raise RuntimeError(msg)
    return mode


def _get_int_type(size):
    # The integer type to use for positions in input
    # We will always assume that wsize is int32 however
    return 'size_t' if size > 1 << 31 else 'int'


def _check_args(input, mode):
    if input.dtype.kind == 'c':
        raise TypeError('Complex type not supported.')
    _check_mode(mode)
    int_type = _get_int_type(input.size)
    return int_type


def _check_nd_args(input, weights, mode, origin, wghts_name='filter weights'):
    int_type = _check_args(input, mode)
    weight_dims = [x for x in weights.shape if x != 0]
    if len(weight_dims) != input.ndim:
        raise RuntimeError('{} array has incorrect shape'.format(wghts_name))
    origins = _check_origins(origin, weight_dims)
    return origins, int_type


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
                   origins, cval, preamble='', options=()):
    ndim = len(wshape)
    in_params = 'raw X x, raw W w'
    out_params = 'Y y'

    inds = _generate_indices_ops(
        ndim, int_type,
        [' - {}'.format(wshape[j]//2 + origins[j]) for j in range(ndim)])
    sizes = ['{type} xsize_{j}=x.shape()[{j}], xstride_{j}=x.strides()[{j}];'.
             format(j=j, type=int_type) for j in range(ndim)]
    cond = ' || '.join(['(ix_{0} < 0)'.format(j) for j in range(ndim)])
    expr = ' + '.join(['ix_{0}'.format(j) for j in range(ndim)])

    loops = []
    for j in range(ndim):
        boundary = _generate_boundary_condition_ops(mode, 'ix_{}'.format(j),
                                                    'xsize_{}'.format(j))
        loops.append('''
    for (int iw_{j} = 0; iw_{j} < {wsize}; iw_{j}++)
    {{
        {type} ix_{j} = ind_{j} + iw_{j};
        {boundary}
        ix_{j} *= xstride_{j};
        '''.format(j=j, wsize=wshape[j], boundary=boundary, type=int_type))

    value = '(({cond}) ? (X){cval} : *(X*)&data[{expr}])'.format(
        cond=cond, cval=cval, expr=expr)
    found = found.format(value=value)

    operation = '''
    {sizes}
    {inds}
    const unsigned char* data = (const unsigned char*)&x[0];
    const W* weights = (const W*)&w[0];
    int iw = 0;
    {pre}
    {loops}
        // inner-most loop
        if (weights[iw]) {{
            {found}
        }}
        iw += 1;
    {end_loops}
    {post}
    '''.format(sizes='\n'.join(sizes), inds=inds, pre=pre, post=post,
               loops='\n'.join(loops), found=found, end_loops='}'*ndim)

    name = 'cupy_ndimage_{}_{}d_{}_w{}'.format(
        name, ndim, mode, '_'.join(['{}'.format(j) for j in wshape]))
    if int_type == 'size_t':
        name += '_i64'
    return cupy.ElementwiseKernel(in_params, out_params, operation, name,
                                  reduce_dims=False, preamble=preamble,
                                  options=options)


def _generate_indices_ops(ndim, int_type, extras=None):
    if extras is None:
        extras = ('',)*ndim
    body = ['{type} ind_{dim} = _i % x.shape()[{dim}]{extra}; '
            '_i /= x.shape()[{dim}];'.format(
                type=int_type, dim=dim, extra=extras[dim])
            for dim in range(ndim-1, 0, -1)]
    return '{type} _i = i;\n{body}\n{type} ind_0 = _i{extra};'.format(
        type=int_type, body='\n'.join(body), extra=extras[0])

import cupy
from cupy import util

import warnings


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


def _correlate_or_convolve(input, weights, output, mode, cval, origin,
                           convolution):
    if input.dtype.kind == 'c':
        raise TypeError('Complex type not supported.')
    if not hasattr(origin, '__getitem__'):
        origin = [origin, ] * input.ndim
    else:
        origin = list(origin)
    wshape = [ii for ii in weights.shape if ii > 0]
    if len(wshape) != input.ndim:
        raise RuntimeError('filter weights array has incorrect shape.')
    if convolution:
        weights = weights[tuple([slice(None, None, -1)] * weights.ndim)]
        for ii in range(len(origin)):
            origin[ii] = -origin[ii]
            if weights.shape[ii] % 2 == 0:
                origin[ii] -= 1
    for _origin, lenw in zip(origin, wshape):
        if (lenw // 2 + _origin < 0) or (lenw // 2 + _origin >= lenw):
            raise ValueError('invalid origin')
    if mode not in ('reflect', 'constant', 'nearest', 'mirror', 'wrap'):
        msg = 'boundary mode not supported (actual: {}).'.format(mode)
        raise RuntimeError(msg)

    output = _get_output(output, input)
    if weights.size == 0:
        return output
    input = cupy.ascontiguousarray(input)
    weights = cupy.ascontiguousarray(weights, cupy.float64)
    return _get_correlete_kernel(
        input.ndim, mode, cval, input.shape, tuple(wshape), tuple(origin))(
        input, weights, output)


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


@util.memoize()
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


@util.memoize()
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

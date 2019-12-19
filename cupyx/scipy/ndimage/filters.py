import numpy

import cupy
from cupy import util


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
    if input.dtype in (numpy.complex64, numpy.complex128, numpy.complex256):
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

    # The kernel needs only the non-zero kernel values and their coordinates.
    # This allows us to use a single for loop to compute the ndim convolution.
    # The loop will be over only the the non-zero entries of the filter.
    wlocs = cupy.where(weights)
    wvals = weights[wlocs]     # (nnz,) array of non-zero values
    wlocs = cupy.stack(wlocs)  # (ndim, nnz) array of indices for these values

    return _get_correlete_kernel(
        input.ndim, mode, cval, input.shape, weights.shape, wvals.size,
        tuple(origin))(input, wlocs, wvals, output)


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


def _generate_correlete_kernel(mode, cval, xshape, wshape, nnz, origin):
    in_params = "raw X x, raw I wlocs, raw W wvals"
    out_params = "Y y"

    ndim = len(wshape)

    ops = []
    ops.append('const int sx_{} = 1;'.format(ndim - 1))
    for j in range(ndim - 1, 0, -1):
        ops.append(
            'int sx_{jm} = sx_{j} * {xsize_j};'.format(
                jm=j - 1, j=j, xsize_j=xshape[j]
            )
        )
    ops.append('int _i = i;')
    for j in range(ndim - 1, -1, -1):
        ops.append(
            'int cx_{j} = _i % {xsize} - ({wsize} / 2) - ({origin});'.format(
                j=j, xsize=xshape[j], wsize=wshape[j], origin=origin[j]
            )
        )
        if j > 0:
            ops.append('_i /= {xsize};'.format(xsize=xshape[j]))
    ops.append('W sum = (W)0;')
    ops.append(
        """
        for (int iw = 0; iw < {nnz}; iw++)
        {{
        """.format(nnz=nnz)
    )
    for j in range(ndim):
        ops.append(
            """
                int iw_{j} = wlocs[iw + {j} * {nnz}];
                int ix_{j} = cx_{j} + iw_{j};""".format(j=j, nnz=nnz)
        )
        ixvar = 'ix_{}'.format(j)
        ops.append(_generate_boundary_condition_ops(mode, ixvar, xshape[j]))
        ops.append('        ix_{j} *= sx_{j};'.format(j=j))
    _cond = ' || '.join(['(ix_{0} < 0)'.format(j) for j in range(ndim)])
    _expr = ' + '.join(['ix_{0}'.format(j) for j in range(ndim)])
    ops.append(
        """
        if ({cond}) {{
            sum += (W){cval} * wvals[iw];
        }} else {{
            int ix = {expr};
            sum += (W)x[ix] * wvals[iw];
        }}
        """.format(
            cond=_cond, expr=_expr, cval=cval,
        )
    )

    ops.append('}')
    ops.append('y = (Y)sum;')
    operation = '\n'.join(ops)

    name = 'cupy_ndimage_correlate_{}d_{}_x{}_w{}_nnz{}'.format(
        ndim,
        mode,
        '_'.join(['{}'.format(j) for j in xshape]),
        '_'.join(['{}'.format(j) for j in wshape]),
        nnz,
    )
    return in_params, out_params, operation, name


@util.memoize()
def _get_correlete_kernel(ndim, mode, cval, xshape, wshape, nnz, origin):
    # weights is always casted to float64 in order to get an output compatible
    # with SciPy, thought float32 might be sufficient when input dtype is low
    # precision.
    in_params, out_params, operation, name = _generate_correlete_kernel(
        mode, cval, xshape, wshape, nnz, origin)
    return cupy.ElementwiseKernel(in_params, out_params, operation, name)

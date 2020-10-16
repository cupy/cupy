import cupy


def _is_integer_output(output, input):
    if output is None:
        return input.dtype.kind in 'iu'
    elif isinstance(output, cupy.ndarray):
        return output.dtype.kind in 'iu'
    return cupy.dtype(output).kind in 'iu'


def _check_cval(mode, cval, integer_output):
    if mode == 'constant' and integer_output and not cupy.isfinite(cval):
        raise NotImplementedError("Non-finite cval is not supported for "
                                  "outputs with integer dtype.")


def _get_output(output, input, shape=None):
    if not isinstance(output, cupy.ndarray):
        return cupy.zeros_like(input, shape=shape, dtype=output, order='C')
    if output.shape != (input.shape if shape is None else tuple(shape)):
        raise ValueError('output shape is not correct')
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


def _get_inttype(input):
    # The integer type to use for indices in the input array
    # The indices actually use byte positions and we can't just use
    # input.nbytes since that won't tell us the number of bytes between the
    # first and last elements when the array is non-contiguous
    nbytes = sum((x-1)*abs(stride) for x, stride in
                 zip(input.shape, input.strides)) + input.dtype.itemsize
    return 'int' if nbytes < (1 << 31) else 'ptrdiff_t'


def _generate_boundary_condition_ops(mode, ix, xsize):
    if mode == 'reflect':
        ops = '''
        if ({ix} < 0) {{
            {ix} = - 1 -{ix};
        }}
        {ix} %= {xsize} * 2;
        {ix} = min({ix}, 2 * {xsize} - 1 - {ix});'''.format(ix=ix, xsize=xsize)
    elif mode == 'mirror':
        ops = '''
        if ({xsize} == 1) {{
            {ix} = 0;
        }} else {{
            if ({ix} < 0) {{
                {ix} = -{ix};
            }}
            {ix} = 1 + ({ix} - 1) % (({xsize} - 1) * 2);
            {ix} = min({ix}, 2 * {xsize} - 2 - {ix});
        }}'''.format(ix=ix, xsize=xsize)
    elif mode == 'nearest':
        ops = '''
        {ix} = min(max({ix}, 0), {xsize} - 1);'''.format(ix=ix, xsize=xsize)
    elif mode == 'wrap':
        ops = '''
        {ix} %= {xsize};
        if ({ix} < 0) {{
            {ix} += {xsize};
        }}'''.format(ix=ix, xsize=xsize)
    elif mode == 'constant':
        ops = '''
        if ({ix} >= {xsize}) {{
            {ix} = -1;
        }}'''.format(ix=ix, xsize=xsize)
    return ops


def _generate_indices_ops(ndim, int_type, offsets):
    code = '{type} ind_{j} = _i % ysize_{j} - {offset}; _i /= ysize_{j};'
    body = [code.format(type=int_type, j=j, offset=offsets[j])
            for j in range(ndim-1, 0, -1)]
    return '{type} _i = i;\n{body}\n{type} ind_0 = _i - {offset};'.format(
        type=int_type, body='\n'.join(body), offset=offsets[0])

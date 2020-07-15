import cupy


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


def _get_inttype(input):
    # The integer type to use for indices in the input array
    # The indices actually use byte positions and we can't just use
    # input.nbytes since that won't tell us the number of bytes between the
    # first and last elements when the array is non-contiguous
    nbytes = sum((x-1)*abs(stride) for x, stride in
                 zip(input.shape, input.strides)) + input.dtype.itemsize
    return 'int' if nbytes < (1 << 31) else 'ptrdiff_t'


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
        import warnings
        warnings.warn("ignoring size because {} is set".format(
            'structure' if footprint is None else 'footprint'),
            UserWarning, stacklevel=stacklevel+1)

    if footprint is not None:
        footprint = cupy.array(footprint, bool, True, 'C')
        if not footprint.any():
            raise ValueError("all-zero footprint is not supported")

    if structure is None:
        if not force_footprint and footprint.all():
            if footprint.ndim != ndim:
                raise RuntimeError("size must have length equal to input rank")
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


def _check_nd_args(input, weights, mode, origin, wghts_name='filter weights'):
    if input.dtype.kind == 'c':
        raise TypeError('Complex type not supported')
    _check_mode(mode)
    # Weights must always be 2 GiB or less
    if weights.nbytes > (1 << 31):
        raise RuntimeError('weights must be 2 GiB or less, use FFTs instead')
    weight_dims = [x for x in weights.shape if x != 0]
    if len(weight_dims) != input.ndim:
        raise RuntimeError('{} array has incorrect shape'.format(wghts_name))
    origins = _fix_sequence_arg(origin, len(weight_dims), 'origin', int)
    for origin, width in zip(origins, weight_dims):
        _check_origin(origin, width)
    return tuple(origins), _get_inttype(input)


def _run_1d_filters(filters, input, args, output, mode, cval, origin=0):
    """
    Runs a series of 1D filters forming an nd filter. The filters must be a
    list of callables that take input, arg, axis, output, mode, cval, origin.
    The args is a list of values that are passed for the arg value to the
    filter. Individual filters can be None causing that axis to be skipped.
    """
    output_orig = output
    output = _get_output(output, input)
    modes = _fix_sequence_arg(mode, input.ndim, 'mode', _check_mode)
    origins = _fix_sequence_arg(origin, input.ndim, 'origin', int)
    n_filters = sum(filter is not None for filter in filters)
    if n_filters == 0:
        output[...] = input[...]
        return output
    # We can't operate in-place efficiently, so use a 2-buffer system
    temp = _get_output(output.dtype, input) if n_filters > 1 else None
    first = True
    iterator = zip(filters, args, modes, origins)
    for axis, (fltr, arg, mode, origin) in enumerate(iterator):
        if fltr is None:
            continue
        fltr(input, arg, axis, output, mode, cval, origin)
        input, output = output, temp if first else input
    if isinstance(output_orig, cupy.ndarray) and input is not output_orig:
        output_orig[...] = input
        input = output_orig
    return input


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


_CAST_FUNCTION = """
// Implements a casting function to make it compatible with scipy
// Use like cast<to_type>(value)
// It's actually really simple - most of this is <type_traits>

// Small bit of <type_traits> which cannot be imported in NVRTC
// Requires compiling with --std=c++11 or higher
template<bool B, class T=void> struct enable_if {};
template<class T> struct enable_if<true, T> { typedef T type; };
template<class T> struct remove_const          { typedef T type; };
template<class T> struct remove_const<const T> { typedef T type; };
template<class T> struct remove_volatile             { typedef T type; };
template<class T> struct remove_volatile<volatile T> { typedef T type; };
template<class T> struct remove_cv {
  typedef typename remove_volatile<typename remove_const<T>::type>::type type;
};
template<class T, T v>
struct integral_constant { static constexpr T value = v; };
typedef integral_constant<bool, true> true_type;
typedef integral_constant<bool, false> false_type;
template<class T> struct __is_fp : public false_type {};
template<>        struct __is_fp<float16> : public true_type {};
template<>        struct __is_fp<float> : public true_type {};
template<>        struct __is_fp<double> : public true_type {};
template<>        struct __is_fp<long double> : public true_type {};
template<class T> struct is_floating_point
    : public __is_fp<typename remove_cv<T>::type> {};
template<class T> struct is_signed : integral_constant<bool, (T)(-1)<0> {};

template <typename B, typename A>
__device__
typename enable_if<!is_floating_point<A>::value||is_signed<B>::value, B>::type
cast(A a) { return (B)a; }

template <typename B, typename A>
__device__
typename enable_if<is_floating_point<A>::value&&!is_signed<B>::value, B>::type
cast(A a) { return (a >= 0) ? (B)a : -(B)(-a); }


"""


def _generate_nd_kernel(name, pre, found, post, mode, wshape, int_type,
                        origins, cval, ctype='X', preamble='', options=(),
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

    # CArray: remove xstride_{j}=... from string
    sizes = ['{type} xsize_{j}=x.shape()[{j}], xstride_{j}=x.strides()[{j}];'.
             format(j=j, type=int_type) for j in range(ndim)]
    inds = _generate_indices_ops(ndim, int_type,
                                 [x//2 + o for x, o in zip(wshape, origins)])
    # CArray: remove expr entirely
    expr = ' + '.join(['ix_{}'.format(j) for j in range(ndim)])

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
        cond = ' || '.join(['(ix_{} < 0)'.format(j) for j in range(ndim)])
        value = '(({cond}) ? cast<{ctype}>({cval}) : {value})'.format(
            cond=cond, ctype=ctype, cval=cval, value=value)
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
        name, ndim, mode, '_'.join(['{}'.format(x) for x in wshape]))
    if int_type == 'ptrdiff_t':
        name += '_i64'
    if has_structure:
        name += '_with_structure'
    preamble = _CAST_FUNCTION + preamble
    options += ('--std=c++11',)

    return cupy.ElementwiseKernel(in_params, out_params, operation, name,
                                  reduce_dims=False, preamble=preamble,
                                  options=options)


def _generate_indices_ops(ndim, int_type, offsets):
    code = '{type} ind_{j} = _i % xsize_{j} - {offset}; _i /= xsize_{j};'
    body = [code.format(type=int_type, j=j, offset=offsets[j])
            for j in range(ndim-1, 0, -1)]
    return '{type} _i = i;\n{body}\n{type} ind_0 = _i - {offset};'.format(
        type=int_type, body='\n'.join(body), offset=offsets[0])

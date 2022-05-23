import warnings

import numpy
import cupy

from cupy_backends.cuda.api import runtime
from cupy import _core
from cupy._core import internal
from cupyx.scipy.ndimage import _util


def _origins_to_offsets(origins, w_shape):
    return tuple(x//2+o for x, o in zip(w_shape, origins))


def _check_size_footprint_structure(ndim, size, footprint, structure,
                                    stacklevel=3, force_footprint=False):
    if structure is None and footprint is None:
        if size is None:
            raise RuntimeError("no footprint or filter size provided")
        sizes = _util._fix_sequence_arg(size, ndim, 'size', int)
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
    axis = internal._normalize_axis_index(axis, ndim)
    w_shape = [1]*ndim
    w_shape[axis] = weights.size
    weights = weights.reshape(w_shape)
    origins = [0]*ndim
    origins[axis] = _util._check_origin(origin, weights.size)
    return weights, tuple(origins)


def _check_nd_args(input, weights, mode, origin, wghts_name='filter weights'):
    _util._check_mode(mode)
    # Weights must always be less than 2 GiB
    if weights.nbytes >= (1 << 31):
        raise RuntimeError('weights must be 2 GiB or less, use FFTs instead')
    weight_dims = [x for x in weights.shape if x != 0]
    if len(weight_dims) != input.ndim:
        raise RuntimeError('{} array has incorrect shape'.format(wghts_name))
    origins = _util._fix_sequence_arg(origin, len(weight_dims), 'origin', int)
    for origin, width in zip(origins, weight_dims):
        _util._check_origin(origin, width)
    return tuple(origins), _util._get_inttype(input)


def _run_1d_filters(filters, input, args, output, mode, cval, origin=0):
    """
    Runs a series of 1D filters forming an nd filter. The filters must be a
    list of callables that take input, arg, axis, output, mode, cval, origin.
    The args is a list of values that are passed for the arg value to the
    filter. Individual filters can be None causing that axis to be skipped.
    """
    output_orig = output
    output = _util._get_output(output, input)
    modes = _util._fix_sequence_arg(mode, input.ndim, 'mode',
                                    _util._check_mode)
    # for filters, "wrap" is a synonym for "grid-wrap".
    modes = ['grid-wrap' if m == 'wrap' else m for m in modes]
    origins = _util._fix_sequence_arg(origin, input.ndim, 'origin', int)
    n_filters = sum(filter is not None for filter in filters)
    if n_filters == 0:
        _core.elementwise_copy(input, output)
        return output
    # We can't operate in-place efficiently, so use a 2-buffer system
    temp = _util._get_output(output.dtype, input) if n_filters > 1 else None
    first = True
    iterator = zip(filters, args, modes, origins)
    for axis, (fltr, arg, mode, origin) in enumerate(iterator):
        if fltr is None:
            continue
        fltr(input, arg, axis, output, mode, cval, origin)
        input, output = output, temp if first else input
        first = False
    if isinstance(output_orig, cupy.ndarray) and input is not output_orig:
        _core.elementwise_copy(input, output_orig)
        input = output_orig
    return input


def _call_kernel(kernel, input, weights, output, structure=None,
                 weights_dtype=numpy.float64, structure_dtype=numpy.float64):
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
    complex_output = input.dtype.kind == 'c'
    if weights is not None:
        weights = cupy.ascontiguousarray(weights, weights_dtype)
        complex_output = complex_output or weights.dtype.kind == 'c'
        args.append(weights)
    if structure is not None:
        structure = cupy.ascontiguousarray(structure, structure_dtype)
        args.append(structure)
    output = _util._get_output(output, input, None, complex_output)
    needs_temp = cupy.shares_memory(output, input, 'MAY_SHARE_BOUNDS')
    if needs_temp:
        output, temp = _util._get_output(output.dtype, input), output
    args.append(output)
    kernel(*args)
    if needs_temp:
        _core.elementwise_copy(temp, output)
        output = temp
    return output


if runtime.is_hip:
    includes = r'''
// workaround for HIP: line begins with #include
#include <cupy/math_constants.h>\n
'''
else:
    includes = r'''
#include <type_traits>  // let Jitify handle this
#include <cupy/math_constants.h>

template<> struct std::is_floating_point<float16> : std::true_type {};
template<> struct std::is_signed<float16> : std::true_type {};
template<class T> struct std::is_signed<complex<T>> : std::is_signed<T> {};
'''


_CAST_FUNCTION = """
// Implements a casting function to make it compatible with scipy
// Use like cast<to_type>(value)
template <class B, class A>
__device__ __forceinline__
typename std::enable_if<(!std::is_floating_point<A>::value
                         || std::is_signed<B>::value), B>::type
cast(A a) { return (B)a; }

template <class B, class A>
__device__ __forceinline__
typename std::enable_if<(std::is_floating_point<A>::value
                         && (!std::is_signed<B>::value)), B>::type
cast(A a) { return (a >= 0) ? (B)a : -(B)(-a); }

template <class T>
__device__ __forceinline__ bool nonzero(T x) { return x != static_cast<T>(0); }
"""


def _generate_nd_kernel(name, pre, found, post, mode, w_shape, int_type,
                        offsets, cval, ctype='X', preamble='', options=(),
                        has_weights=True, has_structure=False, has_mask=False,
                        binary_morphology=False, all_weights_nonzero=False):
    # Currently this code uses CArray for weights but avoids using CArray for
    # the input data and instead does the indexing itself since it is faster.
    # If CArray becomes faster than follow the comments that start with
    # CArray: to switch over to using CArray for the input data as well.

    ndim = len(w_shape)
    in_params = 'raw X x'
    if has_weights:
        in_params += ', raw W w'
    if has_structure:
        in_params += ', raw S s'
    if has_mask:
        in_params += ', raw M mask'
    out_params = 'Y y'

    # for filters, "wrap" is a synonym for "grid-wrap"
    mode = 'grid-wrap' if mode == 'wrap' else mode

    # CArray: remove xstride_{j}=... from string
    size = ('%s xsize_{j}=x.shape()[{j}], ysize_{j} = _raw_y.shape()[{j}]'
            ', xstride_{j}=x.strides()[{j}];' % int_type)
    sizes = [size.format(j=j) for j in range(ndim)]
    inds = _util._generate_indices_ops(ndim, int_type, offsets)
    # CArray: remove expr entirely
    expr = ' + '.join(['ix_{}'.format(j) for j in range(ndim)])

    ws_init = ws_pre = ws_post = ''
    if has_weights or has_structure:
        ws_init = 'int iws = 0;'
        if has_structure:
            ws_pre = 'S sval = s[iws];\n'
        if has_weights:
            ws_pre += 'W wval = w[iws];\n'
            if not all_weights_nonzero:
                ws_pre += 'if (nonzero(wval))'
        ws_post = 'iws++;'

    loops = []
    for j in range(ndim):
        if w_shape[j] == 1:
            # CArray: string becomes 'inds[{j}] = ind_{j};', remove (int_)type
            loops.append('{{ {type} ix_{j} = ind_{j} * xstride_{j};'.
                         format(j=j, type=int_type))
        else:
            boundary = _util._generate_boundary_condition_ops(
                mode, 'ix_{}'.format(j), 'xsize_{}'.format(j), int_type)
            # CArray: last line of string becomes inds[{j}] = ix_{j};
            loops.append('''
    for (int iw_{j} = 0; iw_{j} < {wsize}; iw_{j}++)
    {{
        {type} ix_{j} = ind_{j} + iw_{j};
        {boundary}
        ix_{j} *= xstride_{j};
        '''.format(j=j, wsize=w_shape[j], boundary=boundary, type=int_type))

    # CArray: string becomes 'x[inds]', no format call needed
    value = '(*(X*)&data[{expr}])'.format(expr=expr)
    if mode == 'constant':
        cond = ' || '.join(['(ix_{} < 0)'.format(j) for j in range(ndim)])

    if cval is numpy.nan:
        cval = 'CUDART_NAN'
    elif cval == numpy.inf:
        cval = 'CUDART_INF'
    elif cval == -numpy.inf:
        cval = '-CUDART_INF'

    if binary_morphology:
        found = found.format(cond=cond, value=value)
    else:
        if mode == 'constant':
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

    mode_str = mode.replace('-', '_')  # avoid potential hyphen in kernel name
    name = 'cupyx_scipy_ndimage_{}_{}d_{}_w{}'.format(
        name, ndim, mode_str, '_'.join(['{}'.format(x) for x in w_shape]))
    if all_weights_nonzero:
        name += '_all_nonzero'
    if int_type == 'ptrdiff_t':
        name += '_i64'
    if has_structure:
        name += '_with_structure'
    if has_mask:
        name += '_with_mask'
    preamble = includes + _CAST_FUNCTION + preamble
    options += ('--std=c++11', '-DCUPY_USE_JITIFY')
    return cupy.ElementwiseKernel(in_params, out_params, operation, name,
                                  reduce_dims=False, preamble=preamble,
                                  options=options)

import cupy

from cupy import _util
from cupyx.scipy.ndimage import _filters_core


def _get_sub_kernel(f):
    """
    Takes the "function" given to generic_filter and returns the "sub-kernel"
    that will be called, one of RawKernel or ReductionKernel.

    This supports:
     * cupy.RawKernel
       no checks are possible
     * cupy.ReductionKernel
       checks that there is a single input and output
    """
    if isinstance(f, cupy.RawKernel):
        # We will assume that it has the correct API
        return f
    elif isinstance(f, cupy.ReductionKernel):
        if f.nin != 1 or f.nout != 1:
            raise TypeError('ReductionKernel must have 1 input and output')
        return f
    elif isinstance(f, cupy.ElementwiseKernel):
        # special error message for ElementwiseKernels
        raise TypeError('only ReductionKernel allowed (not ElementwiseKernel)')
    else:
        raise TypeError('bad function type')


@_util.memoize(for_each_device=True)
def _get_generic_filter_red(rk, in_dtype, out_dtype, filter_size, mode,
                            wshape, offsets, cval, int_type):
    """Generic filter implementation based on a reduction kernel."""
    # Get the temporary output c type
    in_param, out_param = rk.in_params[0], rk.out_params[0]
    out_ctype = out_param.ctype
    if out_param.dtype is None:  # resolve template
        out_ctype = cupy._core._scalar.get_typename(
            in_dtype if out_param.ctype == in_param.ctype else out_dtype)

    # Get code chunks
    setup = '''
    int iv = 0;
    X values[{size}];
    CArray<X, 1, true, true> sub_in(values, {{{size}}});
    {out_ctype} val_out;
    CArray<{out_ctype}, 1, true, true> sub_out(&val_out, {{1}});
    '''.format(size=filter_size, out_ctype=out_ctype)

    sub_call = '''reduction_kernel::{}(sub_in, sub_out);
    y = cast<Y>(val_out);'''.format(rk.name)

    sub_kernel = _reduction_kernel_code(rk, filter_size, out_dtype, in_dtype)

    # Get the final kernel
    return _filters_core._generate_nd_kernel(
        'generic_{}_{}'.format(filter_size, rk.name),
        setup, 'values[iv++] = {value};', sub_call,
        mode, wshape, int_type, offsets, cval, preamble=sub_kernel,
        options=getattr(rk, 'options', ()))


def _reduction_kernel_code(rk, filter_size, out_dtype, in_dtype):
    # NOTE: differences from the code generated for real reduction kernels:
    #  * input is always 1D and always less than 2^31 elements
    #  * output is always 1D with a single element
    #  * never across threads (no _block_stride, _sdata, _sdata_raw, _REDUCE,
    #       _tid, _J, _i, _i_base, _j_offset, _J_offset, _j_stride, _J_stride)
    # Also, the code is moved into a namespace so that clashes are minimized
    # between the typedefs for the "template" variables.

    # figure out the types
    types = {}
    in_param, out_param = rk.in_params[0], rk.out_params[0]
    in_ctype = _get_type_info(in_param, in_dtype, types)
    out_ctype = _get_type_info(out_param, out_dtype, types)
    types = '\n'.join('typedef {} {};'.format(typ, name)
                      for name, typ in types.items())

    return '''namespace reduction_kernel {{
{type_preamble}
{preamble}
__device__
void {name}({in_const} CArray<{in_ctype}, 1, true, true>& _raw_{in_name},
            CArray<{out_ctype}, 1, true, true>& _raw_{out_name}) {{
    // these are just provided so if they are available for the RK
    CIndexer<1> _in_ind({{{size}}});
    CIndexer<0> _out_ind;

    #define REDUCE(a, b) ({reduce_expr})
    #define POST_MAP(a) ({post_map_expr})
    typedef {reduce_type} _type_reduce;
    _type_reduce _s = _type_reduce({identity});
    for (int _j = 0; _j < {size}; ++_j) {{
        _in_ind.set(_j);
        {in_const} {in_ctype}& {in_name} = _raw_{in_name}[_j];
        _type_reduce _a = static_cast<_type_reduce>({pre_map_expr});
        _s = REDUCE(_s, _a);
    }}
    _out_ind.set(0);
    {out_ctype} &{out_name} = _raw_{out_name}[0];
    POST_MAP(_s);
    #undef REDUCE
    #undef POST_MAP
}}
}}'''.format(
        name=rk.name, type_preamble=types, preamble=rk.preamble,
        in_const='const' if in_param.is_const else '',
        in_ctype=in_ctype, in_name=in_param.name,
        out_ctype=out_ctype, out_name=out_param.name,

        pre_map_expr=rk.map_expr,
        identity='' if rk.identity is None else rk.identity,
        size=filter_size,
        reduce_type=rk.reduce_type, reduce_expr=rk.reduce_expr,
        post_map_expr=rk.post_map_expr,
    )


def _get_type_info(param, dtype, types):
    if param.dtype is not None:
        return param.ctype
    # Template type -> map to actual output type
    ctype = cupy._core._scalar.get_typename(dtype)
    types.setdefault(param.ctype, ctype)
    return ctype


@_util.memoize(for_each_device=True)
def _get_generic_filter_raw(rk, filter_size, mode, wshape, offsets, cval,
                            int_type):
    """Generic filter implementation based on a raw kernel."""
    setup = '''
    int iv = 0;
    double values[{}];
    double val_out;'''.format(filter_size)

    sub_call = '''raw_kernel::{}(values, {}, &val_out);
    y = cast<Y>(val_out);'''.format(rk.name, filter_size)

    return _filters_core._generate_nd_kernel(
        'generic_{}_{}'.format(filter_size, rk.name),
        setup, 'values[iv++] = cast<double>({value});', sub_call,
        mode, wshape, int_type, offsets, cval,
        preamble='namespace raw_kernel {{\n{}\n}}'.format(
            # Users can test RawKernel independently, but when passed to here
            # it must be used as a device function here. In fact, RawKernel
            # wouldn't compile if code only contains device functions, so this
            # is necessary.
            rk.code.replace('__global__', '__device__')),
        options=rk.options)


@_util.memoize(for_each_device=True)
def _get_generic_filter1d(rk, length, n_lines, filter_size, origin, mode, cval,
                          in_ctype, out_ctype, int_type):
    """
    The generic 1d filter is different than all other filters and thus is the
    only filter that doesn't use _generate_nd_kernel() and has a completely
    custom raw kernel.
    """
    in_length = length + filter_size - 1
    start = filter_size // 2 + origin
    end = start + length

    if mode == 'constant':
        boundary, boundary_early = '', '''
        for (idx_t j = 0; j < {start}; ++j) {{ input_line[j] = {cval}; }}
        for (idx_t j = {end}; j<{in_length}; ++j) {{ input_line[j] = {cval}; }}
        '''.format(start=start, end=end, in_length=in_length, cval=cval)
    else:
        if length == 1:
            a = b = 'j_ = 0;'
        elif mode == 'reflect':
            j = ('j_ = ({j}) % ({length} * 2);\n'
                 'j_ = min(j_, 2 * {length} - 1 - j_);')
            a = j.format(j='-1 - j_', length=length)
            b = j.format(j='j_', length=length)
        elif mode == 'mirror':
            j = ('j_ = 1 + (({j}) - 1) % (({length} - 1) * 2);\n'
                 'j_ = min(j_, 2 * {length} - 2 - j_);')
            a = j.format(j='-j_', length=length)
            b = j.format(j='j_', length=length)
        elif mode == 'nearest':
            a, b = 'j_ = 0;', 'j_ = {length}-1;'.format(length=length)
        elif mode == 'wrap':
            a = 'j_ = j_ % {length} + {length};'.format(length=length)
            b = 'j_ = j_ % {length};'.format(length=length)
        loop = '''for (idx_t j = {{}}; j < {{}}; ++j) {{{{
            idx_t j_ = j - {start};
            {{}}
            input_line[j] = input_line[j_ + {start}];
        }}}}'''.format(start=start)
        boundary_early = ''
        boundary = (loop.format(0, start, a) + '\n' +
                    loop.format(end, in_length, b))

    name = 'generic1d_{}_{}_{}'.format(length, filter_size, rk.name)
    code = '''#include "cupy/carray.cuh"
#include "cupy/complex.cuh"
#include <type_traits>  // let Jitify handle this

namespace raw_kernel {{\n{rk_code}\n}}

{CAST}

typedef unsigned char byte;
typedef {in_ctype} X;
typedef {out_ctype} Y;
typedef {int_type} idx_t;

__device__ idx_t offset(idx_t i, idx_t axis, idx_t ndim,
                        const idx_t* shape, const idx_t* strides) {{
    idx_t index = 0;
    for (idx_t a = ndim; --a > 0; ) {{
        if (a == axis) {{ continue; }}
        index += (i % shape[a]) * strides[a];
        i /= shape[a];
    }}
    return index + strides[0] * i;
}}

extern "C" __global__
void {name}(const byte* input, byte* output, const idx_t* x) {{
    const idx_t axis = x[0], ndim = x[1],
        *shape = x+2, *in_strides = x+2+ndim, *out_strides = x+2+2*ndim;

    const idx_t in_elem_stride = in_strides[axis];
    const idx_t out_elem_stride = out_strides[axis];

    double input_line[{in_length}];
    double output_line[{length}];
    {boundary_early}

    for (idx_t i = ((idx_t)blockIdx.x) * blockDim.x + threadIdx.x;
            i < {n_lines};
            i += ((idx_t)blockDim.x) * gridDim.x) {{
        // Copy line from input (with boundary filling)
        const byte* input_ = input + offset(i, axis, ndim, shape, in_strides);
        for (idx_t j = 0; j < {length}; ++j) {{
            input_line[j+{start}] = (double)*(X*)(input_+j*in_elem_stride);
        }}
        {boundary}

        raw_kernel::{rk_name}(input_line, {in_length}, output_line, {length});

        // Copy line to output
        byte* output_ = output + offset(i, axis, ndim, shape, out_strides);
        for (idx_t j = 0; j < {length}; ++j) {{
            *(Y*)(output_+j*out_elem_stride) = cast<Y>(output_line[j]);
        }}
    }}
}}'''.format(n_lines=n_lines, length=length, in_length=in_length, start=start,
             in_ctype=in_ctype, out_ctype=out_ctype, int_type=int_type,
             boundary_early=boundary_early, boundary=boundary,
             name=name, rk_name=rk.name,
             # Users can test RawKernel independently, but when passed to here
             # it must be used as a device function here. In fact, RawKernel
             # wouldn't compile if code only contains device functions, so this
             # is necessary.
             rk_code=rk.code.replace('__global__', '__device__'),
             CAST=_filters_core._CAST_FUNCTION)
    return cupy.RawKernel(code, name, ('--std=c++11',) + rk.options,
                          jitify=True)

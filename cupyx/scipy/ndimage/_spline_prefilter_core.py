"""
Spline poles and boundary handling implemented as in SciPy

https://github.com/scipy/scipy/blob/master/scipy/ndimage/src/ni_splines.c
"""
import functools
import math
import operator
import textwrap

import cupy


def get_poles(order):
    if order == 2:
        # sqrt(8.0) - 3.0
        return (-0.171572875253809902396622551580603843,)
    elif order == 3:
        # sqrt(3.0) - 2.0
        return (-0.267949192431122706472553658494127633,)
    elif order == 4:
        # sqrt(664.0 - sqrt(438976.0)) + sqrt(304.0) - 19.0
        # sqrt(664.0 + sqrt(438976.0)) - sqrt(304.0) - 19.0
        return (-0.361341225900220177092212841325675255,
                -0.013725429297339121360331226939128204)
    elif order == 5:
        # sqrt(67.5 - sqrt(4436.25)) + sqrt(26.25) - 6.5
        # sqrt(67.5 + sqrt(4436.25)) - sqrt(26.25) - 6.5
        return (-0.430575347099973791851434783493520110,
                -0.043096288203264653822712376822550182)
    else:
        raise ValueError('only order 2-5 supported')


def get_gain(poles):
    return functools.reduce(operator.mul,
                            [(1.0 - z) * (1.0 - 1.0 / z) for z in poles])


def _causal_init_code(mode):
    """Code for causal initialization step of IIR filtering.

    c is a 1d array of length n and z is a filter pole
    """
    code = f'''
        // causal init for mode={mode}'''
    if mode == 'mirror':
        code += '''
        z_i = z;
        z_n_1 = pow(z, (P)(n - 1));

        c[0] = c[0] + z_n_1 * c[(n - 1) * element_stride];
        for (i = 1; i < min(n - 1, {n_boundary}); ++i) {{
            c[0] += z_i * (c[i * element_stride] +
                           z_n_1 * c[(n - 1 - i) * element_stride]);
            z_i *= z;
        }}
        c[0] /= 1 - z_n_1 * z_n_1;'''
    elif mode == 'grid-wrap':
        code += '''
        z_i = z;

        for (i = 1; i < min(n, {n_boundary}); ++i) {{
            c[0] += z_i * c[(n - i) * element_stride];
            z_i *= z;
        }}
        c[0] /= 1 - z_i; /* z_i = pow(z, n) */'''
    elif mode == 'reflect':
        code += '''
        z_i = z;
        z_n = pow(z, (P)n);
        c0 = c[0];

        c[0] = c[0] + z_n * c[(n - 1) * element_stride];
        for (i = 1; i < min(n, {n_boundary}); ++i) {{
            c[0] += z_i * (c[i * element_stride] +
                           z_n * c[(n - 1 - i) * element_stride]);
            z_i *= z;
        }}
        c[0] *= z / (1 - z_n * z_n);
        c[0] += c0;'''
    else:
        raise ValueError('invalid mode: {}'.format(mode))
    return code


def _anticausal_init_code(mode):
    """Code for the anti-causal initialization step of IIR filtering.

    c is a 1d array of length n and z is a filter pole
    """
    code = f'''
        // anti-causal init for mode={mode}'''
    if mode == 'mirror':
        code += '''
        c[(n - 1) * element_stride] = (
            z * c[(n - 2) * element_stride] +
            c[(n - 1) * element_stride]) * z / (z * z - 1);'''
    elif mode == 'grid-wrap':
        code += '''
        z_i = z;

        for (i = 0; i < min(n - 1, {n_boundary}); ++i) {{
            c[(n - 1) * element_stride] += z_i * c[i * element_stride];
            z_i *= z;
        }}
        c[(n - 1) * element_stride] *= z / (z_i - 1); /* z_i = pow(z, n) */'''
    elif mode == 'reflect':
        code += '''
        c[(n - 1) * element_stride] *= z / (z - 1);'''
    else:
        raise ValueError('invalid mode: {}'.format(mode))
    return code


def _get_spline_mode(mode):
    """spline boundary mode for interpolation with order >= 2."""
    if mode in ['mirror', 'reflect', 'grid-wrap']:
        # exact analytic boundary conditions exist for these modes.
        return mode
    elif mode == 'grid-mirror':
        # grid-mirror is a synonym for 'reflect'
        return 'reflect'
    # No exact analytical spline boundary condition implemented. Reflect gives
    # lower error than using mirror or wrap for mode 'nearest'. Otherwise, a
    # mirror spline boundary condition is used.
    return 'reflect' if mode == 'nearest' else 'mirror'


def _get_spline1d_code(mode, poles, n_boundary):
    """Generates the code required for IIR filtering of a single 1d signal.

    Prefiltering is done by causal filtering followed by anti-causal filtering.
    Multiple boundary conditions have been implemented.
    """
    code = ['''
    __device__ void spline_prefilter1d(
        T* __restrict__ c, idx_t signal_length, idx_t element_stride)
    {{''']

    # variables common to all boundary modes
    code.append('''
        idx_t i, n = signal_length;
        P z, z_i;''')

    # retrieve the spline boundary extension mode to use
    mode = _get_spline_mode(mode)

    if mode == 'mirror':
        # variables specific to mirror boundary mode
        code.append('''
        P z_n_1;''')
    elif mode == 'reflect':
        # variables specific to reflect boundary mode
        code.append('''
        P z_n;
        T c0;''')

    for pole in poles:

        code.append(f'''
        // select the current pole
        z = {pole};''')

        # initialize and apply the causal filter
        code.append(_causal_init_code(mode))
        code.append('''
        // apply the causal filter for the current pole
        for (i = 1; i < n; ++i) {{
            c[i * element_stride] += z * c[(i - 1) * element_stride];
        }}''')

        # initialize and apply the anti-causal filter
        code.append(_anticausal_init_code(mode))
        code.append('''
        // apply the anti-causal filter for the current pole
        for (i = n - 2; i >= 0; --i) {{
            c[i * element_stride] = z * (c[(i + 1) * element_stride] -
                                         c[i * element_stride]);
        }}''')

    code += ['''
    }}''']
    return textwrap.dedent('\n'.join(code)).format(n_boundary=n_boundary)


_FILTER_GENERAL = '''
#include "cupy/carray.cuh"
#include "cupy/complex.cuh"
typedef {data_type} T;
typedef {pole_type} P;
typedef {index_type} idx_t;
template <typename T>
__device__ T* row(
        T* ptr, idx_t i, idx_t axis, idx_t ndim, const idx_t* shape) {{
    idx_t index = 0, stride = 1;
    for (idx_t a = ndim - 1; a > 0; --a) {{
        if (a != axis) {{
            index += (i % shape[a]) * stride;
            i /= shape[a];
        }}
        stride *= shape[a];
    }}
    return ptr + index + stride * i;
}}
'''


_batch_spline1d_strided_template = """
extern "C" __global__
__launch_bounds__({block_size})
void cupyx_spline_filter(T* __restrict__ y, const idx_t* __restrict__ info) {{
    const idx_t n_signals = info[0], n_samples = info[1],
        * __restrict__ shape = info+2;
    idx_t y_elem_stride = 1;
    for (int a = {ndim} - 1; a > {axis}; --a) {{ y_elem_stride *= shape[a]; }}
    idx_t unraveled_idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx_t batch_idx = unraveled_idx;
    if (batch_idx < n_signals)
    {{
        T* __restrict__ y_i = row(y, batch_idx, {axis}, {ndim}, shape);
        spline_prefilter1d(y_i, n_samples, y_elem_stride);
    }}
}}
"""


@cupy.memoize(for_each_device=True)
def get_raw_spline1d_kernel(axis, ndim, mode, order, index_type='int',
                            data_type='double', pole_type='double',
                            block_size=128):
    """Generate a kernel for applying a spline prefilter along a given axis."""
    poles = get_poles(order)

    # determine number of samples for the boundary approximation
    # (SciPy uses n_boundary = n_samples but this is excessive)
    largest_pole = max([abs(p) for p in poles])
    # tol < 1e-7 fails test cases comparing to SciPy at atol = rtol = 1e-5
    tol = 1e-10 if pole_type == 'float' else 1e-18
    n_boundary = math.ceil(math.log(tol, largest_pole))

    # headers and general utility function for extracting rows of data
    code = _FILTER_GENERAL.format(index_type=index_type,
                                  data_type=data_type,
                                  pole_type=pole_type)

    # generate source for a 1d function for a given boundary mode and poles
    code += _get_spline1d_code(mode, poles, n_boundary)

    # generate code handling batch operation of the 1d filter
    code += _batch_spline1d_strided_template.format(ndim=ndim, axis=axis,
                                                    block_size=block_size)
    return cupy.RawKernel(code, 'cupyx_spline_filter')

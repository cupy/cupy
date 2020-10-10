"""
Spline poles and boundary handling implemented as in SciPy

https://github.com/scipy/scipy/blob/master/scipy/ndimage/src/ni_splines.c
"""
import functools
import operator

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
        raise ValueError("only order 2-5 supported")


def get_gain(poles):
    return functools.reduce(operator.mul,
                            [(1.0 - z) * (1.0 - 1.0 / z) for z in poles])


def _causal_init_code(mode):
    """Code for causal initialization step of IIR filtering.

    c is a 1d array of length n and z is a filter pole
    """
    if mode in ["nearest", "constant"]:
        mode = "mirror"
    code = """
        // causal init for mode={mode}""".format(
        mode=mode
    )
    if mode == "mirror":
        code += """
        z_i = z;
        z_n_1 = pow(z, ({dtype_pole})(n - 1));

        c[0] = c[0] + z_n_1 * c[n - 1];
        for (i = 1; i < n - 1; ++i) {{
            c[0] += z_i * (c[i] + z_n_1 * c[n - 1 - i]);
            z_i *= z;
        }}
        c[0] /= 1 - z_n_1 * z_n_1;"""
    elif mode == "wrap":
        code += """
        z_i = z;

        for (i = 1; i < n; ++i) {{
            c[0] += z_i * c[n - i];
            z_i *= z;
        }}
        c[0] /= 1 - z_i; /* z_i = pow(z, n) */"""
    elif mode == "reflect":
        code += """
        z_i = z;
        z_n = pow(z, ({dtype_pole})n);
        c0 = c[0];

        c[0] = c[0] + z_n * c[n - 1];
        for (i = 1; i < n; ++i) {{
            c[0] += z_i * (c[i] + z_n * c[n - 1 - i]);
            z_i *= z;
        }}
        c[0] *= z / (1 - z_n * z_n);
        c[0] += c0;"""
    else:
        raise ValueError("invalid mode: {}".format(mode))
    return code


def _anticausal_init_code(mode):
    """Code for the anti-causal initialization step of IIR filtering.

    c is a 1d array of length n and z is a filter pole
    """
    if mode in ["nearest", "constant"]:
        mode = "mirror"
    code = """
        // anti-causal init for mode={mode}""".format(
        mode=mode
    )
    if mode == "mirror":
        code += """
        c[n - 1] = (z * c[n - 2] + c[n - 1]) * z / (z * z - 1);"""
    elif mode == "wrap":
        code += """
        z_i = z;

        for (i = 0; i < n - 1; ++i) {{
            c[n - 1] += z_i * c[i];
            z_i *= z;
        }}
        c[n - 1] *= z / (z_i - 1); /* z_i = pow(z, n) */"""
    elif mode == "reflect":
        code += """
        c[n - 1] *= z / (z - 1);"""
    else:
        raise ValueError("invalid mode: {}".format(mode))
    return code


def get_spline1d_code(mode, poles):
    """Generates the code required for IIR filtering of a single 1d signal.

    Prefiltering is done by causal filtering followed by anti-causal filtering.

    Currently this filtering can only be applied along the axis which is
    contiguous in memory (e.g. the last axis for C-contiguous arrays). This
    function will be applied in a batched fashion (see
    ``batch_spline1d_template``).
    """
    code = ["""
    #include <cupy/complex.cuh>

    __device__ void spline_prefilter1d(
        {dtype_data}* __restrict__ c, {dtype_index} signal_length)
    {{"""]

    # variables common to all boundary modes
    code.append("""
        {dtype_index} i, n = signal_length;
        {dtype_pole} z, z_i;""")

    if mode in ["mirror", "constant", "nearest"]:
        # variables specific to these modes
        code.append("""
        {dtype_pole} z_n_1;""")
    elif mode == "reflect":
        # variables specific to this modes
        code.append("""
        {dtype_pole} z_n;
        {dtype_data} c0;""")

    for pole in poles:

        code.append("""
        // select the current pole
        z = {pole};""".format(pole=pole))

        # initialize and apply the causal filter
        code.append(_causal_init_code(mode))
        code.append("""
        // apply the causal filter for the current pole
        for (i = 1; i < n; ++i) {{
            c[i] += z * c[i - 1];
        }}""")

        # initialize and apply the anti-causal filter
        code.append(_anticausal_init_code(mode))
        code.append("""
        // apply the anti-causal filter for the current pole
        for (i = n - 2; i >= 0; --i) {{
            c[i] = z * (c[i + 1] - c[i]);
        }}""")

    code += ["""
    }}"""]
    return "\n".join(code)


batch_spline1d_template = """

    extern "C" {{
    __global__ void batch_spline_prefilter(
        {dtype_data}* __restrict__ x,
        {dtype_index} len_x,
        {dtype_index} n_batch)
    {{
        {dtype_index} unraveled_idx = blockDim.x * blockIdx.x + threadIdx.x;
        {dtype_index} batch_idx = unraveled_idx;
        if (batch_idx < n_batch)
        {{
            {dtype_index} offset_x = batch_idx * len_x;  // current line offset
            spline_prefilter1d(&x[offset_x], len_x);
        }}
    }}
    }}
"""


@cupy.memoize(for_each_device=True)
def get_raw_spline1d_code(
    mode, order=3, dtype_index="int", dtype_data="double", dtype_pole="double"
):
    """Get kernel code for a spline prefilter.

    The kernels assume the data has been reshaped to shape (n_batch, size) and
    filtering is to be performed along the last axis.

    See cupyimg.scipy.ndimage.interpolation.spline_filter1d for how this can
    be used to filter along any axis of an array via swapping axes and
    reshaping. For n-dimensional filtering, the prefilter is seperable across
    axes and thus a 1d filter is applied along each axis in turn.
    """
    poles = get_poles(order)

    # generate source for a 1d function for a given boundary mode and poles
    code = get_spline1d_code(mode, poles)

    # generate code handling batch operation of the 1d filter
    code += batch_spline1d_template
    code = code.format(
        dtype_index=dtype_index, dtype_data=dtype_data, dtype_pole=dtype_pole
    )
    return code

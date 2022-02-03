from cupy import _core

_gamma_body = """

    if (isinf(in0) && in0 < 0) {
        out0 = -1.0 / 0.0;
    } else if (in0 < 0. && in0 == floor(in0)) {
        out0 = 1.0 / 0.0;
    } else {
        out0 = tgamma(in0);
    }
"""

# Also define a standalone Gamma device function for internal use in other code
# like beta, betaln, etc.
gamma_definition = f"""

__noinline__ __device__ double Gamma(double in0)
{{
    double out0;
    {_gamma_body}
    return out0;
}}
"""

gamma = _core.create_ufunc(
    'cupyx_scipy_gamma', ('f->f', 'd->d'),
    _gamma_body,
    doc="""Gamma function.

    Args:
        z (cupy.ndarray): The input of gamma function.

    Returns:
        cupy.ndarray: Computed value of gamma function.

    .. seealso:: :data:`scipy.special.gamma`

    """)

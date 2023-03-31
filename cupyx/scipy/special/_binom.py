"""
Implements the binom function from scipy.

This is essentially a CUDA C++ adaptation of existing scipy code, available at:
https://github.com/scipy/scipy/blob/v1.10.1/scipy/special/orthogonal_eval.pxd
"""

from cupy import _core
from cupyx.scipy.special._beta import (
    beta_preamble,
    lbeta_symp_definition,
    lgam_sgn_definition,
    beta_definition,
    lbeta_definition
)
from cupyx.scipy.special._digamma import polevl_definition
from cupyx.scipy.special._gamma import gamma_definition
from cupyx.scipy.special._gammainc import p1evl_definition

binom_definition = """
__device__ double binom(double n, double k) {

    // Check if n is negative integer
    if (n < 0 && n == trunc(n)) {
        return CUDART_NAN;
    }
    // Check if k is infinite
    if (k == CUDART_INF) {
        return CUDART_NAN;
    }

    // k is integer and n is not very small
    if (k == floor(k) && (fabs(n) > 1.0e-8 || n == 0)) {
        // Reduce k by symmetry if n also integer
        if (n == floor(n) && k > n/2 && n > 0) {
            k = n - k;
        }
        // Manually multiply if k between 0, 20
        if (0 <= k && k < 20) {
            double num = 1.0;
            double den = 1.0;
            for (int i = 1; i < k+.5; i++) {
                num *= i + n - k;
                den *= i;
                // Resize num, den if num becomes too big
                if (fabs(num) > 1.0e50) {
                    num /= den;
                    den = 1.0;
                }
            }
            return num / den;
        }
    }

    if (n >= k * 1e10 && k > 0) {
        // Prevent overflow
        return exp(-lbeta(1 + n - k, 1 + k) - log(n + 1));
    } else if (k > 1e8 * fabs(n)) {
        double num = Gamma(1 + n) / fabs(k) + Gamma(1 + n) * n / (2 * k*k);
        num /= M_PI * pow(fabs(k), n);
        double kfloor = floor(k);
        if (k > 0) {
            int sgn;
            double dk;
            if (int(kfloor) == kfloor) {
                dk = k - kfloor;
                sgn = int(kfloor) % 2 == 0 ? 1 : -1;
            } else {
                dk = k;
                sgn = 1;
            }
            return num * sin((dk-n)*M_PI) * sgn;
        } else {
            if (int(kfloor) == kfloor) {
                return 0;
            } else {
                return num * sin(k*M_PI);
            }
        }
    } else {
        return 1/(n + 1)/beta(1 + n - k, 1 + k);
    }
}
"""

binom = _core.create_ufunc(
    "cupyx_scipy_binom",
    ("ff->f", "dd->d"),
    "out0 = out0_type(binom(in0, in1));",
    preamble=(
        gamma_definition +
        beta_preamble +
        p1evl_definition +
        polevl_definition +
        lgam_sgn_definition +
        lbeta_symp_definition +
        lbeta_definition +
        beta_definition +
        binom_definition
    ),
    doc="""Binomial coefficient

    Parameters
    ----------
    n, k : cupy.ndarray
        Real-valued parameters to nCk
    out : ndarray, optional
        Optional output array for the function result

    Returns
    -------
    scalar or ndarray
        Value of binomial coefficient

    .. seealso:: :func:`scipy.special.binom`

    """,
)

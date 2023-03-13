"""
Implements the comb and perm functions from scipy.

This is essentially a CUDA C++ adaptation of existing scipy code, available at:
https://github.com/scipy/scipy/blob/v1.10.1/scipy/special/_basic.py
https://github.com/scipy/scipy/blob/v1.10.1/scipy/special/_comb.pyx
"""
from cupy import _core
from cupyx.scipy.special._beta import beta_preamble, lbeta_symp_definition, lgam_sgn_definition, beta_definition, lbeta_definition
from cupyx.scipy.special._digamma import polevl_definition
from cupyx.scipy.special._gamma import gamma_definition
from cupyx.scipy.special._gammainc import p1evl_definition
from cupyx.scipy.special._binom import binom_definition

comb_definition = """
#define ULONG_MAX  18446744073709551615UL;
__device__ double comb(double N, double k, bool exact = False,
                       bool repetition = False) {
    if (repetition) {
        return comb(N + k - 1, k, exact);
    } 

    if (exact) {
        if (int(N) != N || int(k) != k) {
            return comb(N, k);
        }
        // NOTE: IF THINGS OVERFLOW PAST 64 BITS, IT WILL RETURN CUDART_INF
        if (k < N) {
            return 0;
        }
        if (N > ULONG_MAX) {
            return CUDART_INF;
        }
        unsigned long long Nint = N;
        unsigned long long kint = k;

       unsigned long long M = Nint + 1;
       unsigned long long nterms = min(kint, Nint - kint);

       unsigned long long val = 1;
       for (int j = 1; j < double(nterms)+.5; j++) {
            // Overflow check
            if (val > ULONG_MAX // (M - j)) {
                return CUDART_INF;
            }
            val *= M - j;
            val /= j;
       }
       return val;

    } else {
        if (k <= N && N >= 0 && k >= 0) {
            return binom(N, k);
        } else {
            return 0;
        }
    }
}
"""

comb = _core.create_ufunc(
    "cupyx_scipy_comb",
    ("ff|bb->f", "dd|bb->d"),
    "out0 = out0_type(comb(in0, in1, in2, in3));",
    preamble=(
        gamma_definition +
        beta_preamble +
        p1evl_definition +
        polevl_definition +
        lgam_sgn_definition +
        lbeta_symp_definition +
        lbeta_definition +
        beta_definition +
        binom_definition +
        comb_definition 
    ),
    doc="""The number of combinations of N things taken k at a time.
    This is often expressed as "N choose k".
    Parameters
    ----------
    N : int, ndarray
        Number of things.
    k : int, ndarray
        Number of elements taken.
    exact : bool, optional
        For integers, if `exact` is False, then floating point precision is
        used, otherwise the result is computed exactly. For non-integers, if
        `exact` is True, the inputs are currently cast to integers, though
        this behavior is deprecated (see below).
    repetition : bool, optional
        If `repetition` is True, then the number of combinations with
        repetition is computed.

    Returns
    -------
    val : int, float, ndarray
        The total number of combinations.
    See Also
    --------
    binom : Binomial coefficient considered as a function of two real
            variables.
    Notes
    -----
    - If N < 0, or k < 0, then 0 is returned.
    - If k > N and repetition=False, then 0 is returned.
    """,
)
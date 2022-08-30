"""Statistical distribution functions (Beta, Binomial, Poisson, etc.)

The source code here is an adaptation with minimal changes from the following
files in SciPy's bundled Cephes library:

https://github.com/scipy/scipy/blob/main/scipy/special/cephes/bdtr.c
https://github.com/scipy/scipy/blob/main/scipy/special/cephes/chdtr.c
https://github.com/scipy/scipy/blob/main/scipy/special/cephes/fdtr.c
https://github.com/scipy/scipy/blob/main/scipy/special/cephes/gdtr.c
https://github.com/scipy/scipy/blob/main/scipy/special/cephes/nbdtr.c
https://github.com/scipy/scipy/blob/main/scipy/special/cephes/pdtr.c

Cephes Math Library, Release 2.3:  March, 1995
Copyright 1984, 1995 by Stephen L. Moshier
"""

from cupy import _core
from cupyx.scipy.special._beta import incbet_preamble, incbi_preamble
from cupyx.scipy.special._gammainc import _igam_preamble, _igami_preamble


# Normal distribution functions

ndtr = _core.create_ufunc(
    'cupyx_scipy_special_ndtr',
    (('f->f', 'out0 = normcdff(in0)'), 'd->d'),
    'out0 = normcdf(in0)',
    doc='''Cumulative distribution function of normal distribution.

    .. seealso:: :data:`scipy.special.ndtr`

    ''')


log_ndtr_definition = """

#define NPY_SQRT1_2   0.707106781186547524400844362104849039  /* 1/sqrt(2) */

static __device__ double log_ndtr(double x)
{
    double t = x * NPY_SQRT1_2;
    if (x < -1.0) {
        return log(erfcx(-t) / 2) - t * t;
    } else {
        return log1p(-erfc(t) / 2);
    }
}

static __device__ float log_ndtrf(float x)
{
    float t = x * NPY_SQRT1_2;
    if (x < -1.0) {
        return logf(erfcxf(-t) / 2) - t * t;
    } else {
        return log1pf(-erfcf(t) / 2);
    }
}

"""


log_ndtr = _core.create_ufunc(
    'cupyx_scipy_special_log_ndtr',
    (('f->f', 'out0 = log_ndtrf(in0)'), 'd->d'),
    'out0 = log_ndtr(in0)',
    preamble=log_ndtr_definition,
    doc="""Logarithm of Gaussian cumulative distribution function.

    Returns the log of the area under the standard Gaussian propability
    density function.

    Parameters
    ----------
    x : array-like
        The input array

    Returns
    -------
    y : cupy.ndarray
        The value of the log of the normal cumulative distribution
        function evaluated at x

    See Also
    --------
    :func:`scipy.special.log_ndtr`

    """,
)


ndtri = _core.create_ufunc(
    'cupyx_scipy_special_ndtri',
    (('f->f', 'out0 = normcdfinvf(in0)'), 'd->d'),
    'out0 = normcdfinv(in0)',
    doc='''Inverse of the cumulative distribution function of the standard
           normal distribution.

    .. seealso:: :data:`scipy.special.ndtri`
''')


# Binomial distribution functions

bdtr_definition = """

__device__ double bdtr(double k, int n, double p)
{
    double dk, dn;
    double fk = floor(k);

    if (isnan(p) || isnan(k)) {
        return CUDART_NAN;
    }

    if (p < 0.0 || p > 1.0 || fk < 0 || n < fk) {
        return CUDART_NAN;
    }

    if (fk == n) {
        return 1.0;
    }

    dn = n - fk;
    if (fk == 0) {
        dk = pow(1.0 - p, dn);
    } else {
        dk = fk + 1.;
        dk = incbet(dn, dk, 1.0 - p);
    }
    return dk;
}


__device__ double bdtr_unsafe(double k, double n, double p)
{
    if (isnan(n) || isinf(n)) {
        return CUDART_NAN;
    } else {
        return bdtr(k, (int)n, p);
    }
}

"""


bdtrc_definition = """

__device__ double bdtrc(double k, int n, double p)
{
    double dk, dn;
    double fk = floor(k);

    if (isnan(p) || isnan(k)) {
        return CUDART_NAN;
    }

    if (p < 0.0 || p > 1.0 || n < fk) {
        return CUDART_NAN;
    }

    if (fk < 0) {
        return 1.0;
    }

    if (fk == n) {
        return 0.0;
    }

    dn = n - fk;
    if (k == 0) {
        if (p < .01) {
            dk = -expm1(dn * log1p(-p));
        } else {
            dk = 1.0 - pow(1.0 - p, dn);
        }
    } else {
        dk = fk + 1;
        dk = incbet(dk, dn, p);
    }
    return dk;
}

__device__ double bdtrc_unsafe(double k, double n, double p)
{
    if (isnan(n) || isinf(n)) {
        return CUDART_NAN;
    } else {
        return bdtrc(k, (int)n, p);
    }
}

"""


bdtri_definition = """

__device__ double bdtri(double k, int n, double y)
{
    double p, dn, dk;
    double fk = floor(k);

    if (isnan(k)) {
        return CUDART_NAN;
    }

    if (y < 0.0 || y > 1.0 || fk < 0.0 || n <= fk) {
        return CUDART_NAN;
    }

    dn = n - fk;

    if (fk == n) {
        return 1.0;
    }

    if (fk == 0) {
        if (y > 0.8) {
            p = -expm1(log1p(y - 1.0) / dn);
        } else {
            p = 1.0 - pow(y, 1.0 / dn);
        }
    } else {
        dk = fk + 1;
        p = incbet(dn, dk, 0.5);
        if (p > 0.5) {
            p = incbi(dk, dn, 1.0 - y);
        } else {
            p = 1.0 - incbi(dn, dk, y);
        }
    }
    return p;
}

__device__ double bdtri_unsafe(double k, double n, double p)
{
    if (isnan(n) || isinf(n)) {
        return CUDART_NAN;
    } else {
        return bdtri(k, (int)n, p);
    }
}

"""


# Note: bdtr ddd->d and fff-> are deprecated as of SciPy 1.7
bdtr = _core.create_ufunc(
    "cupyx_scipy_bdtr",
    (
        ('fff->f', 'out0 = out0_type(bdtr_unsafe(in0, in1, in2));'),
        'dld->d',
        ('ddd->d', 'out0 = bdtr_unsafe(in0, in1, in2);'),
    ),
    "out0 = bdtr(in0, (int)in1, in2);",
    preamble=incbet_preamble + bdtr_definition,
    doc="""Binomial distribution cumulative distribution function.

    Parameters
    ----------
    k : cupy.ndarray
        Number of successes (float), rounded down to the nearest integer.
    n : cupy.ndarray
        Number of events (int).
    p : cupy.ndarray
        Probability of success in a single event (float).

    Returns
    -------
    y : cupy.ndarray
        Probability of floor(k) or fewer successes in n independent events with
        success probabilities of p.

    See Also
    --------
    :func:`scipy.special.bdtr`

    """,
)


# Note: bdtrc ddd->d and fff->f are deprecated as of SciPy 1.7
bdtrc = _core.create_ufunc(
    "cupyx_scipy_bdtrc",
    (
        ('fff->f', 'out0 = out0_type(bdtrc_unsafe(in0, in1, in2));'),
        'dld->d',
        ('ddd->d', 'out0 = bdtrc_unsafe(in0, in1, in2);'),
    ),
    "out0 = out0_type(bdtrc(in0, in1, in2));",
    preamble=incbet_preamble + bdtrc_definition,
    doc="""Binomial distribution survival function.

    Returns the complemented binomial distribution function (the integral of
    the density from x to infinity).

    Parameters
    ----------
    k : cupy.ndarray
        Number of successes (float), rounded down to the nearest integer.
    n : cupy.ndarray
        Number of events (int).
    p : cupy.ndarray
        Probability of success in a single event (float).

    Returns
    -------
    y : cupy.ndarray
        Probability of floor(k) + 1 or more successes in n independent events
        with success probabilities of p.

    See Also
    --------
    :func:`scipy.special.bdtrc`

    """,
)


# Note: bdtri ddd->d and fff->f are deprecated as of SciPy 1.7
bdtri = _core.create_ufunc(
    "cupyx_scipy_bdtri",
    (
        ('fff->f', 'out0 = out0_type(bdtri_unsafe(in0, in1, in2));'),
        'dld->d',
        ('ddd->d', 'out0 = bdtri_unsafe(in0, in1, in2);'),
    ),
    "out0 = out0_type(bdtri(in0, in1, in2));",
    preamble=incbi_preamble + bdtri_definition,
    doc="""Inverse function to `bdtr` with respect to `p`.

    Parameters
    ----------
    k : cupy.ndarray
        Number of successes (float), rounded down to the nearest integer.
    n : cupy.ndarray
        Number of events (int).
    y : cupy.ndarray
        Cumulative probability (probability of k or fewer successes in n
        events).

    Returns
    -------
    p : cupy.ndarray
        The event probability such that bdtr(floor(k), n, p) = y.

    See Also
    --------
    :func:`scipy.special.bdtri`

    """,
)


# Beta distribution functions

btdtr = _core.create_ufunc(
    "cupyx_scipy_btdtr",
    ("fff->f", "ddd->d"),
    "out0 = out0_type(incbet(in0, in1, in2));",
    preamble=incbet_preamble,
    doc="""Cumulative distribution function of the beta distribution.

    Parameters
    ----------
    a : cupy.ndarray
        Shape parameter (a > 0).
    b : cupy.ndarray
        Shape parameter (b > 0).
    x : cupy.ndarray
        Upper limit of integration, in [0, 1].

    Returns
    -------
    I : cupy.ndarray
        Cumulative distribution function of the beta distribution with
        parameters a and b at x.

    See Also
    --------
    :func:`scipy.special.btdtr`

    """,
)


btdtri = _core.create_ufunc(
    "cupyx_scipy_btdtri",
    ("fff->f", "ddd->d"),
    "out0 = out0_type(incbi(in0, in1, in2));",
    preamble=incbi_preamble,
    doc="""The p-th quantile of the beta distribution.

    This function is the inverse of the beta cumulative distribution function,
    `btdtr`, returning the value of `x` for which ``btdtr(a, b, x) = p``.

    Parameters
    ----------
    a : cupy.ndarray
        Shape parameter (a > 0).
    b : cupy.ndarray
        Shape parameter (b > 0).
    p : cupy.ndarray
        Cumulative probability, in [0, 1].

    Returns
    -------
    x : cupy.ndarray
        The quantile corresponding to p.

    See Also
    --------
    :func:`scipy.special.btdtri`

    """,
)


# Chi square distribution functions

chdtrc_definition = """

__device__ double chdtrc(double df, double x)
{

    if (x < 0.0) {
        return 1.0;     /* modified by T. Oliphant */
    }
    return igamc(df / 2.0, x / 2.0);
}
"""


chdtr_definition = """

__device__ double chdtr(double df, double x)
{
    if (x < 0.0) {   /* || (df < 1.0) ) */
        return CUDART_NAN;
    }
    return igam(df / 2.0, x / 2.0);
}
"""


chdtri_definition = """
__device__ double chdtri(double df, double y)
{
    double x;

    if ((y < 0.0) || (y > 1.0)) {   /* || (df < 1.0) ) */
        return CUDART_NAN;
    }

    x = igamci(0.5 * df, y);
    return 2.0 * x;
}
"""


chdtrc = _core.create_ufunc(
    "cupyx_scipy_chdtrc",
    ("ff->f", "dd->d"),
    "out0 = out0_type(chdtrc(in0, in1));",
    preamble=_igam_preamble + chdtrc_definition,
    doc="""Chi square survival function.

    Returns the complemented chi-squared distribution function (the integral of
    the density from x to infinity).

    Parameters
    ----------
    v : cupy.ndarray
        Degrees of freedom.
    x : cupy.ndarray
        Upper bound of the integral (nonnegative float).

    Returns
    -------
    y : cupy.ndarray
        The complemented chi-squared distribution function with parameter df at
        x.

    See Also
    --------
    :func:`scipy.special.chdtrc`

    """,
)

chdtri = _core.create_ufunc(
    "cupyx_scipy_chdtri",
    ("ff->f", "dd->d"),
    "out0 = out0_type(chdtri(in0, in1));",
    preamble=_igami_preamble + chdtri_definition,
    doc="""Inverse to `chdtrc` with respect to `x`.

    Parameters
    ----------
    v : cupy.ndarray
        Degrees of freedom.
    p : cupy.ndarray
        Probability.
    p : cupy.ndarray, optional
        Optional output array for the function results.

    Returns
    -------
    x : cupy.ndarray
        Value so that the probability a Chi square random variable with `v`
        degrees of freedom is greater than `x` equals `p`.

    See Also
    --------
    :func:`scipy.special.chdtri`

    """,
)


chdtr = _core.create_ufunc(
    "cupyx_scipy_chdtr",
    ("ff->f", "dd->d"),
    "out0 = out0_type(chdtr(in0, in1));",
    preamble=_igam_preamble + chdtr_definition,
    doc="""Chi-square cumulative distribution function.

    Parameters
    ----------
    v : cupy.ndarray
        Degrees of freedom.
    x : cupy.ndarray
        Upper bound of the integral (nonnegative float).

    Returns
    -------
    y : cupy.ndarray
        The CDF of the chi-squared distribution with parameter df at x.

    See Also
    --------
    :func:`scipy.special.chdtr`

    """,
)


# F distribution functions

fdtrc_definition = """

__device__ double fdtrc(double a, double b, double x)
{
    double w;

    if ((a <= 0.0) || (b <= 0.0) || (x < 0.0)) {
        // sf_error("fdtrc", SF_ERROR_DOMAIN, NULL);
        return CUDART_NAN;
    }
    w = b / (b + a * x);
    return incbet(0.5 * b, 0.5 * a, w);
}
"""


fdtr_definition = """

__device__ double fdtr(double a, double b, double x)
{
    double w;

    if ((a <= 0.0) || (b <= 0.0) || (x < 0.0)) {
        // sf_error("fdtr", SF_ERROR_DOMAIN, NULL);
        return CUDART_NAN;
    }
    w = a * x;
    w = w / (b + w);
    return incbet(0.5 * a, 0.5 * b, w);
}
"""


fdtri_definition = """
__device__ double fdtri(double a, double b, double y)
{
    double w, x;

    if ((a <= 0.0) || (b <= 0.0) || (y <= 0.0) || (y > 1.0)) {
        // sf_error("fdtri", SF_ERROR_DOMAIN, NULL);
        return CUDART_NAN;
    }
    y = 1.0 - y;
    /* Compute probability for x = 0.5.  */
    w = incbet(0.5 * b, 0.5 * a, 0.5);
    /* If that is greater than y, then the solution w < .5.
     * Otherwise, solve at 1-y to remove cancellation in (b - b*w).  */
    if (w > y || y < 0.001) {
        w = incbi(0.5 * b, 0.5 * a, y);
        x = (b - b * w) / (a * w);
    }
    else {
        w = incbi(0.5 * a, 0.5 * b, 1.0 - y);
        x = b * w / (a * (1.0 - w));
    }
    return x;
}
"""


fdtrc = _core.create_ufunc(
    "cupyx_scipy_fdtrc",
    ("fff->f", "ddd->d"),
    "out0 = out0_type(fdtrc(in0, in1, in2));",
    preamble=incbi_preamble + fdtrc_definition,
    doc="""F survival function.

    Returns the complemented F-distribution function (the integral of the
    density from x to infinity).

    Parameters
    ----------
    dfn : cupy.ndarray
        First parameter (positive float).
    dfd : cupy.ndarray
        Second parameter (positive float).
    x : cupy.ndarray
        Argument (nonnegative float).

    Returns
    -------
    y : cupy.ndarray
        The complemented F-distribution function with parameters dfn and dfd at
        x.

    .. seealso:: :meth:`scipy.special.fdtrc`

    """,
)

fdtri = _core.create_ufunc(
    "cupyx_scipy_fdtri",
    ("fff->f", "ddd->d"),
    "out0 = out0_type(fdtri(in0, in1, in2));",
    preamble=incbi_preamble + fdtri_definition,
    doc="""The p-th quantile of the F-distribution.

    This function is the inverse of the F-distribution CDF, `fdtr`, returning
    the `x` such that `fdtr(dfn, dfd, x)` = `p`.

    Parameters
    ----------
    dfn : cupy.ndarray
        First parameter (positive float).
    dfd : cupy.ndarray
        Second parameter (positive float).
    p : cupy.ndarray
        Cumulative probability, in [0, 1].

    Returns
    -------
    y : cupy.ndarray
        The quantile corresponding to p.

    .. seealso:: :meth:`scipy.special.fdtri`

    """,
)


fdtr = _core.create_ufunc(
    "cupyx_scipy_fdtr",
    ("fff->f", "ddd->d"),
    "out0 = out0_type(fdtr(in0, in1, in2));",
    preamble=incbi_preamble + fdtr_definition,
    doc="""F cumulative distribution function.


    Parameters
    ----------
    dfn : cupy.ndarray
        First parameter (positive float).
    dfd : cupy.ndarray
        Second parameter (positive float).
    x : cupy.ndarray
        Argument (nonnegative float).

    Returns
    -------
    y : cupy.ndarray
        The CDF of the F-distribution with parameters dfn and dfd at x.

    .. seealso:: :meth:`scipy.special.fdtr`

    """,
)


# Gamma distribution functions

gdtr_definition = """

__device__ double gdtr(double a, double b, double x)
{

    if (x < 0.0) {
        return CUDART_NAN;
    }
    return igam(b, a * x);
}
"""


gdtrc_definition = """

__device__ double gdtrc(double a, double b, double x)
{
    if (x < 0.0) {
        return CUDART_NAN;
    }
    return (igamc(b, a * x));
}

"""


gdtr = _core.create_ufunc(
    "cupyx_scipy_gdtr",
    ("fff->f", "ddd->d"),
    "out0 = out0_type(gdtr(in0, in1, in2));",
    preamble=_igam_preamble + gdtr_definition,
    doc="""Gamma distribution cumulative distribution function.

    Parameters
    ----------
    a : cupy.ndarray
        The rate parameter of the gamma distribution, sometimes denoted
        beta (float). It is also the reciprocal of the scale parameter theta.
    b : cupy.ndarray
        The shape parameter of the gamma distribution, sometimes denoted
        alpha (float).
    x : cupy.ndarray
        The quantile (upper limit of integration; float).

    Returns
    -------
    F : cupy.ndarray
        The CDF of the gamma distribution with parameters `a` and `b` evaluated
        at `x`.

    See Also
    --------
    :func:`scipy.special.gdtr`

    """,
)


gdtrc = _core.create_ufunc(
    "cupyx_scipy_gdtrc",
    ("fff->f", "ddd->d"),
    "out0 = out0_type(gdtrc(in0, in1, in2));",
    preamble=_igam_preamble + gdtrc_definition,
    doc="""Gamma distribution survival function.

    Parameters
    ----------
    a : cupy.ndarray
        The rate parameter of the gamma distribution, sometimes denoted
        beta (float). It is also the reciprocal of the scale parameter theta.
    b : cupy.ndarray
        The shape parameter of the gamma distribution, sometimes denoted
        alpha (float).
    x : cupy.ndarray
        The quantile (lower limit of integration; float).

    Returns
    -------
    I : cupy.ndarray
        The survival function of the gamma distribution with parameters `a` and
        `b` at `x`.

    See Also
    --------
    :func:`scipy.special.gdtrc`

    """,
)


# Negative Binomial distribution functions

nbdtr_definition = """

__device__ double nbdtr(int k, int n, double p)
{
    double dk, dn;

    if (((p < 0.0) || (p > 1.0)) || (k < 0))
    {
        return CUDART_NAN;
    }
    dk = k + 1;
    dn = n;
    return (incbet(dn, dk, p));
}

__device__ double nbdtr_unsafe(double k, double n, double p)
{
    if (isnan(k) || isnan(n))
    {
        return CUDART_NAN;
    }
    return nbdtr((int)k, (int)n, p);
}

"""


nbdtrc_definition = """

__device__ double nbdtrc(int k, int n, double p)
{
    double dk, dn;

    if (((p < 0.0) || (p > 1.0)) || k < 0)
    {
        return CUDART_NAN;
    }

    dk = k + 1;
    dn = n;
    return (incbet(dk, dn, 1.0 - p));
}

__device__ double nbdtrc_unsafe(double k, double n, double p)
{
    if (isnan(k) || isnan(n))
    {
        return CUDART_NAN;
    }
    return nbdtrc((int)k, (int)n, p);
}

"""


nbdtri_definition = """

__device__ double nbdtri(int k, int n, double y)
{
    double dk, dn, w;

    if (((y < 0.0) || (y > 1.0)) || (k < 0)) {
        return CUDART_NAN;
    }
    dk = k + 1;
    dn = n;
    w = incbi(dn, dk, y);
    return (w);
}

__device__ double nbdtri_unsafe(double k, double n, double y)
{
    if (isnan(k) || isnan(n))
    {
        return CUDART_NAN;
    }
    return nbdtri((int)k, (int)n, y);
}

"""

# Note: as in scipy we have a safe iid->d version and unsafe ddd->d one
nbdtr = _core.create_ufunc(
    "cupyx_scipy_nbdtr",
    (
        'lld->d',
        ('fff->f', 'out0 = out0_type(nbdtr_unsafe(in0, in1, in2));'),
        ('ddd->d', 'out0 = nbdtr_unsafe(in0, in1, in2);'),
    ),
    "out0 = nbdtr(in0, in1, in2);",
    preamble=incbet_preamble + nbdtr_definition,
    doc="""Negative binomial distribution cumulative distribution function.

    Parameters
    ----------
    k : cupy.ndarray
        The maximum number of allowed failures (nonnegative int).
    n : cupy.ndarray
        The target number of successes (positive int).
    p : cupy.ndarray
        Probability of success in a single event (float).

    Returns
    -------
    F : cupy.ndarray
        The probability of `k` or fewer failures before `n` successes in a
        sequence of events with individual success probability `p`.

    See Also
    --------
    :func:`scipy.special.nbdtr`

    """,
)


nbdtrc = _core.create_ufunc(
    "cupyx_scipy_nbdtrc",
    (
        'lld->d',
        ('fff->f', 'out0 = out0_type(nbdtrc_unsafe(in0, in1, in2));'),
        ('ddd->d', 'out0 = nbdtrc_unsafe(in0, in1, in2);'),
    ),
    "out0 = nbdtrc(in0, in1, in2);",
    preamble=incbet_preamble + nbdtrc_definition,
    doc="""Negative binomial distribution survival function.

    Parameters
    ----------
    k : cupy.ndarray
        The maximum number of allowed failures (nonnegative int).
    n : cupy.ndarray
        The target number of successes (positive int).
    p : cupy.ndarray
        Probability of success in a single event (float).

    Returns
    -------
    F : cupy.ndarray
        The probability of ``k + 1`` or more failures before `n` successes in a
        sequence of events with individual success probability `p`.

    See Also
    --------
    :func:`scipy.special.nbdtrc`

    """,
)

nbdtri = _core.create_ufunc(
    "cupyx_scipy_nbdtri",
    (
        'lld->d',
        ('fff->f', 'out0 = out0_type(nbdtri_unsafe(in0, in1, in2));'),
        ('ddd->d', 'out0 = nbdtri_unsafe(in0, in1, in2);'),
    ),
    "out0 = nbdtri(in0, in1, in2);",
    preamble=incbi_preamble + nbdtri_definition,
    doc="""Inverse function to `nbdtr` with respect to `p`.

    Parameters
    ----------
    k : cupy.ndarray
        The maximum number of allowed failures (nonnegative int).
    n : cupy.ndarray
        The target number of successes (positive int).
    y : cupy.ndarray
        The probability of `k` or fewer failures before `n` successes (float).

    Returns
    -------
    p : cupy.ndarray
        Probability of success in a single event (float) such that
        ``nbdtr(k, n, p) = y``.

    See Also
    --------
    :func:`scipy.special.nbdtri`

    """,
)


# Poisson distribution functions

pdtr_definition = """

__device__ double pdtr(double k, double m)
{
    double v;

    if ((k < 0) || (m < 0)) {
        return CUDART_NAN;
    }
    if (m == 0.0) {
        return 1.0;
    }
    v = floor(k) + 1;
    return igamc(v, m);
}

"""


pdtrc_definition = """

__device__ double pdtrc(double k, double m)
{
    double v;

    if ((k < 0.0) || (m < 0.0)) {
        return CUDART_NAN;
    }
    if (m == 0.0) {
        return 0.0;
    }
    v = floor(k) + 1;
    return igam(v, m);
}

"""


pdtri_definition = """

__device__ double pdtri(int k, double y)
{
    double v;

    if ((k < 0) || (y < 0.0) || (y >= 1.0)) {
        return CUDART_NAN;
    }
    v = k + 1;
    return igamci(v, y);
}

__device__ double pdtri_unsafe(double k, double y)
{
    if (isnan(k)) {
        return CUDART_NAN;
    } else {
        return pdtri((int)k, y);
    }
}

"""


pdtr = _core.create_ufunc(
    "cupyx_scipy_pdtr",
    ('ff->f', 'dd->d'),
    "out0 = out0_type(pdtr(in0, in1));",
    preamble=_igam_preamble + pdtr_definition,
    doc="""Poisson cumulative distribution function.

    Parameters
    ----------
    k : cupy.ndarray
        Nonnegative real argument.
    m : cupy.ndarray
        Nonnegative real shape parameter.

    Returns
    -------
    y : cupy.ndarray
        Values of the Poisson cumulative distribution function.

    See Also
    --------
    :func:`scipy.special.pdtr`

    """,
)


pdtrc = _core.create_ufunc(
    "cupyx_scipy_pdtrc",
    ('ff->f', 'dd->d'),
    "out0 = out0_type(pdtrc(in0, in1));",
    preamble=_igam_preamble + pdtrc_definition,
    doc="""Binomial distribution survival function.

    Returns the complemented binomial distribution function (the integral of
    the density from x to infinity).

    Parameters
    ----------
    k : cupy.ndarray
        Nonnegative real argument.
    m : cupy.ndarray
        Nonnegative real shape parameter.

    Returns
    -------
    y : cupy.ndarray
        The sum of the terms from k+1 to infinity of the Poisson
        distribution.

    See Also
    --------
    :func:`scipy.special.pdtrc`

    """,
)


pdtri = _core.create_ufunc(
    "cupyx_scipy_pdtri",
    # Note order of entries here is important to match SciPy behavior
    ('ld->d',
     ('ff->f', 'out0 = out0_type(pdtri_unsafe(in0, in1));'),
     ('dd->d', 'out0 = pdtri_unsafe(in0, in1);')),
    "out0 = pdtri((int)in0, in1);",
    preamble=_igami_preamble + pdtri_definition,
    doc="""Inverse function to `pdtr` with respect to `m`.

    Parameters
    ----------
    k : cupy.ndarray
        Nonnegative real argument.
    y : cupy.ndarray
        Cumulative probability.

    Returns
    -------
    m : cupy.ndarray
        The Poisson variable `m` such that the sum from 0 to `k` of the Poisson
        density is equal to the given probability `y`.

    See Also
    --------
    :func:`scipy.special.pdtri`

    """,
)

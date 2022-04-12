from cupy import _core


logit_definition = """
template <typename T>
static __device__ T logit(T x) {
    x /= 1 - x;
    return log(x);
}

"""


logit = _core.create_ufunc(
    'cupy_logit',
    ('e->f', 'f->f', 'd->d'),
    'out0 = logit(in0)',
    preamble=logit_definition,
    doc='''Logit function.

    Args:
        x (cupy.ndarray): input data

    Returns:
        cupy.ndarray: values of logit(x)

    .. seealso:: :data:`scipy.special.logit`

    ''')


expit_definition = """
template <typename T>
static __device__ T expit(T x) {
    return 1 / (1 + exp(-x));
}

"""


expit = _core.create_ufunc(
    'cupy_expit',
    ('e->f', 'f->f', 'd->d'),
    'out0 = expit(in0)',
    preamble=expit_definition,
    doc='''Logistic sigmoid function (expit).

    Args:
        x (cupy.ndarray): input data (must be real)

    Returns:
        cupy.ndarray: values of expit(x)

    .. seealso:: :data:`scipy.special.expit`

    .. note::
        expit is the inverse of logit.

    ''')


# log_expit implemented based on log1p as in SciPy's scipy/special/_logit.h

log_expit_definition = """
template <typename T>
static __device__ T log_expit(T x)
{
    if (x < 0.0) {
        return x - log1p(exp(x));
    } else {
        return -log1p(exp(-x));
    }
}

"""


log_expit = _core.create_ufunc(
    'cupy_log_expit',
    ('e->f', 'f->f', 'd->d'),
    'out0 = log_expit(in0)',
    preamble=log_expit_definition,
    doc='''Logarithm of the logistic sigmoid function.

    Args:
        x (cupy.ndarray): input data (must be real)

    Returns:
        cupy.ndarray: values of log(expit(x))

    .. seealso:: :data:`scipy.special.log_expit`

    .. note::
        The function is mathematically equivalent to ``log(expit(x))``, but
        is formulated to avoid loss of precision for inputs with large
        (positive or negative) magnitude.
    ''')


boxcox_definition = """
static __device__ double boxcox(double x, double lmbda) {
    // if lmbda << 1 and log(x) < 1.0, the lmbda*log(x) product can lose
    // precision, furthermore, expm1(x) == x for x < eps.
    // For doubles, the range of log is -744.44 to +709.78, with eps being
    // the smallest value produced.  This range means that we will have
    // abs(lmbda)*log(x) < eps whenever abs(lmbda) <= eps/-log(min double)
    // which is ~2.98e-19.
    if (fabs(lmbda) < 1e-19) {
        return log(x);
    } else {
        return expm1(lmbda * log(x)) / lmbda;
    }
}

"""


boxcox = _core.create_ufunc(
    'cupy_boxcox',
    ('ee->f', 'ff->f', 'dd->d'),
    'out0 = out0_type(boxcox(in0, in1))',
    preamble=boxcox_definition,
    doc='''Compute the Box-Cox transformation.

    Args:
        x (cupy.ndarray): input data (must be real)

    Returns:
        cupy.ndarray: values of boxcox(x)

    .. seealso:: :data:`scipy.special.boxcox`

    ''')


boxcox1p_definition = """
static __device__ double boxcox1p(double x, double lmbda) {
    // The argument given above in boxcox applies here with the modification
    // that the smallest value produced by log1p is the minimum representable
    // value, rather than eps.  The second condition here prevents underflow
    // when log1p(x) is < eps.
    double lgx = log1p(x);
    if ((fabs(lmbda) < 1e-19)
        || ((fabs(lgx) < 1e-289) && (fabs(lmbda) < 1e273))) {
        return lgx;
    } else {
        return expm1(lmbda * lgx) / lmbda;
    }
}

"""


boxcox1p = _core.create_ufunc(
    'cupy_boxcox1p',
    ('ee->f', 'ff->f', 'dd->d'),
    'out0 = out0_type(boxcox1p(in0, in1))',
    preamble=boxcox1p_definition,
    doc='''Compute the Box-Cox transformation op 1 + `x`.

    Args:
        x (cupy.ndarray): input data (must be real)

    Returns:
        cupy.ndarray: values of boxcox1p(x)

    .. seealso:: :data:`scipy.special.boxcox1p`

    ''')


inv_boxcox_definition = """
static __device__ double inv_boxcox(double x, double lmbda) {
    if (lmbda == 0.0) {
        return exp(x);
    } else {
        return exp(log1p(lmbda * x) / lmbda);
    }
}

"""


inv_boxcox = _core.create_ufunc(
    'cupy_inv_boxcox',
    ('ee->f', 'ff->f', 'dd->d'),
    'out0 = out0_type(inv_boxcox(in0, in1))',
    preamble=inv_boxcox_definition,
    doc='''Compute the Box-Cox transformation.

    Args:
        x (cupy.ndarray): input data (must be real)

    Returns:
        cupy.ndarray: values of inv_boxcox(x)

    .. seealso:: :data:`scipy.special.inv_boxcox`

    ''')


inv_boxcox1p_definition = """
static __device__ double inv_boxcox1p(double x, double lmbda) {
    if (lmbda == 0.0) {
        return expm1(x);
    } else if (fabs(lmbda * x) < 1e-154) {
        return x;
    } else {
        return expm1(log1p(lmbda * x) / lmbda);
    }
}

"""


inv_boxcox1p = _core.create_ufunc(
    'cupy_inv_boxcox1p',
    ('ee->f', 'ff->f', 'dd->d'),
    'out0 = out0_type(inv_boxcox1p(in0, in1))',
    preamble=inv_boxcox1p_definition,
    doc='''Compute the Box-Cox transformation op 1 + `x`.

    Args:
        x (cupy.ndarray): input data (must be real)

    Returns:
        cupy.ndarray: values of inv_boxcox1p(x)

    .. seealso:: :data:`scipy.special.inv_boxcox1p`
''')

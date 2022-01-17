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


ndtr = _core.create_ufunc(
    'cupyx_scipy_special_ndtr', ('f->f', 'd->d'),
    'out0 = normcdf(in0)',
    doc='''Cumulative distribution function of normal distribution.

    .. seealso:: :meth:`scipy.special.ndtr`

    ''')

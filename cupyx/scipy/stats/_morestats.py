import cupy


def boxcox_llf(lmb, data):
    """The boxcox log-likelihood function.

    Parameters
    ----------
    lmb : scalar
        Parameter for Box-Cox transformation
    data : array-like
        Data to calculate Box-Cox log-likelihood for. If
        `data` is multi-dimensional, the log-likelihood
        is calculated along the first axis

    Returns
    -------
    llf : float or cupy.ndarray
        Box-Cox log-likelihood of `data` given `lmb`. A float
        for 1-D `data`, an array otherwise

    See Also
    --------
    scipy.stats.boxcox_llf

    """

    if data.ndim == 1 and data.dtype == cupy.float16:
        data = data.astype(cupy.float64)
    if data.ndim == 1 and data.dtype == cupy.float32:
        data = data.astype(cupy.float64)
    if data.ndim == 1 and data.dtype == cupy.complex64:
        data = data.astype(cupy.complex128)

    N = data.shape[0]
    if N == 0:
        return cupy.array(cupy.nan)

    logdata = cupy.log(data)

    # Compute the variance of the transformed data
    if lmb == 0:
        variance = cupy.var(logdata, axis=0)
    else:
        variance = cupy.var(data**lmb / lmb, axis=0)

    return (lmb - 1) * cupy.sum(logdata, axis=0) - N/2 * cupy.log(variance)

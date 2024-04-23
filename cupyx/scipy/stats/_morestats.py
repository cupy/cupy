import cupy
from cupyx.scipy import special


def _log_mean(logx):
    # compute log of mean of x from log(x)
    return special.logsumexp(logx, axis=0) - cupy.log(len(logx))


def _log_var(logx):
    # compute log of variance of x from log(x)
    neg_logmean = cupy.broadcast_to(_log_mean(logx) - cupy.pi * 1j, logx.shape)
    logxmu = special.logsumexp(cupy.asarray([logx, neg_logmean]), axis=0)
    return special.logsumexp(2 * logxmu, axis=0).real - cupy.log(len(logx))


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
        logvar = cupy.log(variance)
    else:
        logx = lmb * logdata - cupy.log(abs(lmb))
        logvar = _log_var(logx)

    return (lmb - 1) * cupy.sum(logdata, axis=0) - N/2 * logvar


def kstat(data, n=2):
    """
    Return the nth k-statistic (1<=n<=4 so far).

    The nth k-statistic k_n is the unique symmetric unbiased estimator of the
    nth cumulant kappa_n.

    Parameters
    ----------
    data : array_like
        Input array. Note that n-D input gets flattened.
    n : int, {1, 2, 3, 4}, optional
        Default is equal to 2.

    Returns
    -------
    kstat : float
        The nth k-statistic.

    See Also
    --------
    scipy.stats.kstat
    """
    if n > 4 or n < 1:
        raise ValueError("k-statistics only supported for 1<=n<=4")
    n = int(n)
    S = cupy.zeros(n + 1, cupy.float64)
    data = data.ravel()
    N = data.size

    # raise ValueError on empty input
    if N == 0:
        raise ValueError("Data input must not be empty")

    # on nan input, return nan without warning
    if cupy.isnan(cupy.sum(data)):
        return cupy.nan
    
    for k in range(1, n + 1):
        S[k] = cupy.sum(data**k, axis=0)
    if n == 1:
        return S[1] * 1.0/N
    elif n == 2:
        return (N*S[2] - S[1]**2.0) / (N*(N - 1.0))
    elif n == 3:
        return (2*S[1]**3 - 3*N*S[1]*S[2] + N*N*S[3]) / (N*(N - 1.0)*(N - 2.0))
    elif n == 4:
        return ((-6*S[1]**4 + 12*N*S[1]**2 * S[2] - 3*N*(N-1.0)*S[2]**2 -
                 4*N*(N+1)*S[1]*S[3] + N*N*(N+1)*S[4]) /
                (N*(N-1.0)*(N-2.0)*(N-3.0)))
    else:
        raise ValueError("Should not be here.")



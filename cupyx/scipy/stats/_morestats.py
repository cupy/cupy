import math

import cupy
from cupyx.scipy import special


def _log_mean(logx):
    # compute log of mean of x from log(x)
    return special.logsumexp(logx, axis=0) - math.log(len(logx))


def _log_var(logx):
    # compute log of variance of x from log(x)
    neg_logmean = cupy.broadcast_to(_log_mean(logx) - cupy.pi * 1j, logx.shape)
    logxmu = special.logsumexp(cupy.asarray([logx, neg_logmean]), axis=0)
    return special.logsumexp(2 * logxmu, axis=0).real - math.log(len(logx))


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

    if data.dtype.kind in "biu":
        data = data.astype(cupy.float64)

    dtype = data.dtype

    if data.dtype == cupy.float16:
        # Avoid large numerical errors in float16
        data = data.astype(cupy.float32)

    N = data.shape[0]
    if N == 0:
        return cupy.array(cupy.nan)

    logdata = cupy.log(data)

    # Compute the variance of the transformed data
    if lmb == 0:
        variance = cupy.var(logdata, axis=0)
        logvar = cupy.log(variance)
    else:
        logx = lmb * logdata
        logvar = _log_var(logx) - 2 * math.log(abs(lmb))

    res = (lmb - 1) * cupy.sum(logdata, axis=0) - N/2 * logvar
    return res.astype(dtype, copy=False)

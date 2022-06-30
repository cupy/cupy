import cupy as cp


_log_softmax_kernel = cp._core.ReductionKernel(
    'T x1',
    'T y',
    'exp(x1)',
    'a + b',
    'y = log(a)',
    '0',
    name='log_softmax'
)


def log_softmax(x, axis=None):
    """Compute logarithm of softmax function

    Parameters
    ----------
    x : array-like
        Input array
    axis : int or tuple of ints, optional
        Axis to compute values along. Default is None and softmax
        will be  computed over the entire array `x`

    Returns
    -------
    s : cupy.ndarry
        An array with the same shape as `x`. Exponential of the
        result will sum to 1 along the specified axis. If `x` is a
        scalar, a scalar is returned

    """

    if x.dtype == cp.int8:
        x = x.astype(cp.float16)
    elif x.dtype == cp.int16:
        x = x.astype(cp.float32)
    elif x.dtype == cp.int32:
        x = x.astype(cp.float64)
    elif x.dtype == cp.int64:
        x = x.astype(cp.float64)

    x_max = cp.amax(x, axis=axis, keepdims=True)

    if x_max.ndim > 0:
        x_max[~cp.isfinite(x_max)] = 0
    elif not cp.isfinite(x_max):
        x_max = 0

    tmp = x - x_max

    out = _log_softmax_kernel(tmp, axis=axis, keepdims=True)

    out = tmp - out
    return out

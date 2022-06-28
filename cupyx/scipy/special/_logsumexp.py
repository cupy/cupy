import cupy as cp


__all__ = ["log_softmax"]


_log_preamble = '''

template <typename T>
__device__ T _log(T a)
{
    return log(a);
}

'''


_preamble = _log_preamble+'''

template <typename T>
__device__ T _post_map(T a, T* y)
{
    *y = _log(static_cast<float>(a));
    return *y;
}

'''


_log_softmax_kernel = cp._core.ReductionKernel(
   'T x1',
   'T y',
   'exp(static_cast<float>(x1))',
   'a + b',
   'y = _post_map(a, &y)',
   '0',
   name='log_softmax',
   preamble=_preamble
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
    s : cupy.ndarry or scalar
        An array with the same shape as `x`. Exponential of the
        result will sum to 1 along the specified axis. If `x` is a
        scalar, a scalar is returned

    """

    x_max = cp.amax(x, axis=axis, keepdims=True)

    if x_max.ndim > 0:
        x_max[~cp.isfinite(x_max)] = 0
    elif not cp.isfinite(x_max):
        x_max = 0

    tmp = x - x_max

    out = _log_softmax_kernel(tmp, axis=axis, keepdims=True)

    out = tmp - out
    return out

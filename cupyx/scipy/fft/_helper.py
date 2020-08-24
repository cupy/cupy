import math


_next_fast_len_cache = {}


def _next_fast_len_impl(n, primes):
    if len(primes) == 0:
        return math.inf
    result = _next_fast_len_cache.get((n, primes), None)
    if result is None:
        if n == 1:
            result = 1
        else:
            p = primes[0]
            result = min(
                _next_fast_len_impl((n + p - 1) // p, primes) * p,
                _next_fast_len_impl(n, primes[1:]))
        _next_fast_len_cache[(n, primes)] = result
    return result


def next_fast_len(target, real=False):
    """Find the next fast size to ``fft``.

    Args:
        target (int): The size of input array.
        real (bool): ``True`` if the FFT involves real input or output.
            This parameter is of no use, and only for compatibility to
            SciPy's interface.

    Returns:
        int: The smallest fast length greater than or equal to the input value.

    .. seealso:: :func:`scipy.fft.next_fast_len`

    .. note::
        It may return a different value to :func:`scipy.fft.next_fast_len`
        as pocketfft's prime factors are different from cuFFT's factors.
        For details, see the `cuFFT documentation`_.

    .. _cuFFT documentation:
        https://docs.nvidia.com/cuda/cufft/index.html#accuracy-and-performance
    """
    if target == 0:
        return 0

    primes = (2, 3, 5, 7)
    return _next_fast_len_impl(target, primes)

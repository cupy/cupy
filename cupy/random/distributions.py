import cupy
from cupy.random import generator


# TODO(beam2d): Implement many distributions


def gumbel(loc=0.0, scale=1.0, size=None, dtype=float):
    """Returns an array of samples drawn from a Gumbel distribution.

    The samples are drawn from a Gumbel distribution with location ``loc``
    and scale ``scale``.
    Its probability density function is defined as

    .. math::
       f(x) = \\frac{1}{\\eta} \
           \\exp\\left\\{ - \\frac{x - \\mu}{\\eta} \\right\\} \
           \\exp\\left[-\\exp\\left\\{-\\frac{x - \\mu}{\\eta} \
           \\right\\}\\right],

    where :math:`\\mu` is ``loc`` and :math:`\\eta` is ``scale``.

    Args:
        loc (float): The location of the mode :math:`\\mu`.
        scale (float): The scale parameter :math:`\\eta`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the Gumbel destribution.

    .. seealso::
        :func:`cupy.random.RandomState.gumbel`
        :func:`numpy.random.gumbel`
    """
    rs = generator.get_random_state()
    return rs.gumbel(loc, scale, size, dtype)


def laplace(loc=0.0, scale=1.0, size=None, dtype=float):
    """Laplace distribution.

    Returns an array of samples drawn from the laplace distribution. Its
    probability density function is defined as

    .. math::
       f(x) = \\frac{1}{2b}\\exp\\left(-\\frac{|x-\\mu|}{b}\\right),

    Args:
        loc (float): The location of the mode :math:`\\mu`.
        scale (float): The scale parameter :math:`b`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the laplace destribution.

    .. seealso::
        :func:`cupy.random.RandomState.laplace`
        :func:`numpy.random.laplace`
    """
    rs = generator.get_random_state()
    return rs.laplace(loc, scale, size, dtype)


def lognormal(mean=0.0, sigma=1.0, size=None, dtype=float):
    """Returns an array of samples drawn from a log normal distribution.

    The samples are natural log of samples drawn from a normal distribution
    with mean ``mean`` and deviation ``sigma``.

    Args:
        mean (float): Mean of the normal distribution.
        sigma (float): Standard deviation of the normal distribution.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the log normal distribution.

    .. seealso:: :func:`numpy.random.lognormal`

    """
    rs = generator.get_random_state()
    return rs.lognormal(mean, sigma, size=size, dtype=dtype)


def normal(loc=0.0, scale=1.0, size=None, dtype=float):
    """Returns an array of normally distributed samples.

    Args:
        loc (float or array_like of floats): Mean of the normal distribution.
        scale (float or array_like of floats):
            Standard deviation of the normal distribution.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Normally distributed samples.

    .. seealso:: :func:`numpy.random.normal`

    """
    rs = generator.get_random_state()
    x = rs.normal(0, 1, size, dtype)
    cupy.multiply(x, scale, out=x)
    cupy.add(x, loc, out=x)
    return x


def multivariate_normal(mean, cov, size=None, check_valid='ignore', tol=1e-8,
                        dtype=float):
    """Multivariate normal distribution.

    Returns an array of samples drawn from the multivariate normal
    distribution. Its probability density function is defined as

    .. math::
       f(x) = \\frac{1}{(2\\pi|\\Sigma|)^(n/2)} \
           \\exp\\left(-\\frac{1}{2} \
           (x-\\mu)^{\\top}\\Sigma^{-1}(x-\\mu)\\right),

    Args:
        mean (1-D array_like, of length N): Mean of the multivariate normal
            distribution :math:`\\mu`.
        cov (2-D array_like, of shape (N, N)): Covariance matrix of the
            multivariate normal distribution. It must be symmetric and
            positive-semidefinite for proper sampling :math:`\\Sigma`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        check_valid ('warn', 'raise', 'ignore'): Behavior when the covariance
            matrix is not positive semidefinite.
        tol (float): Tolerance when checking the singular values in
            covariance matrix.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the multivariate normal distribution.

    .. seealso:: :func:`numpy.random.multivariate_normal`

    """
    rs = generator.get_random_state()
    x = rs.multivariate_normal(mean, cov, size, check_valid, tol, dtype)
    return x


def standard_normal(size=None, dtype=float):
    """Returns an array of samples drawn from the standard normal distribution.

    This is a variant of :func:`cupy.random.randn`.

    Args:
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier.

    Returns:
        cupy.ndarray: Samples drawn from the standard normal distribution.

    .. seealso:: :func:`numpy.random.standard_normal`

    """
    return normal(size=size, dtype=dtype)


def uniform(low=0.0, high=1.0, size=None, dtype=float):
    """Returns an array of uniformly-distributed samples over an interval.

    Samples are drawn from a uniform distribution over the half-open interval
    ``[low, high)``.

    Args:
        low (float): Lower end of the interval.
        high (float): Upper end of the interval.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier.

    Returns:
        cupy.ndarray: Samples drawn from the uniform distribution.

    .. seealso:: :func:`numpy.random.uniform`

    """
    rs = generator.get_random_state()
    x = rs.uniform(0.0, 1.0, size=size, dtype=dtype)
    cupy.multiply(x, (high - low), out=x)
    cupy.add(x, low, out=x)
    return x

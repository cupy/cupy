import cupy
from cupy.random import generator


# TODO(beam2d): Implement many distributions


def beta(a, b, size=None, dtype=float):
    """Beta distribution.

    Returns an array of samples drawn from the beta distribution. Its
    probability density function is defined as

    .. math::
       f(x) = \\frac{x^{\\alpha-1}(1-x)^{\\beta-1}}{B(\\alpha,\\beta)}

    Args:
        a (float): Parameter of the beta distribution :math:`\\alpha`.
        b (float): Parameter of the beta distribution :math:`\\beta`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the beta destribution.

    .. seealso::
        :func:`cupy.random.RandomState.beta`
        :func:`numpy.random.beta`
    """
    rs = generator.get_random_state()
    return rs.beta(a, b, size, dtype)


def binomial(n, p, size=None, dtype=int):
    """Binomial distribution.

    Returns an array of samples drawn from the binomial distribution. Its
    probability mass function is defined as

    .. math::
        f(x) = \\binom{n}{x}p^x(1-p)^{n-x}

    Args:
        n (int): Trial number of the binomial distribution.
        p (float): Success probability of the binomial distribution.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.int32` and
            :class:`numpy.int64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the binomial destribution.

    .. seealso::
        :func:`cupy.random.RandomState.binomial`
        :func:`numpy.random.binomial`
    """
    rs = generator.get_random_state()
    return rs.binomial(n, p, size, dtype)


def chisquare(df, size=None, dtype=float):
    """Chi-square distribution.

    Returns an array of samples drawn from the chi-square distribution. Its
    probability density function is defined as

    .. math::
       f(x) = \\frac{(1/2)^{k/2}}{\\Gamma(k/2)}x^{k/2-1}e^{-x/2}

    Args:
        df (int): Degree of freedom :math:`k`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the chi-square distribution.

    .. seealso::
        :func:`cupy.random.RandomState.chisquare`
        :func:`numpy.random.chisquare`
    """
    rs = generator.get_random_state()
    return rs.chisquare(df, size, dtype)


def dirichlet(alpha, size=None, dtype=float):
    """Returns an array of samples drawn from a Dirichlet distribution.

    Its probability density function is defined as

    .. math::

    Args:
        alpha (int):
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the Dirichret destribution.

    .. seealso::
        :func:`cupy.random.RandomState.dirichlet`
        :func:`numpy.random.dirichlet`
    """
    rs = generator.get_random_state()
    return rs.dirichlet(alpha, size, dtype)


def f(dfnum, dfden, size=None, dtype=float):
    """Returns an array of samples drawn from the F distribution.

    Its probability density function is defined as

    .. math::

    Args:
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the F destribution.

    .. seealso::
        :func:`cupy.random.RandomState.f`
        :func:`numpy.random.f`
    """
    rs = generator.get_random_state()
    return rs.f(dfnum, dfden, size, dtype)


def gamma(shape, scale=1.0, size=None, dtype=float):
    """Returns an array of samples drawn from a Gamma distribution.

    Its probability density function is defined as

    .. math::
       f(x) = \\frac{1}{\\Gamma(k)\\theta^k}x^{k-1}\\mathrm{e}^{-x/\\theta}

    Args:
        shape (float):
        scale (float):
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the Gamma destribution.

    .. seealso::
        :func:`cupy.random.RandomState.gamma`
        :func:`numpy.random.gamma`
    """
    rs = generator.get_random_state()
    return rs.gamma(shape, scale, size, dtype)


def geometric(p, size=None, dtype=int):
    """Returns an array of samples drawn from a Geometric distribution.

    Its probability mass function is defined as

    .. math::

    Args:
        p (float):
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the Geometric destribution.

    .. seealso::
        :func:`cupy.random.RandomState.geometric`
        :func:`numpy.random.geometric`
    """
    rs = generator.get_random_state()
    return rs.geometric(p, size, dtype)


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


def multivariate_normal(mean, cov, size=None, check_valid='warn', tol=1e-8,
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
        check_valid (‘warn’, ‘raise’, ‘ignore’): Behavior when the covariance
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


def pareto(a, size=None, dtype=float):
    """Returns an array of samples drawn from a pareto distribution.

    Args:
        a (float):
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the pareto distribution.

    .. seealso:: :func:`numpy.random.pareto`

    """
    rs = generator.get_random_state()
    x = rs.pareto(a, size, dtype)
    return x


def poisson(lam=1.0, size=None, dtype=int):
    """Returns an array of samples drawn from a poisson distribution.

    Args:
        lam (float):
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the poisson distribution.

    .. seealso:: :func:`numpy.random.poisson`

    """
    rs = generator.get_random_state()
    x = rs.poisson(lam, size, dtype)
    return x


def standard_cauchy(size=None, dtype=float):
    """Returns an array of samples drawn from the standard cauchy distribution.

    Args:
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the standard cauchy distribution.

    .. seealso:: :func:`numpy.random.standard_cauchy`

    """
    rs = generator.get_random_state()
    x = rs.standard_cauchy(size, dtype)
    return x


def standard_exponential(size=None, dtype=float):
    """Standard Exponential

    Returns an array of samples drawn from the standard exponential
    distribution.

    Args:
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the standard exponential distribution.

    .. seealso:: :func:`numpy.random.standard_exponential`

    """
    rs = generator.get_random_state()
    x = rs.standard_exponential(size, dtype)
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


def standard_t(df, size=None, dtype=float):
    """Returns an array of samples drawn from a standard Student's t distribution.

    Its probability density function is defined as

    .. math::

    Args:
        df (float):
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the standard Student's t destribution.

    .. seealso::
        :func:`cupy.random.RandomState.standard_t`
        :func:`numpy.random.standard_t`
    """
    rs = generator.get_random_state()
    return rs.standard_t(df, size, dtype)


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
    return rs.uniform(low, high, size=size, dtype=dtype)

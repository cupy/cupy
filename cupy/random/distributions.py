import cupy
from cupy.random import generator
from cupy import util


# TODO(beam2d): Implement many distributions


def beta(a, b, size=None, dtype=float):
    """Beta distribution.

    Returns an array of samples drawn from the beta distribution. Its
    probability density function is defined as

    .. math::
       f(x) = \\frac{x^{\\alpha-1}(1-x)^{\\beta-1}}{B(\\alpha,\\beta)},

    Args:
        a (float): Parameter of the beta distribution :math:`\\alpha`.
        b (float): Parameter of the beta distribution :math:`\\beta`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the beta distribution.

    .. seealso::
        :func:`numpy.random.beta`
    """
    rs = generator.get_random_state()
    return rs.beta(a, b, size, dtype)


def binomial(n, p, size=None, dtype=int):
    """Binomial distribution.

    Returns an array of samples drawn from the binomial distribution. Its
    probability mass function is defined as

    .. math::
        f(x) = \\binom{n}{x}p^x(1-p)^{n-x},

    Args:
        n (int): Trial number of the binomial distribution.
        p (float): Success probability of the binomial distribution.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.int32` and
            :class:`numpy.int64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the binomial distribution.

    .. seealso::
        :func:`numpy.random.binomial`
    """
    rs = generator.get_random_state()
    return rs.binomial(n, p, size, dtype)


def chisquare(df, size=None, dtype=float):
    """Chi-square distribution.

    Returns an array of samples drawn from the chi-square distribution. Its
    probability density function is defined as

    .. math::
       f(x) = \\frac{(1/2)^{k/2}}{\\Gamma(k/2)}x^{k/2-1}e^{-x/2},

    Args:
        df (int or array_like of ints): Degree of freedom :math:`k`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the chi-square distribution.

    .. seealso::
        :func:`numpy.random.chisquare`
    """
    rs = generator.get_random_state()
    return rs.chisquare(df, size, dtype)


def dirichlet(alpha, size=None, dtype=float):
    """Dirichlet distribution.

    Returns an array of samples drawn from the dirichlet distribution. Its
    probability density function is defined as

    .. math::
        f(x) = \\frac{\\Gamma(\\sum_{i=1}^K\\alpha_i)} \
            {\\prod_{i=1}^{K}\\Gamma(\\alpha_i)} \
            \\prod_{i=1}^Kx_i^{\\alpha_i-1},

    Args:
        alpha (array): Parameters of the dirichlet distribution
            :math:`\\alpha`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the dirichlet distribution.

    .. seealso::
        :func:`numpy.random.dirichlet`
    """
    rs = generator.get_random_state()
    return rs.dirichlet(alpha, size, dtype)


def exponential(scale, size=None, dtype=float):
    """Exponential distribution.

    Returns an array of samples drawn from the exponential distribution. Its
    probability density function is defined as

    .. math::
       f(x) = \\frac{1}{\\beta}\\exp (-\\frac{x}{\\beta}),

    Args:
        scale (float or array_like of floats): The scale parameter
            :math:`\\beta`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the exponential distribution.

    .. seealso::
        :func:`numpy.random.exponential`
    """
    rs = generator.get_random_state()
    return rs.exponential(scale, size, dtype)


def f(dfnum, dfden, size=None, dtype=float):
    """F distribution.

    Returns an array of samples drawn from the f distribution. Its probability
    density function is defined as

    .. math::
        f(x) = \\frac{1}{B(\\frac{d_1}{2},\\frac{d_2}{2})} \
            \\left(\\frac{d_1}{d_2}\\right)^{\\frac{d_1}{2}} \
            x^{\\frac{d_1}{2}-1} \
            \\left(1+\\frac{d_1}{d_2}x\\right) \
            ^{-\\frac{d_1+d_2}{2}},

    Args:
        dfnum (float or array_like of floats): Parameter of the f distribution
            :math:`d_1`.
        dfden (float or array_like of floats): Parameter of the f distribution
            :math:`d_2`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the f distribution.

    .. seealso::
        :func:`numpy.random.f`
    """
    rs = generator.get_random_state()
    return rs.f(dfnum, dfden, size, dtype)


def gamma(shape, scale=1.0, size=None, dtype=float):
    """Gamma distribution.

    Returns an array of samples drawn from the gamma distribution. Its
    probability density function is defined as

    .. math::
       f(x) = \\frac{1}{\\Gamma(k)\\theta^k}x^{k-1}e^{-x/\\theta},

    Args:
        shape (array): Parameter of the gamma distribution :math:`k`.
        scale (array): Parameter of the gamma distribution :math:`\\theta`
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:cupy.ndarray: Samples drawn from the gamma distribution.

    .. seealso::
        :func:`numpy.random.gamma`
    """
    rs = generator.get_random_state()
    return rs.gamma(shape, scale, size, dtype)


def geometric(p, size=None, dtype=int):
    """Geometric distribution.

    Returns an array of samples drawn from the geometric distribution. Its
    probability mass function is defined as

    .. math::
        f(x) = p(1-p)^{k-1},

    Args:
        p (float): Success probability of the geometric distribution.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.int32` and
            :class:`numpy.int64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the geometric distribution.

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
        cupy.ndarray: Samples drawn from the Gumbel distribution.

    .. seealso::
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
        cupy.ndarray: Samples drawn from the laplace distribution.

    .. seealso::
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
    """(experimental) Multivariate normal distribution.

    Returns an array of samples drawn from the multivariate normal
    distribution. Its probability density function is defined as

    .. math::
       f(x) = \\frac{1}{(2\\pi|\\Sigma|)^(n/2)} \
           \\exp\\left(-\\frac{1}{2} \
           (x-\\mu)^{\\top}\\Sigma^{-1}(x-\\mu)\\right),

    Args:
        mean (1-D array_like, of length N): Mean of the multivariate normal
            distribution :math:`\\mu`.
        cov (2-D array_like, of shape (N, N)): Covariance matrix
            :math:`\\Sigma` of the multivariate normal distribution. It must be
            symmetric and positive-semidefinite for proper sampling.
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
    util.experimental('cupy.random.multivariate_normal')
    rs = generator.get_random_state()
    x = rs.multivariate_normal(mean, cov, size, check_valid, tol, dtype)
    return x


def pareto(a, size=None, dtype=float):
    """Pareto II or Lomax distribution.

    Returns an array of samples drawn from the Pareto II distribution. Its
    probability density function is defined as

    .. math::
        f(x) = \\alpha(1+x)^{-(\\alpha+1)},

    Args:
        a (float or array_like of floats): Parameter of the Pareto II
            distribution :math:`\\alpha`.
        size (int or tuple of ints): The shape of the array. If ``None``, this
            function generate an array whose shape is `a.shape`.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the Pareto II distribution.

    .. seealso:: :func:`numpy.random.pareto`
    """
    rs = generator.get_random_state()
    x = rs.pareto(a, size, dtype)
    return x


def poisson(lam=1.0, size=None, dtype=int):
    """Poisson distribution.

    Returns an array of samples drawn from the poisson distribution. Its
    probability mass function is defined as

    .. math::
        f(x) = \\frac{\\lambda^xe^{-\\lambda}}{k!},

    Args:
        lam (array_like of floats): Parameter of the poisson distribution
            :math:`\\lambda`.
        size (int or tuple of ints): The shape of the array. If ``None``, this
            function generate an array whose shape is `lam.shape`.
        dtype: Data type specifier. Only :class:`numpy.int32` and
            :class:`numpy.int64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the poisson distribution.

    .. seealso:: :func:`numpy.random.poisson`
    """
    rs = generator.get_random_state()
    x = rs.poisson(lam, size, dtype)
    return x


def standard_cauchy(size=None, dtype=float):
    """Standard cauchy distribution.

    Returns an array of samples drawn from the standard cauchy distribution.
    Its probability density function is defined as

      .. math::
         f(x) = \\frac{1}{\\pi(1+x^2)},

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
    """Standard exponential distribution.

    Returns an array of samples drawn from the standard exponential
    distribution. Its probability density function is defined as

      .. math::
         f(x) = e^{-x},

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
    return rs.standard_exponential(size, dtype)


def standard_gamma(shape, size=None, dtype=float):
    """Standard gamma distribution.

    Returns an array of samples drawn from the standard gamma distribution. Its
    probability density function is defined as

    .. math::
       f(x) = \\frac{1}{\\Gamma(k)}x^{k-1}e^{-x},

    Args:
        shape (array): Parameter of the gamma distribution :math:`k`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the standard gamma distribution.

    .. seealso::
        :func:`numpy.random.standard_gamma`
    """
    rs = generator.get_random_state()
    return rs.standard_gamma(shape, size, dtype)


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
    """Standard Student's t distribution.

    Returns an array of samples drawn from the standard Student's t
    distribution. Its probability density function is defined as

    .. math::
        f(x) = \\frac{\\Gamma(\\frac{\\nu+1}{2})} \
            {\\sqrt{\\nu\\pi}\\Gamma(\\frac{\\nu}{2})} \
            \\left(1 + \\frac{x^2}{\\nu} \\right)^{-(\\frac{\\nu+1}{2})},

    Args:
        df (float or array_like of floats): Degree of freedom :math:`\\nu`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the standard Student's t distribution.

    .. seealso::
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


def vonmises(mu, kappa, size=None, dtype=float):
    """von Mises distribution.

    Returns an array of samples drawn from the von Mises distribution. Its
    probability density function is defined as

    .. math::
       f(x) = \\frac{e^{\\kappa \\cos(x-\\mu)}}{2\\pi I_0(\\kappa)},

    Args:
        mu (float): Parameter of the von Mises distribution :math:`\\mu`.
        kappa (float): Parameter of the von Mises distribution :math:`\\kappa`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the von Mises distribution.

    .. seealso::
        :func:`numpy.random.vonmises`
    """
    rs = generator.get_random_state()
    return rs.vonmises(mu, kappa, size=size, dtype=dtype)


def zipf(a, size=None, dtype=int):
    """Zipf distribution.

    Returns an array of samples drawn from the Zipf distribution. Its
    probability mass function is defined as

    .. math::
        f(x) = \\frac{x^{-a}}{ \\zeta (a)},

    where :math:`\\zeta` is the Riemann Zeta function.

    Args:
        a (float): Parameter of the beta distribution :math:`a`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.int32` and
            :class:`numpy.int64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the Zipf distribution.

    .. seealso::
        :func:`numpy.random.zipf`
    """
    rs = generator.get_random_state()
    return rs.zipf(a, size=size, dtype=dtype)

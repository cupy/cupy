from cupy.random import _generator
from cupy import _util


# TODO(beam2d): Implement many distributions


def beta(a, b, size=None, dtype=float):
    """Beta distribution.

    Returns an array of samples drawn from the beta distribution. Its
    probability density function is defined as

    .. math::
       f(x) = \\frac{x^{\\alpha-1}(1-x)^{\\beta-1}}{B(\\alpha,\\beta)}.

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
    rs = _generator.get_random_state()
    return rs.beta(a, b, size, dtype)


def binomial(n, p, size=None, dtype=int):
    """Binomial distribution.

    Returns an array of samples drawn from the binomial distribution. Its
    probability mass function is defined as

    .. math::
        f(x) = \\binom{n}{x}p^x(1-p)^{n-x}.

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
    rs = _generator.get_random_state()
    return rs.binomial(n, p, size, dtype)


def chisquare(df, size=None, dtype=float):
    """Chi-square distribution.

    Returns an array of samples drawn from the chi-square distribution. Its
    probability density function is defined as

    .. math::
       f(x) = \\frac{(1/2)^{k/2}}{\\Gamma(k/2)}x^{k/2-1}e^{-x/2}.

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
    rs = _generator.get_random_state()
    return rs.chisquare(df, size, dtype)


def dirichlet(alpha, size=None, dtype=float):
    """Dirichlet distribution.

    Returns an array of samples drawn from the dirichlet distribution. Its
    probability density function is defined as

    .. math::
        f(x) = \\frac{\\Gamma(\\sum_{i=1}^K\\alpha_i)} \
            {\\prod_{i=1}^{K}\\Gamma(\\alpha_i)} \
            \\prod_{i=1}^Kx_i^{\\alpha_i-1}.

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
    rs = _generator.get_random_state()
    return rs.dirichlet(alpha, size, dtype)


def exponential(scale, size=None, dtype=float):
    """Exponential distribution.

    Returns an array of samples drawn from the exponential distribution. Its
    probability density function is defined as

    .. math::
       f(x) = \\frac{1}{\\beta}\\exp (-\\frac{x}{\\beta}).

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
    rs = _generator.get_random_state()
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
            ^{-\\frac{d_1+d_2}{2}}.

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
    rs = _generator.get_random_state()
    return rs.f(dfnum, dfden, size, dtype)


def gamma(shape, scale=1.0, size=None, dtype=float):
    """Gamma distribution.

    Returns an array of samples drawn from the gamma distribution. Its
    probability density function is defined as

    .. math::
       f(x) = \\frac{1}{\\Gamma(k)\\theta^k}x^{k-1}e^{-x/\\theta}.

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
    rs = _generator.get_random_state()
    return rs.gamma(shape, scale, size, dtype)


def geometric(p, size=None, dtype=int):
    """Geometric distribution.

    Returns an array of samples drawn from the geometric distribution. Its
    probability mass function is defined as

    .. math::
        f(x) = p(1-p)^{k-1}.

    Args:
        p (float): Success probability of the geometric distribution.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.int32` and
            :class:`numpy.int64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the geometric distribution.

    .. seealso::
        :func:`numpy.random.geometric`
    """
    rs = _generator.get_random_state()
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
    rs = _generator.get_random_state()
    return rs.gumbel(loc, scale, size, dtype)


def hypergeometric(ngood, nbad, nsample, size=None, dtype=int):
    """hypergeometric distribution.

    Returns an array of samples drawn from the hypergeometric distribution. Its
    probability mass function is defined as

    .. math::
        f(x) = \\frac{\\binom{m}{n}\\binom{N-m}{n-x}}{\\binom{N}{n}}.

    Args:
        ngood (int or array_like of ints): Parameter of the hypergeometric
            distribution :math:`n`.
        nbad (int or array_like of ints): Parameter of the hypergeometric
            distribution :math:`m`.
        nsample (int or array_like of ints): Parameter of the hypergeometric
            distribution :math:`N`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.int32` and
            :class:`numpy.int64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the hypergeometric distribution.

    .. seealso::
        :func:`numpy.random.hypergeometric`
    """
    rs = _generator.get_random_state()
    return rs.hypergeometric(ngood, nbad, nsample, size, dtype)


def logistic(loc=0.0, scale=1.0, size=None, dtype=float):
    """Logistic distribution.

    Returns an array of samples drawn from the logistic distribution. Its
    probability density function is defined as

    .. math::
       f(x) = \\frac{e^{-(x-\\mu)/s}}{s(1+e^{-(x-\\mu)/s})^2}.

    Args:
        loc (float): The location of the mode :math:`\\mu`.
        scale (float): The scale parameter :math:`s`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the logistic distribution.

    .. seealso::
        :func:`numpy.random.logistic`
    """
    rs = _generator.get_random_state()
    return rs.logistic(loc, scale, size, dtype)


def laplace(loc=0.0, scale=1.0, size=None, dtype=float):
    """Laplace distribution.

    Returns an array of samples drawn from the laplace distribution. Its
    probability density function is defined as

    .. math::
       f(x) = \\frac{1}{2b}\\exp\\left(-\\frac{|x-\\mu|}{b}\\right).

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
    rs = _generator.get_random_state()
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
    rs = _generator.get_random_state()
    return rs.lognormal(mean, sigma, size=size, dtype=dtype)


def logseries(p, size=None, dtype=int):
    """Log series distribution.

    Returns an array of samples drawn from the log series distribution. Its
    probability mass function is defined as

    .. math::
       f(x) = \\frac{-p^x}{x\\ln(1-p)}.

    Args:
        p (float): Parameter of the log series distribution :math:`p`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.int32` and
            :class:`numpy.int64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the log series distribution.

    .. seealso:: :func:`numpy.random.logseries`

    """
    rs = _generator.get_random_state()
    return rs.logseries(p, size=size, dtype=dtype)


def negative_binomial(n, p, size=None, dtype=int):
    """Negative binomial distribution.

    Returns an array of samples drawn from the negative binomial distribution.
    Its probability mass function is defined as

    .. math::
        f(x) = \\binom{x + n - 1}{n - 1}p^n(1-p)^{x}.

    Args:
        n (int): Parameter of the negative binomial distribution :math:`n`.
        p (float): Parameter of the negative binomial distribution :math:`p`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.int32` and
            :class:`numpy.int64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the negative binomial distribution.

    .. seealso::
        :func:`numpy.random.negative_binomial`
    """
    rs = _generator.get_random_state()
    return rs.negative_binomial(n, p, size=size, dtype=dtype)


def multivariate_normal(mean, cov, size=None, check_valid='ignore',
                        tol=1e-08, method='cholesky', dtype=float):
    """Multivariate normal distribution.

    Returns an array of samples drawn from the multivariate normal
    distribution. Its probability density function is defined as

    .. math::
       f(x) = \\frac{1}{(2\\pi|\\Sigma|)^(n/2)} \
           \\exp\\left(-\\frac{1}{2} \
           (x-\\mu)^{\\top}\\Sigma^{-1}(x-\\mu)\\right).

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
        method : { 'cholesky', 'eigh', 'svd'}, optional
            The cov input is used to compute a factor matrix A such that
            ``A @ A.T = cov``. This argument is used to select the method
            used to compute the factor matrix A. The default method 'cholesky'
            is the fastest, while 'svd' is the slowest but more robust than
            the fastest method. The method `eigh` uses eigen decomposition to
            compute A and is faster than svd but slower than cholesky.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the multivariate normal distribution.

    .. note:: Default `method` is set to fastest, 'cholesky', unlike numpy
        which defaults to 'svd'. Cholesky decomposition in CuPy will fail
        silently if the input covariance matrix is not positive definite and
        give invalid results, unlike in numpy, where an invalid covariance
        matrix will raise an exception. Setting `check_valid` to 'raise' will
        replicate numpy behavior by checking the input, but will also force
        device synchronization. If validity of input is unknown, setting
        `method` to 'einh' or 'svd' and `check_valid` to 'warn' will use
        cholesky decomposition for positive definite matrices, and fallback to
        the specified `method` for other matrices (i.e., not positive
        semi-definite), and will warn if decomposition is suspect.

    .. seealso:: :func:`numpy.random.multivariate_normal`

    """
    _util.experimental('cupy.random.multivariate_normal')
    rs = _generator.get_random_state()
    return rs.multivariate_normal(
        mean, cov, size, check_valid, tol, method, dtype)


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
    rs = _generator.get_random_state()
    return rs.normal(loc, scale, size, dtype)


def pareto(a, size=None, dtype=float):
    """Pareto II or Lomax distribution.

    Returns an array of samples drawn from the Pareto II distribution. Its
    probability density function is defined as

    .. math::
        f(x) = \\alpha(1+x)^{-(\\alpha+1)}.

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
    rs = _generator.get_random_state()
    return rs.pareto(a, size, dtype)


def noncentral_chisquare(df, nonc, size=None, dtype=float):
    """Noncentral chisquare distribution.

    Returns an array of samples drawn from the noncentral chisquare
    distribution. Its probability density function is defined as

    .. math::
       f(x) = \\frac{1}{2}e^{-(x+\\lambda)/2} \\
        \\left(\\frac{x}{\\lambda}\\right)^{k/4 - 1/2} \\
        I_{k/2 - 1}(\\sqrt{\\lambda x}),

    where :math:`I` is the modified Bessel function of the first kind.

    Args:
        df (float): Parameter of the noncentral chisquare distribution
            :math:`k`.
        nonc (float): Parameter of the noncentral chisquare distribution
            :math:`\\lambda`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the noncentral chisquare distribution.

    .. seealso::
        :func:`numpy.random.noncentral_chisquare`
    """
    rs = _generator.get_random_state()
    return rs.noncentral_chisquare(df, nonc, size=size, dtype=dtype)


def noncentral_f(dfnum, dfden, nonc, size=None, dtype=float):
    """Noncentral F distribution.

    Returns an array of samples drawn from the noncentral F
    distribution.

    Reference: https://en.wikipedia.org/wiki/Noncentral_F-distribution

    Args:
        dfnum (float): Parameter of the noncentral F distribution.
        dfden (float): Parameter of the noncentral F distribution.
        nonc (float): Parameter of the noncentral F distribution.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the noncentral F distribution.

    .. seealso::
        :func:`numpy.random.noncentral_f`
    """
    rs = _generator.get_random_state()
    return rs.noncentral_f(dfnum, dfden, nonc, size=size, dtype=dtype)


def poisson(lam=1.0, size=None, dtype=int):
    """Poisson distribution.

    Returns an array of samples drawn from the poisson distribution. Its
    probability mass function is defined as

    .. math::
        f(x) = \\frac{\\lambda^xe^{-\\lambda}}{k!}.

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
    rs = _generator.get_random_state()
    return rs.poisson(lam, size, dtype)


def power(a, size=None, dtype=float):
    """Power distribution.

    Returns an array of samples drawn from the power distribution. Its
    probability density function is defined as

    .. math::
       f(x) = ax^{a-1}.

    Args:
        a (float): Parameter of the power distribution :math:`a`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the power distribution.

    .. seealso::
        :func:`numpy.random.power`
    """
    rs = _generator.get_random_state()
    return rs.power(a, size, dtype)


def rayleigh(scale=1.0, size=None, dtype=float):
    """Rayleigh distribution.

    Returns an array of samples drawn from the rayleigh distribution.
    Its probability density function is defined as

      .. math::
         f(x) = \\frac{x}{\\sigma^2}e^{\\frac{-x^2}{2-\\sigma^2}}, x \\ge 0.

    Args:
        scale (array): Parameter of the rayleigh distribution :math:`\\sigma`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the rayleigh distribution.

    .. seealso:: :func:`numpy.random.rayleigh`
    """
    rs = _generator.get_random_state()
    return rs.rayleigh(scale, size, dtype)


def standard_cauchy(size=None, dtype=float):
    """Standard cauchy distribution.

    Returns an array of samples drawn from the standard cauchy distribution.
    Its probability density function is defined as

      .. math::
         f(x) = \\frac{1}{\\pi(1+x^2)}.

    Args:
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the standard cauchy distribution.

    .. seealso:: :func:`numpy.random.standard_cauchy`
    """
    rs = _generator.get_random_state()
    return rs.standard_cauchy(size, dtype)


def standard_exponential(size=None, dtype=float):
    """Standard exponential distribution.

    Returns an array of samples drawn from the standard exponential
    distribution. Its probability density function is defined as

      .. math::
         f(x) = e^{-x}.

    Args:
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the standard exponential distribution.

    .. seealso:: :func:`numpy.random.standard_exponential`
    """
    rs = _generator.get_random_state()
    return rs.standard_exponential(size, dtype)


def standard_gamma(shape, size=None, dtype=float):
    """Standard gamma distribution.

    Returns an array of samples drawn from the standard gamma distribution. Its
    probability density function is defined as

    .. math::
       f(x) = \\frac{1}{\\Gamma(k)}x^{k-1}e^{-x}.

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
    rs = _generator.get_random_state()
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
    rs = _generator.get_random_state()
    return rs.standard_normal(size, dtype)


def standard_t(df, size=None, dtype=float):
    """Standard Student's t distribution.

    Returns an array of samples drawn from the standard Student's t
    distribution. Its probability density function is defined as

    .. math::
        f(x) = \\frac{\\Gamma(\\frac{\\nu+1}{2})} \
            {\\sqrt{\\nu\\pi}\\Gamma(\\frac{\\nu}{2})} \
            \\left(1 + \\frac{x^2}{\\nu} \\right)^{-(\\frac{\\nu+1}{2})}.

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
    rs = _generator.get_random_state()
    return rs.standard_t(df, size, dtype)


def triangular(left, mode, right, size=None, dtype=float):
    """Triangular distribution.

    Returns an array of samples drawn from the triangular distribution. Its
    probability density function is defined as

    .. math::
       f(x) = \\begin{cases}
            \\frac{2(x-l)}{(r-l)(m-l)} & \\text{for } l \\leq x \\leq m, \\\\
            \\frac{2(r-x)}{(r-l)(r-m)} & \\text{for } m \\leq x \\leq r, \\\\
            0 & \\text{otherwise}.
          \\end{cases}

    Args:
        left (float): Lower limit :math:`l`.
        mode (float): The value where the peak of the distribution occurs.
            :math:`m`.
        right (float): Higher Limit :math:`r`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the triangular distribution.

    .. seealso::
        :func:`numpy.random.triangular`
    """
    rs = _generator.get_random_state()
    return rs.triangular(left, mode, right, size, dtype)


def uniform(low=0.0, high=1.0, size=None, dtype=float):
    """Returns an array of uniformly-distributed samples over an interval.

    Samples are drawn from a uniform distribution over the half-open interval
    ``[low, high)``. The samples may contain the ``high`` limit due to
    floating-point rounding.

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
    rs = _generator.get_random_state()
    return rs.uniform(low, high, size=size, dtype=dtype)


def vonmises(mu, kappa, size=None, dtype=float):
    """von Mises distribution.

    Returns an array of samples drawn from the von Mises distribution. Its
    probability density function is defined as

    .. math::
       f(x) = \\frac{e^{\\kappa \\cos(x-\\mu)}}{2\\pi I_0(\\kappa)}.

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
    rs = _generator.get_random_state()
    return rs.vonmises(mu, kappa, size=size, dtype=dtype)


def wald(mean, scale, size=None, dtype=float):
    """Wald distribution.

    Returns an array of samples drawn from the Wald distribution. Its
    probability density function is defined as

    .. math::
       f(x) = \\sqrt{\\frac{\\lambda}{2\\pi x^3}}\\
           e^{\\frac{-\\lambda(x-\\mu)^2}{2\\mu^2x}}.

    Args:
        mean (float): Parameter of the wald distribution :math:`\\mu`.
        scale (float): Parameter of the wald distribution :math:`\\lambda`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the wald distribution.

    .. seealso::
        :func:`numpy.random.wald`
    """
    rs = _generator.get_random_state()
    return rs.wald(mean, scale, size, dtype)


def weibull(a, size=None, dtype=float):
    """weibull distribution.

    Returns an array of samples drawn from the weibull distribution. Its
    probability density function is defined as

    .. math::
       f(x) = ax^{(a-1)}e^{-x^a}.

    Args:
        a (float): Parameter of the weibull distribution :math:`a`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the weibull distribution.

    .. seealso::
        :func:`numpy.random.weibull`
    """
    rs = _generator.get_random_state()
    return rs.weibull(a, size=size, dtype=dtype)


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
    rs = _generator.get_random_state()
    return rs.zipf(a, size=size, dtype=dtype)

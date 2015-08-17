import math

import chainer.functions as F
from chainer import variable


def gaussian_kl_divergence(mean, ln_var):
    """Calculate KL-divergence between given gaussian and the standard one.

    Given two variable ``mean`` representing :math:`\\mu` and ``ln_var``
    representing :math:`\\log(\\sigma^2)`, this function returns a variable
    representing KL-divergence between given multi-dimensional gaussian
    :math:`N(\\mu, S)` and the standard Gaussian :math:`N(0, I)`

    .. math::

       D_{\\mathbf{KL}}(N(\\mu, S) \\| N(0, I)),

    where :math:`S` is a diagonal matrix such that :math:`S_{ii} = \\sigma_i^2`
    and :math:`I` is an identity matrix.

    Args:
        mean (~chainer.Variable): A variable representing mean of given
            gaussian distribution, :math:`\\mu`.
        ln_var (~chainer.Variable): A variable representing logarithm of
            variance of given gaussian distribution, :math:`\\log(\\sigma^2)`.

    Returns:
        ~chainer.Variable: A variable representing KL-divergence between
            given gaussian distribution and the standard gaussian.

    """
    assert isinstance(mean, variable.Variable)
    assert isinstance(ln_var, variable.Variable)

    J = mean.data.size
    var = F.exp(ln_var)
    return (F.sum(mean * mean) + F.sum(var) - F.sum(ln_var) - J) * 0.5


def bernoulli_nll(x, y):
    """Calculate negative log-likelihood of Bernoulli distribution.

    This function calculates negative log-likelihood on a Bernoulli
    distribution.

    .. math::

        -B(x; p) = -\\sum_i {x_i \\log(p_i) + (1 - x_i)\\log(1 - p_i)},

    where :math:`p = \\sigma(y)`, and :math:`\\sigma(\\cdot)` is a sigmoid
    funciton.

    .. note::

       As this funtion uses a sigmoid function, you can pass a result of
       fully-connected layer (that means :class:`Linear`) to this function
       directly.

    Args:
        x (~chainer.Variable): Input variable.
        y (~chainer.Variable): A variable representing the parameter of
            Bernoulli distribution.

    Returns:
        ~chainer.Variable: A variable representing negative log-likelihood.

    """
    assert isinstance(x, variable.Variable)
    assert isinstance(y, variable.Variable)

    return F.sum(F.softplus(-y)) + F.sum(y) - F.sum(y * x)


def gaussian_nll(x, mean, ln_var):
    """Calculate negative log-likelihood of Gaussian distribution.

    Given two variable ``mean`` representing :math:`\\mu` and ``ln_var``
    representing :math:`\\log(\\sigma^2)`, this function returns negative
    log-likelihood of :math:`x` on a Gaussian distribution :math:`N(\\mu, S)`,

    .. math::

        -\\log N(x; \\mu, \\sigma^2) =
        \\log\\left(\\sqrt{(2\\pi)^D |S|}\\right) +
        \\frac{1}{2}(x - \\mu)^\\top S^{-1}(x - \\mu),

    where :math:`D` is a dimention of :math:`x` and :math:`S` is a diagonal
    matrix where :math:`S_{ii} = \\sigma_i^2`.

    Args:
        x (~chainer.Variable): Input variable.
        mean (~chainer.Variable): A variable representing mean of a gaussian
            distribution, :math:`\\mu`.
        ln_var (~chainer.Variable): A variable representing logarithm of
            variance of a gaussian distribution, :math:`\\log(\\sigma^2)`.

    Returns:
        ~chainer.Variable: A variable representing negative log-likelihood.

    """
    assert isinstance(x, variable.Variable)
    assert isinstance(mean, variable.Variable)
    assert isinstance(ln_var, variable.Variable)

    D = x.data.size
    x_prec = F.exp(-ln_var)
    x_diff = x - mean
    x_power = (x_diff * x_diff) * x_prec * -0.5
    return (F.sum(ln_var) + D * math.log(2 * math.pi)) / 2 - F.sum(x_power)

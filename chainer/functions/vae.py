import math

import chainer.functions as F
from chainer import variable


def gaussian_kl_divergence(z_mean, z_ln_var):
    """Calculate KL-divergence between a given gaussian and the normal one.

    Given two variable ``z_mean`` representing :math:`\\mu` and ``z_ln_var``
    representing :math:`\\log(\\sigma^2)`, this function returns a variable
    representing KL-divergence between :math:`N(\\mu, \\sigma^2)` and
    the normal Gaussian :math:`N(0, 1)`.

    Args:
        z_mean (~chainer.Variable): A variable representing mean of the given
            gaussian distribution.
        z_ln_var (~chainer.Variable): A variable representing logarithm of
            variance of the given gaussian distribution.

    Returns:
        ~chainer.Variable: A variable representing KL-divergence between the
            given gaussian distribution and the normal gaussian.

    """
    assert isinstance(z_mean, variable.Variable)
    assert isinstance(z_ln_var, variable.Variable)

    J = z_mean.data.size
    z_var = F.exp(z_ln_var)
    return (F.sum(z_mean * z_mean) + F.sum(z_var) - F.sum(z_ln_var) - J) * 0.5


def bernoulli_nll(x, y):
    """Calculate negative log-likelihood of Bernoulli distribution.

    This function calculates negative log-likelihood on a Bernoulli
    distribution.

    .. math::

        -B(x; p) = -\\sum_i {x_i \\log(p_i) + (1 - x_i)\\log(1 - p_i)},

    where :math:`p = \\sigma(y)`, and :math:`\\sigma(\\cdot)` is a sigmoid
    funciton.

    When :math:`x \\in \\{0, 1\\}`, the return value is equal to negative
    log-likelihoood on a Bernoulli distribution.

    .. note::

       As this funtion uses a sigmoid function, you can pass a result of
       full-connect layer (that means :class:`Linear`) to this function
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

    It returns negative log-likelihood of :math:`x` on a Gaussian distribution
    :math:`N(\\mu, \\sigma^2)`, where :math:`\\mu` is given as ``mean`` and
    :math:`\\log(\\sigma^2)` is given as ``ln_var``.

    .. math::

        -\\log N(x; \\mu, \\sigma^2) =
        \\log((\\sqrt{2\\pi})^D \\sqrt{|S|}) +
        \\frac{1}{2}(x - \\mu)S^{-1}(x - \\mu)

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

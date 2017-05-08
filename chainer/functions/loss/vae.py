import math

from chainer.functions.activation import softplus
from chainer.functions.math import exponential
from chainer.functions.math import sum


def gaussian_kl_divergence(mean, ln_var, reduce='sum'):
    """Computes the KL-divergence of Gaussian variables from the standard one.

    Given two variable ``mean`` representing :math:`\\mu` and ``ln_var``
    representing :math:`\\log(\\sigma^2)`, this function calculates
    the KL-divergence in elementwise manner between the given multi-dimensional
    Gaussian :math:`N(\\mu, S)` and the standard Gaussian :math:`N(0, I)`

    .. math::

       D_{\\mathbf{KL}}(N(\\mu, S) \\| N(0, I)),

    where :math:`S` is a diagonal matrix such that :math:`S_{ii} = \\sigma_i^2`
    and :math:`I` is an identity matrix.

    The output is a variable whose value depends on the value of
    the option ``reduce``. If it is ``'no'``, it holds the elementwise
    loss values. If it is ``'sum'``, loss values are summed up.

    Args:
        mean (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): A variable representing mean of given
            gaussian distribution, :math:`\\mu`.
        ln_var (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): A variable representing logarithm of
            variance of given gaussian distribution, :math:`\\log(\\sigma^2)`.
        reduce (str): Reduction option. Its value must be either
            ``'sum'`` or ``'no'``. Otherwise, :class:`ValueError` is raised.

    Returns:
        ~chainer.Variable:
            A variable representing KL-divergence between
            given gaussian distribution and the standard gaussian.
            If ``reduce`` is ``'no'``, the output variable holds array
            whose shape is same as one of (hence both of) input variables.
            If it is ``'sum'``, the output variable holds a scalar value.

    """
    if reduce not in ('sum', 'no'):
        raise ValueError(
            "only 'sum' and 'no' are valid for 'reduce', but '%s' is "
            'given' % reduce)

    var = exponential.exp(ln_var)
    mean_square = mean * mean
    loss = (mean_square + var - ln_var - 1) * 0.5
    if reduce == 'sum':
        return sum.sum(loss)
    else:
        return loss


def bernoulli_nll(x, y, reduce='sum'):
    """Computes the negative log-likelihood of a Bernoulli distribution.

    This function calculates the negative log-likelihood of a Bernoulli
    distribution.

    .. math::

        -\\log B(x; p) = -\\sum_i \{x_i \\log(p_i) + (1 - x_i)\\log(1 - p_i)\},

    where :math:`p = \\sigma(y)`, :math:`\\sigma(\\cdot)` is a sigmoid
    function, and :math:`B(x; p)` is a Bernoulli distribution.


    The output is a variable whose value depends on the value of
    the option ``reduce``. If it is ``'no'``, it holds the elementwise
    loss values. If it is ``'sum'``, loss values are summed up.

    .. note::

       As this function uses a sigmoid function, you can pass a result of
       fully-connected layer (that means :class:`Linear`) to this function
       directly.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable.
        y (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): A variable representing the parameter of
            Bernoulli distribution.
        reduce (str): Reduction option. Its value must be either
            ``'sum'`` or ``'no'``. Otherwise, :class:`ValueError` is raised.

    Returns:
        ~chainer.Variable:
            A variable representing the negative log-likelihood.
            If ``reduce`` is ``'no'``, the output variable holds array
            whose shape is same as one of (hence both of) input variables.
            If it is ``'sum'``, the output variable holds a scalar value.

    """
    if reduce not in ('sum', 'no'):
        raise ValueError(
            "only 'sum' and 'no' are valid for 'reduce', but '%s' is "
            'given' % reduce)

    loss = softplus.softplus(y) - x * y
    if reduce == 'sum':
        return sum.sum(loss)
    else:
        return loss


def gaussian_nll(x, mean, ln_var, reduce='sum'):
    """Computes the negative log-likelihood of a Gaussian distribution.

    Given two variable ``mean`` representing :math:`\\mu` and ``ln_var``
    representing :math:`\\log(\\sigma^2)`, this function computes in
    elementwise manner the negative log-likelihood of :math:`x` on a
    Gaussianx distribution :math:`N(\\mu, S)`,

    .. math::

        -\\log N(x; \\mu, \\sigma^2) =
        \\log\\left(\\sqrt{(2\\pi)^D |S|}\\right) +
        \\frac{1}{2}(x - \\mu)^\\top S^{-1}(x - \\mu),

    where :math:`D` is a dimension of :math:`x` and :math:`S` is a diagonal
    matrix where :math:`S_{ii} = \\sigma_i^2`.

    The output is a varialbe whose value depends on the value of
    the option ``reduce``. If it is ``'no'``, it holds the elementwise
    loss values. If it is ``'sum'``, loss values are summed up.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable.
        mean (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): A variable representing mean of a Gaussian
            distribution, :math:`\\mu`.
        ln_var (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): A variable representing logarithm of
            variance of a Gaussian distribution, :math:`\\log(\\sigma^2)`.
        reduce (str): Reduction option. Its value must be either
            ``'sum'`` or ``'no'``. Otherwise, :class:`ValueError` is raised.

    Returns:
        ~chainer.Variable:
            A variable representing the negative log-likelihood.
            If ``reduce`` is ``'no'``, the output varialbe holds array
            whose shape is same as one of (hence both of) input variables.
            If it is ``'sum'``, the output variable holds a scalar value.

    """
    if reduce not in ('sum', 'no'):
        raise ValueError(
            "only 'sum' and 'no' are valid for 'reduce', but '%s' is "
            'given' % reduce)

    x_prec = exponential.exp(-ln_var)
    x_diff = x - mean
    x_power = (x_diff * x_diff) * x_prec * -0.5
    loss = (ln_var + math.log(2 * math.pi)) / 2 - x_power
    if reduce == 'sum':
        return sum.sum(loss)
    else:
        return loss

import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class BatchNormalizationFunction(function.Function):

    def __init__(self, eps=1e-5):
        self.eps = eps

    def check_type_forward(self, in_types):
        n_in = in_types.size().eval()
        if n_in != 3 and n_in != 5:
            raise type_check.InvalidType(
                '%s or %s' % (in_types.size() == 3, in_types.size() == 5),
                '%s == %s' % (in_types.size(), n_in))

        x_type, gamma_type, beta_type = in_types[:3]
        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim >= gamma_type.ndim + 1,
            # TODO(beam2d): Check shape
            gamma_type.dtype == numpy.float32,
            beta_type.dtype == numpy.float32,
            gamma_type.shape == beta_type.shape,
        )

        if len(in_types) == 5:
            mean_type, var_type = in_types[3:]
            type_check.expect(
                mean_type.dtype == numpy.float32,
                mean_type.shape == gamma_type.shape,
                var_type.dtype == numpy.float32,
                var_type.shape == gamma_type.shape,
            )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, gamma, beta = inputs[:3]

        head_ndim = gamma.ndim + 1
        expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)
        gamma = gamma[expander]
        beta = beta[expander]

        if len(inputs) == 5:
            mean = inputs[3]
            var = inputs[4]
        else:
            axis = (0,) + tuple(range(head_ndim, x.ndim))
            mean = x.mean(axis=axis)
            var = x.var(axis=axis)
            var += self.eps
            self.mean = mean
            self.var = var

        self.std = xp.sqrt(var, dtype=var.dtype)

        if xp is numpy:
            x_mu = x - mean[expander]
            self.x_hat = x_mu / self.std[expander]
            y = gamma * self.x_hat
            y += beta
        else:
            self.x_hat, y = cuda.elementwise(
                'T x, T mean, T std, T gamma, T beta', 'T x_hat, T y',
                '''
                   x_hat = (x - mean) / std;
                   y = gamma * x_hat + beta;
                ''',
                'bn_fwd')(x, mean[expander], self.std[expander], gamma, beta)
        return y,

    def backward(self, inputs, grad_outputs):
        x, gamma = inputs[:2]
        gy = grad_outputs[0]

        head_ndim = gamma.ndim + 1
        expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)
        m = gamma.dtype.type(x.size // gamma.size)

        axis = (0,) + tuple(range(head_ndim, x.ndim))
        gbeta = gy.sum(axis=axis)
        ggamma = (gy * self.x_hat).sum(axis=axis)

        xp = cuda.get_array_module(x)
        if len(inputs) == 5:
            var = inputs[4]
            gs = gamma / self.std
            gmean = -gs * gbeta
            gvar = -0.5 * gamma / var * ggamma
            gx = gs[expander] * gy
            return gx, ggamma, gbeta, gmean, gvar

        if xp is numpy:
            gx = (gamma / self.std)[expander] * (
                gy - (self.x_hat * ggamma[expander] + gbeta[expander]) / m)
        else:
            inv_m = numpy.float32(1) / m
            gx = cuda.elementwise(
                'T gy, T x_hat, T gamma, T std, T ggamma, T gbeta, T inv_m',
                'T gx',
                'gx = (gamma / std) * (gy - (x_hat * ggamma + gbeta) * inv_m)',
                'bn_bwd')(gy, self.x_hat, gamma[expander], self.std[expander],
                          ggamma[expander], gbeta[expander], inv_m)
        return gx, ggamma, gbeta


def batch_normalization(x, gamma, beta, eps=1e-5):
    """Batch normalization function.

    It takes the input variable ``x`` and two parameter variables ``gamma`` and
    ``beta``. The input must have the batch size and the features (or channels)
    as the first two dimensions of its shape. The input can have more than two
    dimensions, where the remained dimensions are considered as spatial
    dimensions, which are considered as a part of the batch size.

    Args:
        x (Variable): The input variable.
        gamma (Variable): The scaling parameter of normalized data.
        beta (Variable): The shifting parameter of scaled normalized data.
        eps (float): Epsilon value for numerical stability.

    See: `Batch Normalization: Accelerating Deep Network Training by Reducing\
          Internal Covariate Shift <http://arxiv.org/abs/1502.03167>`_

    .. seealso:: :class:`links.BatchNormalization`

    """
    return BatchNormalizationFunction(eps)(x, gamma, beta)


def fixed_batch_normalization(x, gamma, beta, mean, var, eps=1e-5):
    """Batch normalization function with fixed statistics.

    This is a variant of batch normalization, where the mean and variance
    statistics are given by the caller as variables. This is used on testing
    mode of the batch normalization layer, where batch statistics cannot be
    used for prediction consistency.

    Args:
        x (Variable): The input variable.
        gamma (Variable): The scaling parameter of normalized data.
        beta (Variable): The shifting parameter of scaled normalized data.
        mean (Variable): The shifting parameter of input.
        var (Variable): The square of scaling parameter of input.
        eps (float): Epsilon value for numerical stability.

    .. seealso::
       :func:`functions.batch_normalization`,
       :class:`links.BatchNormalization`

    """
    return BatchNormalizationFunction(eps)(x, gamma, beta, mean, var)

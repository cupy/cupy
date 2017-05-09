import numpy

from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check


class CrossCovariance(function.Function):

    """Cross-covariance loss."""

    def __init__(self, reduce='half_squared_sum'):
        self.y_centered = None
        self.z_centered = None
        self.covariance = None

        if reduce not in ('half_squared_sum', 'no'):
            raise ValueError(
                "only 'half_squared_sum' and 'no' are valid "
                "for 'reduce', but '%s' is given" % reduce)
        self.reduce = reduce

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        z_type, y_type = in_types

        type_check.expect(
            z_type.dtype == numpy.float32,
            z_type.ndim == 2,
            y_type.dtype == numpy.float32,
            y_type.ndim == 2,

            z_type.shape[0] == y_type.shape[0]
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        y, z = inputs

        self.y_centered = y - y.mean(axis=0, keepdims=True)
        self.z_centered = z - z.mean(axis=0, keepdims=True)
        self.covariance = self.y_centered.T.dot(self.z_centered)
        self.covariance /= len(y)

        if self.reduce == 'half_squared_sum':
            cost = xp.vdot(self.covariance, self.covariance)
            cost *= y.dtype.type(0.5)
            return utils.force_array(cost),
        else:
            return self.covariance,

    def backward(self, inputs, grad_outputs):
        y, z = inputs
        gcost, = grad_outputs
        gcost_div_n = gcost / gcost.dtype.type(len(y))

        if self.reduce == 'half_squared_sum':
            gy = self.z_centered.dot(self.covariance.T)
            gz = self.y_centered.dot(self.covariance)
            gy *= gcost_div_n
            gz *= gcost_div_n
        else:
            gy = self.z_centered.dot(gcost_div_n.T)
            gz = self.y_centered.dot(gcost_div_n)
        return gy, gz


def cross_covariance(y, z, reduce='half_squared_sum'):
    """Computes the sum-squared cross-covariance penalty between ``y`` and ``z``

    The output is a variable whose value depends on the value of
    the option ``reduce``. If it is ``'no'``, it holds the covariant
    matrix that has as many rows (resp. columns) as the dimension of
    ``y`` (resp.z).
    If it is ``'half_squared_sum'``, it holds the half of the
    Frobenius norm (i.e. L2 norm of a matrix flattened to a vector)
    of the covarianct matrix.

    Args:
        y (Variable): Variable holding a matrix where the first dimension
            corresponds to the batches.
        z (Variable): Variable holding a matrix where the first dimension
            corresponds to the batches.
        reduce (str): Reduction option. Its value must be either
            ``'half_squared_sum'`` or ``'no'``.
            Otherwise, :class:`ValueError` is raised.

    Returns:
        Variable:
            A variable holding the cross covariance loss.
            If ``reduce`` is ``'no'``, the output variable holds
            2-dimensional array matrix of shape ``(M, N)`` where
            ``M`` (resp. ``N``) is the number of columns of ``y``
            (resp. ``z``).
            If it is ``'half_squared_sum'``, the output variable
            holds a scalar value.

    .. note::

       This cost can be used to disentangle variables.
       See https://arxiv.org/abs/1412.6583v3 for details.

    """
    return CrossCovariance(reduce)(y, z)

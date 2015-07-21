import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _fwd_kern():
    return cuda.elementwise(
        '''float* y, const float* x, const float* cond, const float* W,
           int cdim, int rdim''',
        'y[i] = cond[i] >= 0 ? x[i] : x[i] * W[i / rdim % cdim]', 'prelu')


class PReLU(function.Function):

    """Parametric ReLU function.

    PReLU function is written in elementwise equation as
    :math:`PReLU(x) = \max(x, ax)`, where :math:`a` is a parameter array.

    When the PReLU function is combined with two-dimensional convolution, the
    elements of parameter :math:`a` are typically shared across the same filter
    of different pixels. In order to support such usage, this function supports
    the shape of parameter array that indicates leading dimensions of input
    arrays except the batch dimension.

    Args:
        shape (tuple of ints): Shape of the parameter array.
        init (float): Initial parameter value.

    See detail in paper: `Delving Deep into Rectifiers: Surpassing \
    Human-Level Performance on ImageNet Classification \
    <http://arxiv.org/abs/1502.01852>`_.

    """
    parameter_names = 'W',
    gradient_names = 'gW',

    def __init__(self, shape=(), init=0.25):
        self.W = numpy.full(shape, init, dtype=numpy.float32)
        self.gW = numpy.empty_like(self.W)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

        x_type, = in_types
        W = type_check.Variable(self.W, 'W')

        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim >= W.shape.__len__() + 1,
            x_type.shape[1: 1 + len(self.W.shape)] == W.shape
        )

    def forward_cpu(self, x):
        y = x[0].copy()
        masked = numpy.ma.masked_greater_equal(y, 0, copy=False)
        shape = self._get_extended_shape_W(y)
        masked *= self.W.reshape(shape)
        return y,

    def forward_gpu(self, x):
        cdim = self.W.size
        rdim = x[0].size // (x[0].shape[0] * cdim)
        y = cuda.empty_like(x[0])
        _fwd_kern()(y, x[0], x[0], self.W, cdim, rdim)
        return y,

    def backward_cpu(self, x, gy):
        mask = x[0] >= 0
        masked_x_gy = numpy.ma.array(x[0] * gy[0], mask=mask)
        axes = (0,) + tuple(six.moves.range(1 + len(self.W.shape), gy[0].ndim))
        self.gW += masked_x_gy.sum(axis=axes)

        gx = gy[0].copy()
        masked = numpy.ma.array(gx, mask=mask)
        shape = self._get_extended_shape_W(gx)
        masked *= self.W.reshape(shape)

        return gx,

    def backward_gpu(self, x, gy):
        ldim = x[0].shape[0]
        cdim = self.W.size
        rdim = x[0].size // (ldim * cdim)

        masked = cuda.empty_like(x[0])
        cuda.elementwise('float* masked, const float* x, const float* gy',
                         'masked[i] = x[i] >= 0 ? 0 : x[i] * gy[i]',
                         'prelu_masked')(masked, x[0], gy[0])

        with cuda.using_cumisc():
            rsum = cuda.cumisc.sum(masked.reshape(ldim * cdim, rdim), axis=1)
            gW = cuda.cumisc.sum(rsum.reshape(ldim, cdim), axis=0)
            self.gW += gW.reshape(self.gW.shape)
            del rsum, gW

        gx = masked  # reuse buffer
        _fwd_kern()(gx, gy[0], x[0], self.W, cdim, rdim)
        return gx,

    def _get_extended_shape_W(self, x):
        return (1,) + self.W.shape + (1,) * (x.ndim - self.W.ndim - 1)

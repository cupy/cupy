import numpy
from scikits.cuda import cublas
import scikits.cuda.linalg as culinalg
import scikits.cuda.misc as cumisc
from chainer import cuda, Function

def _fwd_kern():
    return cuda.elementwise(
        '''float* y, const float* x, const float* cond, const float* W,
           int cdim, int rdim''',
        'y[i] = cond[i] >= 0 ? x[i] : x[i] * W[i / rdim % cdim]', 'prelu')

class PReLU(Function):
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

    See detail in paper: `Delving Deep into Rectifiers: Surpassing Human-Level \
    Performance on ImageNet Classification <http://arxiv.org/abs/1502.01852>`_.

    """
    parameter_names = 'W',
    gradient_names  = 'gW',

    def __init__(self, shape=(), init=0.25):
        self.W  = numpy.full(shape, init, dtype=numpy.float32)
        self.gW = numpy.empty_like(self.W)

    def forward_cpu(self, x):
        self._check_shape(x[0])
        y = x[0].copy()
        masked = numpy.ma.masked_greater_equal(y, 0, copy=False)
        shape = self._get_extended_shape_W(y)
        masked *= self.W.reshape(shape)
        return y,

    def forward_gpu(self, x):
        self._check_shape(x[0])
        cdim = self.W.size
        rdim = x[0].size / (x[0].shape[0] * cdim)
        y    = cuda.empty_like(x[0])
        _fwd_kern()(y, x[0], x[0], self.W, cdim, rdim)
        return y,

    def backward_cpu(self, x, gy):
        mask = x[0] >= 0
        masked_x_gy = numpy.ma.array(x[0] * gy[0], mask=mask)
        axes = (0,) + tuple(xrange(1 + len(self.W.shape), gy[0].ndim))
        self.gW += masked_x_gy.sum(axis=axes)

        gx = gy[0].copy()
        masked = numpy.ma.array(gx, mask=mask)
        shape  = self._get_extended_shape_W(gx)
        masked *= self.W.reshape(shape)

        return gx,

    def backward_gpu(self, x, gy):
        ldim = x[0].shape[0]
        cdim = self.W.size
        rdim = x[0].size / (ldim * cdim)

        masked = cuda.empty_like(x[0])
        cuda.elementwise('float* masked, const float* x, const float* gy',
                         'masked[i] = x[i] >= 0 ? 0 : x[i] * gy[i]',
                         'prelu_masked')(masked, x[0], gy[0])

        with cuda.using_cumisc():
            rsum = cumisc.sum(masked.reshape(ldim * cdim, rdim), axis=1)
            gW   = cumisc.sum(rsum.reshape(ldim, cdim), axis=0)
            self.gW += gW.reshape(self.gW.shape)
            del rsum, gW

        gx = masked  # reuse buffer
        _fwd_kern()(gx, gy[0], x[0], self.W, cdim, rdim)
        return gx,

    def _check_shape(self, x):
        if x.shape[1 : 1 + len(self.W.shape)] != self.W.shape:
            raise ValueError(
                'Shape mismatch: input shape is {} while PReLU weight shape is {}'
                .format(x.shape, self.W.shape))

    def _get_extended_shape_W(self, x):
        return (1,) + self.W.shape + (1,) * (x.ndim - self.W.ndim - 1)

import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _cu_conv_sum(y, x, n):
    # Convolutional sum
    # TODO(beam2d): Use scan computation
    rdim = x.size // (x.shape[0] * x.shape[1])
    cuda.elementwise(
        'raw T x, int32 rdim, int32 N, int32 n_', 'raw T y',
        '''
          int half_n = n_ / 2;
          int offset = i / rdim * N * rdim + i % rdim;

          float sum_part = 0;
          for (int j = 0; j < N + half_n; ++j) {
            if (j < N) {
              sum_part += x[offset + j * rdim];
            }
            if (j >= n_) {
              sum_part -= x[offset + (j - n_) * rdim];
            }
            if (j >= half_n) {
              y[offset + (j - half_n) * rdim] = sum_part;
            }
          }
        ''', 'lrn_conv_sum')(x, rdim, x.shape[1], n, y,
                             size=x.shape[0] * rdim)


class LocalResponseNormalization(function.Function):

    """Cross-channel normalization function used in AlexNet."""

    def __init__(self, n=5, k=2, alpha=1e-4, beta=.75):
        self.n = n
        self.k = k
        self.alpha = alpha
        self.beta = beta

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim >= 2,
        )

    def forward_cpu(self, x):
        half_n = self.n // 2
        x2 = numpy.square(x[0])
        sum_part = x2.copy()
        for i in six.moves.range(1, half_n + 1):
            sum_part[:, i:] += x2[:, :-i]
            sum_part[:, :-i] += x2[:, i:]
        self.unit_scale = self.k + self.alpha * sum_part
        self.scale = self.unit_scale ** -self.beta
        self.y = x[0] * self.scale
        return self.y,

    def backward_cpu(self, x, gy):
        half_n = self.n // 2
        summand = self.y * gy[0] / self.unit_scale
        sum_part = summand.copy()
        for i in six.moves.range(1, half_n + 1):
            sum_part[:, i:] += summand[:, :-i]
            sum_part[:, :-i] += summand[:, i:]

        gx = gy[0] * self.scale - 2 * self.alpha * self.beta * x[0] * sum_part
        return gx,

    def forward_gpu(self, x):
        self.y = cuda.cupy.square(x[0])  # temporary
        self.scale = cuda.cupy.empty_like(self.y)
        _cu_conv_sum(self.scale, self.y, self.n)
        cuda.elementwise(
            'T x, T k, T alpha, T beta',
            'T y, T scale',
            '''scale = k + alpha * scale;
               y = x * pow(scale, -beta);''',
            'lrn_fwd')(x[0], self.k, self.alpha, self.beta,
                       self.y, self.scale)
        return self.y,

    def backward_gpu(self, x, gy):
        summand = cuda.elementwise(
            'T scale, T y, T gy', 'T summand',
            'summand = y * gy / scale',
            'lrn_bwd_summand')(self.scale, self.y, gy[0])
        gx = cuda.cupy.empty_like(x[0])
        _cu_conv_sum(gx, summand, self.n)
        cuda.elementwise(
            ' T x, T gy, T scale, T beta, T coeff', 'T gx',
            'gx = pow(scale, -beta) * gy - coeff * x * gx',
            'lrn_bwd')(x[0], gy[0], self.scale,
                       self.beta, 2 * self.alpha * self.beta, gx)
        return gx,


def local_response_normalization(x, n=5, k=2, alpha=1e-4, beta=.75):
    """Local response normalization across neighboring channels.

    This function implements normalization across channels. Let :math:`x` an
    input image with :math:`N` channels. Then, this function computes an output
    image :math:`y` by following formula:

    .. math::
       y_i = {x_i \\over \\left( k + \\
              \\alpha \\sum_{j=\\max{1, i - n/2}}^{\\min{N, i + n/2}} \\
              x_j^2 \\right)^\\beta}.

    Args:
        x (Variable): Input variable.
        n (int): Normalization window width.
        k (float): Smoothing parameter.
        alpha (float): Normalizer scaling parameter.
        beta (float): Normalizer power parameter.

    Returns:
        Variable: Output variable.

    See: Section 3.3 of `ImageNet Classification with Deep Convolutional \\
    Neural Networks <http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf>`_

    """
    return LocalResponseNormalization(n, k, alpha, beta)(x)

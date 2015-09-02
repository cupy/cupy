import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class BatchNormalization(function.Function):

    """Batch normalization on outputs of linear or convolution functions.

    Args:
        size (int or tuple of ints): Size (or shape) of channel
            dimensions.
        decay (float): Decay rate of moving average.
        eps (float): Epsilon value for numerical stability.
        dtype (numpy.dtype): Type to use in computing.

    See: `Batch Normalization: Accelerating Deep Network Training by Reducing\
          Internal Covariate Shift <http://arxiv.org/abs/1502.03167>`_

    """
    parameter_names = ('gamma',  'beta')
    gradient_names = ('ggamma', 'gbeta')

    def __init__(self, size, decay=0.9, eps=1e-5, dtype=numpy.float32):
        if isinstance(size, tuple):
            self.size = size
        elif isinstance(size, int):
            self.size = (size, )
        else:
            raise TypeError('size must be tuple or int')

        size = numpy.prod(size, dtype=int)
        self.dtype = numpy.dtype(dtype)

        self.avg_mean = numpy.zeros((1, size, 1), dtype=self.dtype)
        self.avg_var = numpy.zeros_like(self.avg_mean)

        self.gamma = numpy.ones_like(self.avg_mean)
        self.ggamma = numpy.full_like(self.gamma, numpy.nan)
        self.beta = numpy.zeros_like(self.avg_mean)
        self.gbeta = numpy.full_like(self.beta, numpy.nan)

        self.decay = decay
        self.N = [0]  # as a reference
        self.eps = eps

    def __call__(self, x, test=False, finetune=False):
        """Invokes the forward propagation of BatchNormalization.

        BatchNormalization accepts additional arguments, which controlls three
        different running mode.

        Args:
            x (Variable): An input variable.
            test (bool): If ``True``, BatchNormalization runs in testing mode;
                it normalizes the input using precomputed statistics.
            finetune (bool): If ``True``, BatchNormalization runs in finetuning
                mode; it accumulates the input array to compute population
                statistics for normalization, and normalizes the input using
                batch statistics.

        If ``test`` and ``finetune`` are both ``False``, then
        BatchNormalization runs in training mode; it computes moving averages
        of mean and variance for evaluation during training, and normalizes the
        input using batch statistics.

        """
        self.use_batch_mean = not test or finetune
        self.is_finetune = finetune
        return function.Function.__call__(self, x)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        self_ = type_check.Variable(self, 'self')
        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim >= self_.size.__len__() + 1,
            x_type.shape[1:len(self.size)+1] == self_.size
        )

    def start_finetuning(self):
        self.N[0] = numpy.array(0)

    def forward(self, x_orig):
        xp = cuda.get_array_module(*x_orig)
        ldim, cdim, rdim = self._internal_shape(x_orig[0])
        x = x_orig[0].reshape(ldim, cdim, rdim)

        if self.use_batch_mean:
            mean = x.mean(axis=(0, 2), keepdims=True)
            var = x.var(axis=(0, 2), keepdims=True)
            var += self.eps
        else:
            mean = self.avg_mean
            var = self.avg_var

        self.std = xp.sqrt(var, dtype=var.dtype)
        x_mu = x - mean
        self.x_hat = x_mu / self.std
        y = self.gamma * self.x_hat
        y += self.beta

        # Compute exponential moving average
        if self.use_batch_mean:
            if self.is_finetune:
                self.N[0] += 1
                decay = 1. / self.N[0]
            else:
                decay = self.decay

            m = ldim * rdim
            adjust = m / max(m - 1., 1.)  # unbiased estimation
            self.avg_mean *= decay
            self.avg_mean += (1 - decay) * mean
            self.avg_var *= decay
            self.avg_var += (1 - decay) * adjust * var

        return y.reshape(x_orig[0].shape),

    def backward(self, x_orig, gy):
        # TODO(beam2d): Support backprop on inference mode
        assert self.use_batch_mean and not self.is_finetune

        ldim, cdim, rdim = self._internal_shape(x_orig[0])
        gy = gy[0].reshape(ldim, cdim, rdim)
        m = ldim * rdim

        gbeta = gy.sum(axis=(0, 2), keepdims=True)
        self.gbeta += gbeta

        ggamma = (gy * self.x_hat).sum(axis=(0, 2), keepdims=True)
        self.ggamma += ggamma

        coeff = self.gamma / self.std
        gbeta /= m
        ggamma /= m

        gx = coeff * (gy - self.x_hat * ggamma - gbeta)
        return gx.reshape(x_orig[0].shape),

    def _internal_shape(self, x):
        ldim = x.shape[0]
        cdim = self.gamma.size
        rdim = x.size // (ldim * cdim)
        assert ldim * cdim * rdim == x.size
        return map(numpy.int32, (ldim, cdim, rdim))

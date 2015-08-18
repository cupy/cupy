import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _kernel_with_I(args, expr, name):
    return cuda.elementwise(
        '{}, int cdim, int rdim'.format(args),
        'int I = i / rdim % cdim; {};'.format(expr),
        name)

_one = None


def _partial_reduce(x):
    global _one
    out_axis, sum_axis = x.shape
    one = _one
    if one is None or one.size < sum_axis:
        one = cuda.ones(sum_axis)
        _one = one
    one = one[:sum_axis]
    handle = cuda.get_cublas_handle()
    ret = cuda.empty(out_axis)
    cuda.cublas.cublasSgemv(handle, 't', sum_axis, out_axis,
                            numpy.float32(
                                1.0), x.gpudata, sum_axis, one.gpudata,
                            1, numpy.float32(0.0), ret.gpudata, 1)
    return ret

if cuda.available:
    @cuda.cutools.context_dependent_memoize
    def _create_reduction_kernel(shape0, expr1, expr2):
        return cuda.elementwise(
            '''
                float* ret1, float* ret2,
                const float* x, const float* y,
                float alpha, int shape12
            ''', '''
                float sum1 = 0, sum2 = 0;
                for (int j = 0; j < {0}; j++) {{
                    int I = j * shape12 + i;
                    sum1 += {1};
                    sum2 += {2};
                }}
                ret1[i] = sum1 * alpha;
                ret2[i] = sum2 * alpha;
            '''.format(shape0, expr1, expr2), 'bn_asix02')


def _cusum_axis02(x, y=None, expr1='x[I]', expr2='x[I] * x[I]', mean=False):
    with cuda.using_cumisc():
        shape = x.shape
        ret1 = cuda.empty_like(x[0])
        ret2 = cuda.empty_like(x[0])
        if y is None:
            y = x
        alpha = 1.0
        if mean:
            alpha = 1.0 / (shape[0] * shape[2])

        # In most cases shape[0] is constant.
        # Therefore, the kernel is compiled only once.
        # If shape[0] is small, Compiler will perform loop unrolling.
        _create_reduction_kernel(shape[0], expr1, expr2)(
            ret1, ret2, x, y, alpha, shape[1] * shape[2])

        if shape[2] != 1:
            ret1 = _partial_reduce(ret1)
            ret2 = _partial_reduce(ret2)
        ret_shape = (1, shape[1], 1)
        return (ret1.reshape(ret_shape), ret2.reshape(ret_shape))


class BatchNormalization(function.Function):

    """Batch normalization on outputs of linear or convolution functions.

    Args:
        size (int or tuple of ints): Size (or shape) of channel
            dimensions.
        decay (float): Decay rate of moving average.
        eps (float): Epsilon value for numerical stability.

    See: `Batch Normalization: Accelerating Deep Network Training by Reducing\
          Internal Covariate Shift <http://arxiv.org/abs/1502.03167>`_

    """
    parameter_names = ('gamma',  'beta')
    gradient_names = ('ggamma', 'gbeta')

    def __init__(self, size, decay=0.9, eps=1e-5):
        if isinstance(size, tuple):
            self.size = size
        elif isinstance(size, int):
            self.size = (size, )
        else:
            raise TypeError('size must be tuple or int')

        size = numpy.prod(size)

        self.avg_mean = numpy.zeros((1, size, 1), dtype=numpy.float32)
        self.avg_var = numpy.zeros_like(self.avg_mean)

        self.gamma = numpy.ones_like(self.avg_mean)
        self.ggamma = numpy.empty_like(self.gamma)
        self.beta = numpy.zeros_like(self.avg_mean)
        self.gbeta = numpy.empty_like(self.beta)

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

    def forward_cpu(self, x_orig):
        ldim, cdim, rdim = self._internal_shape(x_orig[0])
        x = x_orig[0].reshape(ldim, cdim, rdim)

        if self.use_batch_mean:
            mean = x.mean(axis=(0, 2), keepdims=True)
            var = x.var(axis=(0, 2), keepdims=True) + self.eps
        else:
            mean = self.avg_mean
            var = self.avg_var

        self.std = numpy.sqrt(var)
        x_mu = x - mean
        self.x_hat = x_mu / self.std
        y = self.gamma * self.x_hat + self.beta

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
            self.avg_mean += (1 - decay) * adjust * mean
            self.avg_var *= decay
            self.avg_var += (1 - decay) * adjust * var

        return y.reshape(x_orig[0].shape),

    def forward_gpu(self, x_orig):
        ldim, cdim, rdim = self._internal_shape(x_orig[0])
        x = x_orig[0].reshape(ldim, cdim, rdim)

        if self.use_batch_mean:
            mean, sqmean = _cusum_axis02(x, mean=True)
            var = sqmean  # reuse buffer
            cuda.elementwise(
                'float* var, const float* mean, float eps',
                'var[i] = var[i] - mean[i] * mean[i] + eps',
                'bn_var')(var, mean, self.eps)
        else:
            mean = self.avg_mean
            var = self.avg_var

        y = cuda.empty_like(x_orig[0])
        _kernel_with_I(
            '''
                float* y, const float* x,
                const float* mean, const float* var,
                const float* gamma, const float* beta
            ''',
            'y[i] = (x[i] - mean[I]) * rsqrtf(var[I]) * gamma[I] + beta[I];',
            'bn_fwd')(y, x, mean, var, self.gamma, self.beta, cdim, rdim)

        # Compute exponential moving average
        if self.use_batch_mean:
            if self.is_finetune:
                self.N[0] += 1
                decay = 1. / self.N[0]
            else:
                decay = self.decay

            m = ldim * rdim
            adjust = m / max(m - 1., 1.)  # unbiased estimation
            cuda.elementwise(
                '''
                   float* avg_mean, const float* mean,
                   float* avg_var, const float* var,
                   float decay, float adjust
                ''', '''
                   avg_mean[i] = decay * avg_mean[i]
                                 + (1 - decay) * adjust * mean[i];
                   avg_var[i]  = decay * avg_var[i]
                                 + (1 - decay) * adjust * var[i];
                ''',
                'bn_moving_avg')(
                    self.avg_mean, mean, self.avg_var, var, decay, adjust)

        return y,

    def backward_cpu(self, x_orig, gy):
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

    def backward_gpu(self, x_orig, gy):
        # TODO(beam2d): Support backprop on inference mode
        assert self.use_batch_mean and not self.is_finetune
        ldim, cdim, rdim = self._internal_shape(x_orig[0])
        x = x_orig[0].reshape(ldim, cdim, rdim)
        gy = gy[0].reshape(ldim, cdim, rdim)
        m = ldim * rdim

        mean, sqmean = _cusum_axis02(x, mean=True)
        stdinv = sqmean  # reuse buffer
        cuda.elementwise(
            'float* stdinv, const float* mean, float eps',
            'stdinv[i] = rsqrtf(stdinv[i] - mean[i] * mean[i] + eps)',
            'bn_stdinv')(stdinv, mean, self.eps)

        x_hat = cuda.empty_like(x)
        gx = cuda.empty_like(x)

        _kernel_with_I(
            '''
                float* x_hat, const float* x,
                const float* mean, const float* stdinv
            ''', 'x_hat[i] = (x[i] - mean[I]) * stdinv[I]',
            'bn_x_hat')(x_hat, x, mean, stdinv, cdim, rdim)
        mean = None

        gbeta, ggamma = _cusum_axis02(gy, x_hat, expr2='x[I] * y[I]')
        cuda.elementwise(
            '''
                float* self_ggammma, const float* ggamma,
                float* slef_gbeta, const float* gbeta
            ''', '''
                self_ggammma[i] += ggamma[i];
                slef_gbeta[i] += gbeta[i];
            ''', 'bn_add')(
                self.ggamma, ggamma,
                self.gbeta, gbeta)

        _kernel_with_I(
            '''
                float* gx, const float* x_hat,
                const float* gy, const float* stdinv,
                const float* ggamma, const float* gbeta,
                const float* gamma, float inv_m
            ''', '''
                gx[i] = gamma[I] * stdinv[I] *
                    (gy[i] - (x_hat[i] * ggamma[I] + gbeta[I]) * inv_m)
            ''', 'bn_bwd')(
                gx, x_hat, gy, stdinv, ggamma, gbeta,
                self.gamma, 1. / m, cdim, rdim)
        return gx.reshape(x_orig[0].shape),

    def _internal_shape(self, x):
        ldim = x.shape[0]
        cdim = self.gamma.size
        rdim = x.size // (ldim * cdim)
        assert ldim * cdim * rdim == x.size
        return ldim, cdim, rdim

import numpy
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel
from pycuda import cumath
from pytools import memoize
import scikits.cuda.misc as cumisc
from chainer import Function

def _kernel_with_I(args, expr):
    return ElementwiseKernel(
        '{}, int cdim, int rdim'.format(args),
        'int I = i / rdim % cdim; {};'.format(expr))

@memoize
def _var_kernel():
    return ElementwiseKernel(
        'float* var, const float* mean, const float* sqmean, float eps',
        'var[i] = sqmean[i] - mean[i] * mean[i] + eps')

@memoize
def _stdinv_kernel():
    return ElementwiseKernel(
        'float* stdinv, const float* mean, const float* sqmean, float eps',
        'stdinv[i] = rsqrtf(sqmean[i] - mean[i] * mean[i] + eps)')

@memoize
def _coeff_kernel():
    return ElementwiseKernel(
        'float* coeff, const float* var, const float* gamma',
        'coeff[i] = rsqrtf(var[i]) * gamma[i]')

@memoize
def _bias_kernel():
    return ElementwiseKernel(
        'float* bias, const float* coeff, const float* mean, const float* beta',
        'bias[i] = beta[i] - coeff[i] * mean[i]')

@memoize
def _forward_kernel():
    return _kernel_with_I(
        'float* y, const float* x, const float* coeff, const float* bias',
        'y[i] = coeff[I] * x[i] + bias[I]')

@memoize
def _moving_avg_kernel():
    return ElementwiseKernel(
        'float* mean, const float* x, float decay, float adjust',
        'mean[i] = decay * mean[i] + (1 - decay) * adjust * x[i]')

@memoize
def _x_hat_kernel():
    return _kernel_with_I(
        'float* x_hat, const float* x, const float* mean, const float* stdinv',
        'x_hat[i] = (x[i] - mean[I]) * stdinv[I]')

@memoize
def _backward_kernel():
    return _kernel_with_I(
        '''
          float* gx, const float* x_hat, const float* gy, const float* coeff,
          const float* ggamma, const float* gbeta
        ''',
        'gx[i] = coeff[I] * (gy[i] - x_hat[i] * ggamma[I] - gbeta[I])')

def _cumean_axis02(x):
    if x.shape[2] > 1:
        # cumisc.mean does not support more than two dimensions
        shape = x.shape
        x = x.reshape(shape[0] * shape[1], shape[2])
        x = cumisc.mean(x, axis=1)
        x = x.reshape(shape[0], shape[1])
    else:
        x = x.reshape(x.shape[:2])
    return cumisc.mean(x, axis=0)

def _cusum_axis02(x):
    if x.shape[2] > 1:
        # cumisc.sum does not support more than two dimensions
        shape = x.shape
        x = x.reshape(shape[0] * shape[1], shape[2])
        x = cumisc.sum(x, axis=1)
        x = x.reshape(shape[0], shape[1])
    else:
        x = x.reshape(x.shape[:2])
    return cumisc.sum(x, axis=0)


class BatchNormalization(Function):
    """Batch normalization on outputs of linear or convolution function.

    See: http://arxiv.org/abs/1502.03167

    """
    parameter_names = ( 'gamma',  'beta')
    gradient_names  = ('ggamma', 'gbeta')

    def __init__(self, size, decay=0.9, eps=1e-5):
        size = numpy.prod(size)

        self.avg_mean = numpy.zeros((1, size, 1), dtype=numpy.float32)
        self.avg_var  = numpy.zeros_like(self.avg_mean)

        self.gamma  = numpy.ones_like(self.avg_mean)
        self.ggamma = numpy.empty_like(self.gamma)
        self.beta   = numpy.zeros_like(self.avg_mean)
        self.gbeta  = numpy.empty_like(self.beta)

        self.decay = decay
        self.N     = numpy.array(0)  # as a reference
        self.eps   = eps

    def __call__(self, x, test=False, finetune=False):
        self.use_batch_mean = not test or finetune
        self.is_finetune    = finetune
        return Function.__call__(self, x)

    def start_finetuning(self):
        self.N = numpy.array(0)

    def forward_cpu(self, x_orig):
        self._to_cpu()
        ldim, cdim, rdim = self._internal_shape(x_orig[0])
        x = x_orig[0].reshape(ldim, cdim, rdim)

        if self.use_batch_mean:
            mean = x.mean(axis=(0, 2), keepdims=True)
            var  = x.var(axis=(0, 2), keepdims=True) + self.eps
        else:
            mean = self.avg_mean
            var  = self.avg_var

        self.std   = numpy.sqrt(var)
        x_mu       = x - mean
        self.x_hat = x_mu / self.std
        y = self.gamma * self.x_hat + self.beta

        # Compute exponential moving average
        if self.use_batch_mean:
            if self.is_finetune:
                self.N += 1
                decay = 1. / self.N
            else:
                decay = self.decay

            m = ldim * rdim
            adjust = m / max(m - 1., 1.)  # unbiased estimation
            self.avg_mean *= decay
            self.avg_mean += (1 - decay) * adjust * mean
            self.avg_var  *= decay
            self.avg_var  += (1 - decay) * adjust * var

        return y.reshape(x_orig[0].shape),

    def forward_gpu(self, x_orig):
        self._to_gpu()
        ldim, cdim, rdim = self._internal_shape(x_orig[0])
        x = x_orig[0].reshape(ldim, cdim, rdim)

        if self.use_batch_mean:
            mean   = _cumean_axis02(x)
            sqmean = _cumean_axis02(x * x)
            var    = sqmean  # reuse buffer
            _var_kernel()(var, mean, sqmean, self.eps)
        else:
            mean = self.avg_mean
            var  = self.avg_var

        coeff = gpuarray.empty_like(var)
        _coeff_kernel()(coeff, var, self.gamma)
        bias  = gpuarray.empty_like(var)
        _bias_kernel()(bias, coeff, mean, self.beta)

        y = gpuarray.empty_like(x_orig[0])
        _forward_kernel()(y, x, coeff, bias, cdim, rdim)

        # Compute exponential moving average
        if self.use_batch_mean:
            if self.is_finetune:
                self.N += 1
                decay = 1. / self.N
            else:
                decay = self.decay

            m = ldim * rdim
            adjust = m / max(m - 1., 1.)  # unbiased estimation
            kern = _moving_avg_kernel()
            kern(self.avg_mean, mean, decay, adjust)
            kern(self.avg_var,  var,  decay, adjust)

        return y,

    def backward_cpu(self, x_orig, gy):
        # TODO(beam2d): Support backprop on inference mode
        assert self.use_batch_mean and not self.is_finetune
        ldim, cdim, rdim = self._internal_shape(x_orig[0])
        x  = x_orig[0].reshape(ldim, cdim, rdim)
        gy =     gy[0].reshape(ldim, cdim, rdim)
        m = ldim * rdim

        gbeta = gy.sum(axis=(0, 2), keepdims=True)
        self.gbeta += gbeta

        ggamma = (gy * self.x_hat).sum(axis=(0, 2), keepdims=True)
        self.ggamma += ggamma

        coeff = self.gamma / self.std
        gbeta  /= m
        ggamma /= m

        gx = coeff * (gy - self.x_hat * ggamma - gbeta)
        return gx.reshape(x_orig[0].shape),

    def backward_gpu(self, x_orig, gy):
        # TODO(beam2d): Support backprop on inference mode
        assert self.use_batch_mean and not self.is_finetune
        ldim, cdim, rdim = self._internal_shape(x_orig[0])
        x  = x_orig[0].reshape(ldim, cdim, rdim)
        gy = gy[0].reshape(ldim, cdim, rdim)
        m = ldim * rdim

        mean   = _cumean_axis02(x)
        sqmean = _cumean_axis02(x * x)
        stdinv = sqmean  # reuse buffer
        _stdinv_kernel()(stdinv, mean, sqmean, self.eps)

        x_hat = gpuarray.empty_like(x)
        _x_hat_kernel()(x_hat, x, mean, stdinv, cdim, rdim)
        mean = None

        ggamma = _cusum_axis02(x_hat * gy)
        gbeta  = _cusum_axis02(gy)
        self.ggamma += ggamma.reshape(self.ggamma.shape)
        self.gbeta  += gbeta.reshape(self.gbeta.shape)

        coeff = stdinv  # reuse buffer
        coeff *= self.gamma
        ggamma /= m
        gbeta  /= m
        gx = gpuarray.empty_like(x)
        _backward_kernel()(gx, x_hat, gy, coeff, ggamma, gbeta, cdim, rdim)

        return gx.reshape(x_orig[0].shape),

    def _to_cpu(self):
        if type(self.avg_mean) == gpuarray.GPUArray:
            self.avg_mean = self.avg_mean.get()
        if type(self.avg_var) == gpuarray.GPUArray:
            self.avg_var = self.avg_variange.get()

    def _to_gpu(self):
        if type(self.avg_mean) == numpy.ndarray:
            self.avg_mean = gpuarray.to_gpu(self.avg_mean)
        if type(self.avg_var) == numpy.ndarray:
            self.avg_var = gpuarray.to_gpu(self.avg_var)

    def _internal_shape(self, x):
        ldim = x.shape[0]
        cdim = self.gamma.size
        rdim = x.size / (ldim * cdim)
        assert ldim * cdim * rdim == x.size
        return ldim, cdim, rdim

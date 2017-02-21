import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn
    _cudnn_version = libcudnn.getVersion()


def _as4darray(arr):
    if arr.ndim == 0:
        return arr.reshape(1, 1, 1, 1)
    elif arr.ndim == 4:
        return arr
    else:
        return arr.reshape(arr.shape[0], -1, 1, 1)


def _xhat(x, mean, std, expander):
    x_mu = x - mean[expander]
    x_mu /= std[expander]
    return x_mu


class BatchNormalizationFunction(function.Function):

    def __init__(self, eps=2e-5, mean=None, var=None, train=False,
                 decay=0.9, use_cudnn=True):
        self.running_mean = mean
        self.running_var = var

        # If train is true, use batch statistics (training mode). Otherwise, if
        # false, use the supplied mean and variance.
        self.train = train
        # Note: cuDNN v5 requires that eps be greater than 1e-5. Otherwise, an
        # error will occur.
        # See CUDNN_BN_MIN_EPSILON value in cudnn.h to verify minimum allowable
        # value.
        self.eps = eps
        if cuda.cudnn_enabled and use_cudnn:
            if eps < 1e-5:
                msg = 'cuDNN does not allow an eps value less than 1e-5.'
                raise RuntimeError(msg)
        self.use_cudnn = use_cudnn
        self.mean_cache = None
        self.decay = decay

    def check_type_forward(self, in_types):
        n_in = in_types.size().eval()
        if n_in != 3 and n_in != 5:
            raise type_check.InvalidType(
                '%s or %s' % (in_types.size() == 3, in_types.size() == 5),
                '%s == %s' % (in_types.size(), n_in))
        x_type, gamma_type, beta_type = in_types[:3]
        M = gamma_type.ndim.eval()
        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim >= gamma_type.ndim + 1,
            x_type.shape[1:1 + M] == gamma_type.shape,
            # TODO(beam2d): Check shape
            gamma_type.dtype == x_type.dtype,
            beta_type.dtype == x_type.dtype,
            gamma_type.shape == beta_type.shape,
        )
        if len(in_types) == 5:
            mean_type, var_type = in_types[3:]
            type_check.expect(
                mean_type.dtype == x_type.dtype,
                mean_type.shape == gamma_type.shape,
                var_type.dtype == x_type.dtype,
                var_type.shape == gamma_type.shape,
            )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, gamma, beta = inputs[:3]
        if self.train:
            if self.running_mean is None:
                self.running_mean = xp.zeros_like(gamma)
                self.running_var = xp.zeros_like(gamma)
            else:
                self.running_mean = xp.array(self.running_mean)
                self.running_var = xp.array(self.running_var)
        elif len(inputs) == 5:
            self.fixed_mean = inputs[3]
            self.fixed_var = inputs[4]

        # TODO(bkvogel): Check for float16 support again in next cuDNN version.
        if x[0].dtype == numpy.float16:
            # cuDNN v5 batch normalization does not seem to support float16.
            self.use_cudnn = False

        head_ndim = gamma.ndim + 1
        expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)
        gamma = gamma[expander]
        beta = beta[expander]

        # cuDNN only supports these tensor dimensions because they are
        # the most commonly used. If there is a need to support other
        # dimensions with cuDNN, we could consider reshaping the input
        # into a 2-dim array with channels as second dim and m=<product
        # of all dimensions except the 2nd dimension> as the first
        # dimension.
        self.cudnn_dim_ok = x.ndim == 2 or x.ndim == 4

        cudnn_updated_running_stats = False
        if xp is not numpy and cuda.cudnn_enabled and self.use_cudnn and \
                self.cudnn_dim_ok and _cudnn_version >= 5000:
            if x.ndim == 4:
                # for convolutional layer
                self.mode = libcudnn.CUDNN_BATCHNORM_SPATIAL
            else:
                # for linear layer
                self.mode = libcudnn.CUDNN_BATCHNORM_PER_ACTIVATION

            x = cuda.cupy.ascontiguousarray(x)
            gamma = cuda.cupy.ascontiguousarray(gamma)
            beta = cuda.cupy.ascontiguousarray(beta)
            dtype = x.dtype
            handle = cudnn.get_handle()
            x_desc = cudnn.create_tensor_descriptor(_as4darray(x))
            derivedBnDesc = cudnn.create_uninitialized_tensor_descriptor()
            libcudnn.deriveBNTensorDescriptor(derivedBnDesc.value,
                                              x_desc.value, self.mode)
            one = numpy.array(1, dtype=dtype).ctypes
            zero = numpy.array(0, dtype=dtype).ctypes
            y = cuda.cupy.empty_like(x)
            # Factor used in the moving average
            factor = 1 - self.decay

            if self.train:
                if self.mean_cache is None:
                    # Output cache to speed up backward pass.
                    self.mean_cache = xp.empty_like(gamma)
                    # Output cache to speed up backward pass.
                    self.var_cache = xp.empty_like(gamma)
                # Note: cuDNN computes the mini-batch mean and variance
                # internally. We can simply (optionally) pass
                # it the running-average mean and variance arrays.
                libcudnn.batchNormalizationForwardTraining(
                    handle, self.mode, one.data, zero.data,
                    x_desc.value, x.data.ptr, x_desc.value,
                    y.data.ptr, derivedBnDesc.value, gamma.data.ptr,
                    beta.data.ptr, factor, self.running_mean.data.ptr,
                    self.running_var.data.ptr, self.eps,
                    self.mean_cache.data.ptr, self.var_cache.data.ptr)
                cudnn_updated_running_stats = True
            else:
                libcudnn.batchNormalizationForwardInference(
                    handle, self.mode, one.data, zero.data,
                    x_desc.value, x.data.ptr, x_desc.value, y.data.ptr,
                    derivedBnDesc.value, gamma.data.ptr, beta.data.ptr,
                    self.fixed_mean.data.ptr, self.fixed_var.data.ptr,
                    self.eps)
        else:
            if self.train:
                axis = (0,) + tuple(range(head_ndim, x.ndim))
                mean = x.mean(axis=axis)
                var = x.var(axis=axis)
                var += self.eps
            else:
                mean = self.fixed_mean
                var = self.fixed_var + self.eps
            self.std = xp.sqrt(var, dtype=var.dtype)
            if xp is numpy:
                self.x_hat = _xhat(x, mean, self.std, expander)
                y = gamma * self.x_hat
                y += beta
            else:
                self.x_hat, y = cuda.elementwise(
                    'T x, T mean, T std, T gamma, T beta', 'T x_hat, T y',
                    '''
                    x_hat = (x - mean) / std;
                    y = gamma * x_hat + beta;
                    ''',
                    'bn_fwd')(x, mean[expander], self.std[expander], gamma,
                              beta)

        if self.train and (not cudnn_updated_running_stats):
            # Note: If in training mode, the cuDNN forward training function
            # will do this for us, so
            # only run following code if cuDNN was not used.
            # Update running statistics:
            m = x.size // gamma.size
            adjust = m / max(m - 1., 1.)  # unbiased estimation
            self.running_mean *= self.decay
            temp_ar = xp.array(mean)
            temp_ar *= (1 - self.decay)
            self.running_mean += temp_ar
            del temp_ar
            self.running_var *= self.decay
            temp_ar = xp.array(var)
            temp_ar *= (1 - self.decay) * adjust
            self.running_var += temp_ar
            del temp_ar
        return y,

    def backward(self, inputs, grad_outputs):
        x, gamma = inputs[:2]
        gy = grad_outputs[0]
        head_ndim = gamma.ndim + 1
        expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)
        m = gamma.dtype.type(x.size // gamma.size)
        axis = (0,) + tuple(range(head_ndim, x.ndim))
        xp = cuda.get_array_module(x)
        if len(inputs) == 5:
            # This case is unlikely to be used in practice and so does not
            # need to be optimized for performance.
            mean = inputs[3]
            var = inputs[4]
            std = xp.sqrt(var, dtype=var.dtype)
            gs = gamma / std
            gbeta = gy.sum(axis=axis)
            x_hat = _xhat(x, mean, std, expander)
            ggamma = (gy * x_hat).sum(axis=axis)
            gmean = -gs * gbeta
            gvar = -0.5 * gamma / var * ggamma
            gx = gs[expander] * gy
            return gx, ggamma, gbeta, gmean, gvar

        # Note: If length of inputs is not 5, we must be in train mode.
        assert self.train
        if xp is not numpy and cuda.cudnn_enabled and self.use_cudnn and \
                self.cudnn_dim_ok and _cudnn_version >= 5000:
            # Note: cuDNN batch normalization backward only works in
            # "training mode." That is, it does not support
            # computing gradients in fixed-mean-variance mode, because there
            # is normally no reason to call backward()
            # while in test/evaluation mode.
            dtype = x.dtype
            handle = cudnn.get_handle()
            x_desc = cudnn.create_tensor_descriptor(_as4darray(x))
            derivedBnDesc = cudnn.create_uninitialized_tensor_descriptor()
            libcudnn.deriveBNTensorDescriptor(derivedBnDesc.value,
                                              x_desc.value, self.mode)
            one = numpy.array(1, dtype=dtype).ctypes
            zero = numpy.array(0, dtype=dtype).ctypes
            gx = cuda.cupy.empty_like(x)
            ggamma = cuda.cupy.empty_like(gamma)
            gbeta = cuda.cupy.empty_like(gamma)
            libcudnn.batchNormalizationBackward(
                handle, self.mode, one.data, zero.data,
                one.data, zero.data, x_desc.value, x.data.ptr,
                x_desc.value, gy.data.ptr, x_desc.value, gx.data.ptr,
                derivedBnDesc.value, gamma.data.ptr,
                ggamma.data.ptr, gbeta.data.ptr,
                self.eps, self.mean_cache.data.ptr, self.var_cache.data.ptr)
        else:
            gbeta = gy.sum(axis=axis)
            ggamma = (gy * self.x_hat).sum(axis=axis)
            if xp is numpy:
                gx = (gamma / self.std)[expander] * (
                    gy - (self.x_hat * ggamma[expander] + gbeta[expander]) / m)
            else:
                inv_m = numpy.float32(1) / m
                gx = cuda.elementwise(
                    'T gy, T x_hat, T gamma, T std, T ggamma, T gbeta, \
                    T inv_m',
                    'T gx',
                    'gx = (gamma / std) * (gy - (x_hat * ggamma + gbeta) * \
                    inv_m)',
                    'bn_bwd')(gy, self.x_hat, gamma[expander],
                              self.std[expander], ggamma[expander],
                              gbeta[expander], inv_m)
        return gx, ggamma, gbeta


def batch_normalization(x, gamma, beta, eps=2e-5, running_mean=None,
                        running_var=None, decay=0.9, use_cudnn=True):
    """Batch normalization function.

    It takes the input variable ``x`` and two parameter variables ``gamma`` and
    ``beta``. The input must have the batch size and the features (or channels)
    as the first two dimensions of its shape. The input can have more than two
    dimensions, where the remaining dimensions are considered as spatial
    dimensions, which are considered as a part of the batch size. That is,
    the total batch size will be considered to be the product of all
    dimensions except the second dimension.

    Note: If this function is called, it will not be possible to access the
    updated running mean and variance statistics, because they are members
    of the function object, which cannot be accessed by the caller.
    If it is desired to access the updated running statistics, it is necessary
    to get a new instance of the function object, call the object, and then
    access the running_mean and/or running_var attributes. See the
    corresponding Link class for an example of how to do this.

    Args:
        x (Variable): Input variable.
        gamma (Variable): Scaling parameter of normalized data.
        beta (Variable): Shifting parameter of scaled normalized data.
        eps (float): Epsilon value for numerical stability.
        running_mean (array): Running average of the mean. This is a
            running average of the mean over several mini-batches using
            the decay parameter. If ``None``, the running average is not
            computed. If this is ``None``, then ``runnng_var`` must also
            be ``None``.
        running_var (array): Running average of the variance. This is a
            running average of the variance over several mini-batches using
            the decay parameter. If ``None``, the running average is not
            computed. If this is ``None``, then ``running_mean`` must also
            be ``None``.
        decay (float): Decay rate of moving average. It is used during
            training.
        use_cudnn (bool): If ``True`` and cuDNN is enabled, then this function
            uses cuDNN as the core implementation.


    See: `Batch Normalization: Accelerating Deep Network Training by Reducing\
          Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_

    .. seealso:: :class:`links.BatchNormalization`

    """
    return BatchNormalizationFunction(eps, running_mean, running_var, True,
                                      decay, use_cudnn)(x, gamma, beta)


def fixed_batch_normalization(x, gamma, beta, mean, var, eps=2e-5,
                              use_cudnn=True):
    """Batch normalization function with fixed statistics.

    This is a variant of batch normalization, where the mean and variance
    statistics are given by the caller as fixed variables. This is
    used on testing mode of the batch normalization layer, where batch
    statistics cannot be used for prediction consistency.

    Args:
        x (Variable): Input variable.
        gamma (Variable): Scaling parameter of normalized data.
        beta (Variable): Shifting parameter of scaled normalized data.
        mean (Variable): Shifting parameter of input.
        var (Variable): Square of scaling parameter of input.
        eps (float): Epsilon value for numerical stability.
        use_cudnn (bool): If ``True`` and cuDNN is enabled, then this function
            uses cuDNN as the core implementation.

    .. seealso::
       :func:`functions.batch_normalization`,
       :class:`links.BatchNormalization`

    """
    return BatchNormalizationFunction(eps, None, None, False, 0.0,
                                      use_cudnn)(x, gamma, beta, mean, var)

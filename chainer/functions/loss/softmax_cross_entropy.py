import numpy
import six

import chainer
from chainer import cuda
from chainer import function
from chainer.utils import type_check


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn
    _algorithm = libcudnn.CUDNN_SOFTMAX_LOG
    _mode = libcudnn.CUDNN_SOFTMAX_MODE_CHANNEL
    _cudnn_version = libcudnn.getVersion()


def logsumexp(x):
    xp = cuda.get_array_module(x)
    m = x.max(axis=1, keepdims=True)
    y = x - m
    xp.exp(y, out=y)
    return xp.log(y.sum(axis=1, keepdims=True)) + m


def softmax_log(x, use_cudnn):
    xp = cuda.get_array_module(x)
    if xp != numpy and cuda.cudnn_enabled and use_cudnn \
       and _cudnn_version >= 3000:
        dtype = x.dtype
        one = numpy.array(1, dtype=dtype).ctypes
        zero = numpy.array(0, dtype=dtype).ctypes
        handle = cudnn.get_handle()
        x_cube = x.reshape(x.shape[:2] + (-1, 1))
        desc = cudnn.create_tensor_descriptor(x_cube)
        y = xp.empty_like(x)
        libcudnn.softmaxForward(
            handle, _algorithm, _mode, one.data, desc.value,
            x_cube.data.ptr, zero.data, desc.value,
            y.data.ptr)
        return y

    else:
        log_z = logsumexp(x)
        return x - log_z


class SoftmaxCrossEntropy(function.Function):

    """Softmax activation followed by a cross entropy loss."""

    ignore_label = -1

    def __init__(self, use_cudnn=True, normalize=True, cache_score=True):
        self.use_cudnn = use_cudnn
        self.normalize = normalize
        self.cache_score = cache_score

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, t_type = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
            t_type.dtype == numpy.int32,
            t_type.ndim == x_type.ndim - 1,

            x_type.shape[0] == t_type.shape[0],
            x_type.shape[2:] == t_type.shape[1:],
        )

    def _check_input_values(self, x, t):
        if not (((0 <= t) &
                 (t < x.shape[1])) |
                (t == self.ignore_label)).all():
            msg = ('Each label `t` need to satisfty '
                   '`0 <= t < x.shape[1] or t == %d`' % self.ignore_label)
            raise ValueError(msg)

    def forward_cpu(self, inputs):
        x, t = inputs
        if chainer.is_debug():
            self._check_input_values(x, t)

        log_y = softmax_log(x, False)
        if self.cache_score:
            self.y = numpy.exp(log_y)
        log_yd = numpy.rollaxis(log_y, 1)
        log_yd = log_yd.reshape(len(log_yd), -1)

        log_p = log_yd[numpy.maximum(t.ravel(), 0), six.moves.range(t.size)]
        # deal with the case where the SoftmaxCrossEntropy is
        # unpickled from the old version
        if getattr(self, 'normalize', True):
            count = (t != self.ignore_label).sum()
        else:
            count = len(x)
        self._coeff = 1.0 / max(count, 1)

        y = (log_p * (t.ravel() != self.ignore_label)).sum(keepdims=True) \
            * (-self._coeff)
        return y.reshape(()),

    def forward_gpu(self, inputs):
        cupy = cuda.cupy
        x, t = inputs
        if chainer.is_debug():
            self._check_input_values(x, t)

        log_y = softmax_log(x, self.use_cudnn)
        if self.cache_score:
            self.y = cupy.exp(log_y)
        if getattr(self, 'normalize', True):
            coeff = cupy.maximum(1, (t != self.ignore_label).sum())
        else:
            coeff = max(1, len(t))
        self._coeff = cupy.divide(1.0, coeff, dtype=x.dtype)

        log_y = cupy.rollaxis(log_y, 1, log_y.ndim)
        ret = cuda.reduce(
            'S t, raw T log_y, int32 n_channel, raw T coeff', 'T out',
            't == -1 ? 0 : log_y[_j * n_channel + t]',
            'a + b', 'out = a * -coeff[0]', '0', 'crossent_fwd'
        )(t, log_y.reduced_view(), log_y.shape[-1], self._coeff)
        return ret,

    def backward_cpu(self, inputs, grad_outputs):
        x, t = inputs
        gloss = grad_outputs[0]
        n_unit = t.size // len(t)
        if hasattr(self, 'y'):
            y = self.y.copy()
        else:
            log_y = softmax_log(x, self.use_cudnn)
            y = numpy.exp(log_y)
        if y.ndim == 2:
            gx = y
            gx[six.moves.xrange(len(t)), numpy.maximum(t, 0)] -= 1
            gx *= (t != self.ignore_label).reshape((len(t), 1))
        else:
            # in the case where y.ndim is higher than 2,
            # we think that a current implementation is inefficient
            # because it yields two provisional arrays for indexing.
            gx = y.reshape(y.shape[0], y.shape[1], -1)
            fst_index = numpy.arange(t.size) // n_unit
            trd_index = numpy.arange(t.size) % n_unit
            gx[fst_index, numpy.maximum(t.ravel(), 0), trd_index] -= 1
            gx *= (t != self.ignore_label).reshape((len(t), 1, -1))
            gx = gx.reshape(y.shape)

        gx *= gloss * self._coeff
        return gx, None

    def backward_gpu(self, inputs, grad_outputs):
        cupy = cuda.cupy
        x, t = inputs
        if hasattr(self, 'y'):
            y = self.y
        else:
            y = softmax_log(x, self.use_cudnn)
            cupy.exp(y, out=y)
        gloss = grad_outputs[0]
        n_unit = t.size // len(t)
        coeff = gloss * self._coeff
        gx = cuda.elementwise(
            'T y, S t, raw T coeff, S n_channel, S n_unit',
            'T gx',
            '''
               const int c = (i / n_unit % n_channel);
               gx = (t == -1) ? 0 : (coeff[0] * (y - (c == t)));
            ''',
            'softmax_crossent_bwd')(
                y, cupy.expand_dims(t, 1), coeff, x.shape[1], n_unit)
        return gx, None


def softmax_cross_entropy(
        x, t, use_cudnn=True, normalize=True, cache_score=True):
    """Computes cross entropy loss for pre-softmax activations.

    Args:
        x (Variable): Variable holding a multidimensional array whose element
            indicates unnormalized log probability: the first axis of the
            variable represents the number of samples, and the second axis
            represents the number of classes. While this function computes
            a usual softmax cross entropy if the number of dimensions is equal
            to 2, it computes a cross entropy of the replicated softmax if the
            number of dimensions is greater than 2.
        t (Variable): Variable holding an int32 vector of ground truth labels.
            If ``t[i] == -1``, corresponding ``x[i]`` is ignored.
        normalize (Variable): Variable holding a boolean value which
            determines the normalization constant. If true, this function
            normalizes the cross entropy loss across all instances. If else,
            it only normalizes along a batch size.
        cache_score (bool): When it is ``True``, the function stores result
            of forward computation to use it on backward computation. It
            reduces computational cost though consumes more memory.

    Returns:
        Variable: A variable holding a scalar array of the cross entropy loss.

    .. note::

       This function is differentiable only by ``x``.

    """
    return SoftmaxCrossEntropy(use_cudnn, normalize, cache_score)(x, t)

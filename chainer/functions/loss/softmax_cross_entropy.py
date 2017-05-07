import numpy
import six

import chainer
from chainer import cuda
from chainer import function
from chainer.functions.activation import log_softmax
from chainer.utils import type_check


class SoftmaxCrossEntropy(function.Function):

    """Softmax activation followed by a cross entropy loss."""

    normalize = True

    def __init__(self, use_cudnn=True, normalize=True, cache_score=True,
                 class_weight=None, ignore_label=-1, reduce='mean'):
        self.use_cudnn = use_cudnn
        self.normalize = normalize
        self.cache_score = cache_score
        self.class_weight = class_weight
        if class_weight is not None:
            if self.class_weight.ndim != 1:
                raise ValueError('class_weight.ndim should be 1')
            if self.class_weight.dtype.kind != 'f':
                raise ValueError('The dtype of class_weight should be \'f\'')
            if isinstance(self.class_weight, chainer.Variable):
                raise ValueError('class_weight should be a numpy.ndarray or '
                                 'cupy.ndarray, not a chainer.Variable')
        self.ignore_label = ignore_label
        if reduce not in ('mean', 'no'):
            raise ValueError(
                "only 'mean' and 'no' are valid for 'reduce', but '%s' is "
                'given' % reduce)
        self.reduce = reduce

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, t_type = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',
            t_type.dtype == numpy.int32,
            t_type.ndim == x_type.ndim - 1,

            x_type.shape[0] == t_type.shape[0],
            x_type.shape[2:] == t_type.shape[1:],
        )

    def _check_input_values(self, x, t):
        if not (((0 <= t) &
                 (t < x.shape[1])) |
                (t == self.ignore_label)).all():
            msg = ('Each label `t` need to satisfy '
                   '`0 <= t < x.shape[1] or t == %d`' % self.ignore_label)
            raise ValueError(msg)

    def forward_cpu(self, inputs):
        x, t = inputs
        if chainer.is_debug():
            self._check_input_values(x, t)

        log_y = log_softmax._log_softmax(x, self.use_cudnn)
        if self.cache_score:
            self.y = numpy.exp(log_y)
        if self.class_weight is not None:
            shape = [1 if d != 1 else -1 for d in six.moves.range(x.ndim)]
            log_y *= numpy.broadcast_to(
                self.class_weight.reshape(shape), x.shape)
        log_yd = numpy.rollaxis(log_y, 1)
        log_yd = log_yd.reshape(len(log_yd), -1)
        log_p = log_yd[numpy.maximum(t.ravel(), 0), numpy.arange(t.size)]

        log_p *= (t.ravel() != self.ignore_label)
        if self.reduce == 'mean':
            # deal with the case where the SoftmaxCrossEntropy is
            # unpickled from the old version
            if self.normalize:
                count = (t != self.ignore_label).sum()
            else:
                count = len(x)
            self._coeff = 1.0 / max(count, 1)

            y = log_p.sum(keepdims=True) * (-self._coeff)
            return y.reshape(()),
        else:
            return -log_p.reshape(t.shape),

    def forward_gpu(self, inputs):
        cupy = cuda.cupy
        x, t = inputs
        if chainer.is_debug():
            self._check_input_values(x, t)

        log_y = log_softmax._log_softmax(x, self.use_cudnn)
        if self.cache_score:
            self.y = cupy.exp(log_y)
        if self.class_weight is not None:
            shape = [1 if d != 1 else -1 for d in six.moves.range(x.ndim)]
            log_y *= cupy.broadcast_to(
                self.class_weight.reshape(shape), x.shape)
        if self.normalize:
            coeff = cupy.maximum(1, (t != self.ignore_label).sum())
        else:
            coeff = max(1, len(t))
        self._coeff = cupy.divide(1.0, coeff, dtype=x.dtype)

        log_y = cupy.rollaxis(log_y, 1, log_y.ndim)
        if self.reduce == 'mean':
            ret = cuda.reduce(
                'S t, raw T log_y, int32 n_channel, raw T coeff', 'T out',
                't == {} ? T(0) : log_y[_j * n_channel + t]'.format(
                    self.ignore_label),
                'a + b', 'out = a * -coeff[0]', '0', 'crossent_fwd'
            )(t, log_y.reduced_view(), log_y.shape[-1], self._coeff)
        else:
            ret = cuda.elementwise(
                'S t, raw T log_y, int32 n_channel, T ignore', 'T out',
                '''
                if (t == ignore) {
                  out = 0;
                } else {
                  out = -log_y[i * n_channel + t];
                }
                ''',
                'softmax_crossent_no_reduce_fwd'
            )(t, log_y.reduced_view(), log_y.shape[-1], self.ignore_label)
            ret = ret.reshape(t.shape)
        return ret,

    def backward_cpu(self, inputs, grad_outputs):
        x, t = inputs
        gloss = grad_outputs[0]
        if hasattr(self, 'y'):
            y = self.y.copy()
        else:
            y = log_softmax._log_softmax(x, self.use_cudnn)
            numpy.exp(y, out=y)
        if y.ndim == 2:
            gx = y
            gx[numpy.arange(len(t)), numpy.maximum(t, 0)] -= 1
            if self.class_weight is not None:
                shape = [1 if d != 1 else -1 for d in six.moves.range(x.ndim)]
                c = numpy.broadcast_to(
                    self.class_weight.reshape(shape), x.shape)
                c = c[numpy.arange(len(t)), numpy.maximum(t, 0)]
                gx *= numpy.broadcast_to(numpy.expand_dims(c, 1), gx.shape)
            gx *= (t != self.ignore_label).reshape((len(t), 1))
        else:
            # in the case where y.ndim is higher than 2,
            # we think that a current implementation is inefficient
            # because it yields two provisional arrays for indexing.
            n_unit = t.size // len(t)
            gx = y.reshape(y.shape[0], y.shape[1], -1)
            fst_index = numpy.arange(t.size) // n_unit
            trd_index = numpy.arange(t.size) % n_unit
            gx[fst_index, numpy.maximum(t.ravel(), 0), trd_index] -= 1
            if self.class_weight is not None:
                shape = [1 if d != 1 else -1 for d in six.moves.range(x.ndim)]
                c = numpy.broadcast_to(
                    self.class_weight.reshape(shape), x.shape)
                c = c.reshape(gx.shape)
                c = c[fst_index, numpy.maximum(t.ravel(), 0), trd_index]
                c = c.reshape(y.shape[0], 1, -1)
                gx *= numpy.broadcast_to(c, gx.shape)
            gx *= (t != self.ignore_label).reshape((len(t), 1, -1))
            gx = gx.reshape(y.shape)
        if self.reduce == 'mean':
            gx *= gloss * self._coeff
        else:
            gx *= gloss[:, None]
        return gx, None

    def backward_gpu(self, inputs, grad_outputs):
        cupy = cuda.cupy
        x, t = inputs
        if hasattr(self, 'y'):
            y = self.y
        else:
            y = log_softmax._log_softmax(x, self.use_cudnn)
            cupy.exp(y, out=y)
        gloss = grad_outputs[0]
        n_unit = t.size // len(t)
        if self.reduce == 'mean':
            coeff = gloss * self._coeff
        else:
            coeff = gloss[:, None, ...]

        if self.class_weight is None:
            gx = cuda.elementwise(
                'T y, S t, T coeff, S n_channel, S n_unit',
                'T gx',
                '''
                    const int c = (i / n_unit % n_channel);
                    gx = t == {} ? 0 : coeff * (y - (c == t));
                '''.format(self.ignore_label),
                'softmax_crossent_bwd')(
                    y, cupy.expand_dims(t, 1), coeff, x.shape[1], n_unit)
        else:
            gx = cuda.elementwise(
                'T y, raw T w, S t, T coeff, S n_channel, S n_unit',
                'T gx',
                '''
                    const int c = (i / n_unit % n_channel);
                    gx = t == {} ? 0 : coeff * (y - (c == t)) * w[t];
                '''.format(self.ignore_label),
                'softmax_crossent_weight_bwd')(
                    y, self.class_weight, cupy.expand_dims(t, 1), coeff,
                    x.shape[1], n_unit)

        return gx, None


def softmax_cross_entropy(
        x, t, use_cudnn=True, normalize=True, cache_score=True,
        class_weight=None, ignore_label=-1, reduce='mean'):
    """Computes cross entropy loss for pre-softmax activations.

    Args:
        x (~chainer.Variable): Variable holding a multidimensional array whose
            element indicates unnormalized log probability: the first axis of
            the variable represents the number of samples, and the second axis
            represents the number of classes. While this function computes
            a usual softmax cross entropy if the number of dimensions is equal
            to 2, it computes a cross entropy of the replicated softmax if the
            number of dimensions is greater than 2.
        t (~chainer.Variable): Variable holding an int32 vector of ground truth
            labels. If ``t[i] == ignore_label``, corresponding ``x[i]`` is
            ignored.
        normalize (bool): If ``True``, this function normalizes the cross
            entropy loss across all instances. If ``False``, it only
            normalizes along a batch size.
        cache_score (bool): When it is ``True``, the function stores result
            of forward computation to use it on backward computation. It
            reduces computational cost though consumes more memory.
        class_weight (~numpy.ndarray or ~chainer.cuda.cupy.ndarray): An array
            that contains constant weights that will be multiplied with the
            loss values along with the second dimension. The shape of this
            array should be ``(x.shape[1],)``. If this is not ``None``, each
            class weight ``class_weight[i]`` is actually multiplied to
            ``y[:, i]`` that is the corresponding log-softmax output of ``x``
            and has the same shape as ``x`` before calculating the actual loss
            value.
        ignore_label (int): Label value you want to ignore. Its default value
            is ``-1``. See description of the argument `t`.
        reduce (str): A string that determines whether to reduce the loss
            values. If it is ``'mean'``, it computes the sum of the individual
            cross entropy and normalize it according to ``normalize`` option.
            If it is ``'no'``, this function computes cross entropy for each
            instance and does not normalize it (``normalize`` option is
            ignored). In this case, the loss value of the ignored instance,
            which has ``ignore_label`` as its target value, is set to ``0``.

    Returns:
        Variable: A variable holding a scalar array of the cross entropy loss.
        If ``reduce`` is ``'mean'``, it is a scalar array.
        If ``reduce`` is ``'no'``, the shape is same as that of ``x``.

    .. note::

       This function is differentiable only by ``x``.

    """

    return SoftmaxCrossEntropy(
        use_cudnn, normalize, cache_score, class_weight, ignore_label, reduce)(
            x, t)

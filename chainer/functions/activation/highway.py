import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _extract_gates(x):
    r = x.reshape((len(x), x.shape[1] // 4, 4) + x.shape[2:])
    return [r[:, :, i] for i in six.moves.range(4)]


def _sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def _grad_sigmoid(x):
    return x * (1 - x)


def _grad_tanh(x):
    return 1 - x * x


_preamble = '''
template <typename T> __device__ T sigmoid(T x) { return 1 / (1 + exp(-x)); }
template <typename T> __device__ T grad_sigmoid(T y) { return y * (1 - y); }
template <typename T> __device__ T grad_tanh(T y) { return 1 - y * y; }

'''


class Highway(function.Function):

    """Highway Networks
    """
    def check_type_forward(self, in_types):
        n_in = in_types.size().eval()
        if n_in != 3 and n_in != 5:
            raise type_check.InvalidType(
                '%s or %s' % (in_types.size() == 3, in_types.size() == 5),
                '%s == %s' % (in_types.size(), n_in))

        x_type, Wh_type, Wt_type = in_types[:3]
        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim >= Wh_type.ndim + 1,
            Wh_type.dtype == x_type.dtype,
            Wt_type.dtype == x_type.dtype,
            Wh_type.shape == Wt_type.shape,  # Highway net limitation
        )

        if len(in_types) == 5:
            bh_type, bt_type = in_types[3:]
            type_check.expect(
                bh_type.dtype == x_type.dtype,
                bh_type.shape == x_type.shape,
                bt_type.dtype == x_type.dtype,
                bt_type.shape == x_type.shape,
            )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, Wh, Wt = inputs[:3]
        ide = xp.identity(x.shape)

        if len(inputs) == 5:
            bh, bt = inputs[3:]
        else:
            bh = bt = xp.zeros_like(x)

        if isinstance(x, numpy.ndarray):
            self.a = numpy.tanh(x.dot(Wh.T) + bh)
            self.b = _sigmoid(x.dot(Wt.T) + bt)
            self.y = self.a * self.b + x * (ide - self.b)
        else:
            cuda.elementwise(
                'T x, T Wh, T Wt, T bh, T bt, T id, T a_, T b_, T y',
                '''
                    a_ = tanh(x * Wh + bh);
                    b_ = sigmoid(x * Wt + bt);
                    y = a_ * b_ + x * (id - b_);
                ''',
                'hn_fwd', preamble=_preamble)(x, Wh, Wt, bh, bt, ide)

        return y,

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        x, Wh, Wt = inputs[:3]
        gy = grad_outputs[0]
        gx = xp.empty_like(x)
        ones = xp.ones_like(x)

        if xp is numpy:
            gWh = gy.dot((self.b * _grad_tanh(x)).T)
            gWt = gy.dot((self.a * _grad_sigmoid(x).T - x * _grad_sigmoid(x)).T)
            gbh = gy.dot((self.b * _grad_tanh(ones)).T).sum(0)
            gbt = gy.dot((self.a * _grad_sigmoid(ones).T -
                          x * self.a * _grad_sigmoid(ones)).T).sum(0)
            gx = gy.dot((self.b * _grad_tanh(Wh) +
                        (ones - b) +
                         self.a * _grad_sigmoid(Wt) -
                         x.dot(_grad_sigmoid(Wt).T)).T)
            gx = gx.astype(x.dtype).reshape(x.shape)
        else:
            gc_prev = xp.empty_like(c_prev)
            cuda.elementwise(
                'T gx, T sa, T sb, T ones',
                'T gWh, T gWt, T gbh, T gbt',
                '''
                    gWh = gy * self.b * _grad_tanh(x)
                    gWt = gy * (self.a * _grad_sigmoid(x) - x * _grad_sigmoid(x))
                    gbh = gy * (self.b * _grad_tanh(ones))).sum(0)
                    gbt = gy * (self.a * _grad_sigmoid(ones)) -
                          x * self.a * _grad_sigmoid(ones))).sum(0)
                    gx = gy * (self.b * _grad_tanh(Wh) +
                        (ones - b) +
                         self.a * _grad_sigmoid(Wt) -
                         x * _grad_sigmoid(Wt))
                ''',
                'fn_bwd', preamble=_preamble)(
                    gx, self.a, self.b, ones,
                    gWh, gWt, gbh, gbt)

        return gx, gWh, gWt, gbh, gbt


def highway(c_prev, x):
    #TODO math
    """Highway Network units as an activation function.

    It accepts three or five arguments: an input minibatch ``x``, a weight
    matrix of carry gate ``Wh``, and a weight matrix of transform gate ``Wt``.
    A bias vector of carry gate ``bh`` and a bias vector of carry gate ``bt``
    is optional.

    It computes
    :math:`Y = xW^\\top + b`.

    Args:
        x (~chainer.Variable): Input variable. Its first dimension is assumed
            to be the *minibatch dimension*. The other dimensions are treated
            as concatenated one dimension whose size must be ``N``.
        Wh (~chainer.Variable): Weight variable of shape ``(M, N)``.
        Wt (~chainer.Variable): Weight variable of shape ``(M, N)``.
        bh (~chainer.Variable): Bias variable (optional) of shape ``(M,)``.
        bt (~chainer.Variable): Bias variable (optional) of shape ``(M,)``.

    Returns:
        ~chainer.Variable: Output variable.

    .. seealso:: :class:`~chainer.links.Highway`
    """
    return Highway()(x, Wh, Wt, bh, bt)

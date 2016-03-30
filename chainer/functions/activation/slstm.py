import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _extract_gates(x):
    r = x.reshape((x.shape[0], x.shape[1] // 4, 4) + x.shape[2:])
    return (r[:, :, i] for i in six.moves.range(4))


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

#define COMMON_ROUTINE \
    T aa1 = tanh(a1); \
    T ai1 = sigmoid(i1); \
    T af1 = sigmoid(f1); \
    T aa2 = tanh(a2); \
    T ai2 = sigmoid(i2); \
    T af2 = sigmoid(f2); \
    T ao = sigmoid(o1 + o2);
'''


class SLSTM(function.Function):

    """S-LSTM unit.

    It has four inputs (c1, c2, x1, x2) and two outputs (c, h), where c
    indicates the cell state. x1 and x2 must have four times channels compared
    to the number of units.

    """
    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 4)
        c1_type, c2_type, x1_type, x2_type = in_types

        type_check.expect(
            c1_type.dtype == numpy.float32,
            c2_type.dtype == numpy.float32,
            x1_type.dtype == numpy.float32,
            x2_type.dtype == numpy.float32,

            c1_type.ndim >= 2,
            c2_type.ndim >= 2,
            x1_type.ndim >= 2,
            x2_type.ndim >= 2,
            c1_type.ndim == x1_type.ndim,
            c1_type.ndim == c2_type.ndim,
            c1_type.ndim == x2_type.ndim,

            c1_type.shape[0] == x1_type.shape[0],
            c1_type.shape[0] == c2_type.shape[0],
            c1_type.shape[0] == x2_type.shape[0],
            x1_type.shape[1] == 4 * c1_type.shape[1],
            x2_type.shape[1] == 4 * c2_type.shape[1],
        )
        for i in range(2, c1_type.ndim.eval()):
            type_check.expect(x1_type.shape[i] == c1_type.shape[i])
            type_check.expect(x2_type.shape[i] == c2_type.shape[i])
            type_check.expect(x1_type.shape[i] == x2_type.shape[i])

    def forward(self, inputs):
        c_prev1, c_prev2, x1, x2 = inputs
        a1, i1, f1, o1 = _extract_gates(x1)
        a2, i2, f2, o2 = _extract_gates(x2)

        if isinstance(x1, numpy.ndarray):
            self.a1 = numpy.tanh(a1)
            self.i1 = _sigmoid(i1)
            self.f1 = _sigmoid(f1)

            self.a2 = numpy.tanh(a2)
            self.i2 = _sigmoid(i2)
            self.f2 = _sigmoid(f2)

            self.o = _sigmoid(o1 + o2)
            self.c = self.a1 * self.i1 + self.a2 * self.i2 + \
                self.f1 * c_prev1 + self.f2 * c_prev2

            h = self.o * numpy.tanh(self.c)
        else:
            self.c, h = cuda.elementwise(
                '''T c_prev1, T a1, T i1, T f1, T o1,
                   T c_prev2, T a2, T i2, T f2, T o2''',
                'T c, T h',
                '''
                    COMMON_ROUTINE;
                    c = aa1 * ai1 + af1 * c_prev1 + aa2 * ai2 + af2 * c_prev2;
                    h = ao * tanh(c);
                ''',
                'slstm_fwd', preamble=_preamble)(
                    c_prev1, a1, i1, f1, o1, c_prev2, a2, i2, f2, o2)

        return self.c, h

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        c_prev1, c_prev2, x1, x2 = inputs
        gc, gh = grad_outputs

        gx1 = xp.empty_like(x1)
        gx2 = xp.empty_like(x2)
        ga1, gi1, gf1, go1 = _extract_gates(gx1)
        ga2, gi2, gf2, go2 = _extract_gates(gx2)

        # Consider the case that either gradient is not given
        if gc is None:
            gc = 0
        if gh is None:
            gh = 0

        if xp is numpy:
            co = numpy.tanh(self.c)
            # multiply f later
            gc_prev = gh * self.o * _grad_tanh(co) + gc
            ga1[:] = gc_prev * self.i1 * _grad_tanh(self.a1)
            gi1[:] = gc_prev * self.a1 * _grad_sigmoid(self.i1)
            gf1[:] = gc_prev * c_prev1 * _grad_sigmoid(self.f1)
            go1[:] = gh * co * _grad_sigmoid(self.o)
            ga2[:] = gc_prev * self.i2 * _grad_tanh(self.a2)
            gi2[:] = gc_prev * self.a2 * _grad_sigmoid(self.i2)
            gf2[:] = gc_prev * c_prev2 * _grad_sigmoid(self.f2)
            go2[:] = gh * co * _grad_sigmoid(self.o)
            # multiply f here
            gc_prev1 = gc_prev * self.f1
            gc_prev2 = gc_prev * self.f2
        else:
            a1, i1, f1, o1 = _extract_gates(x1)
            a2, i2, f2, o2 = _extract_gates(x2)
            gc_prev1 = xp.empty_like(c_prev1)
            gc_prev2 = xp.empty_like(c_prev2)
            cuda.elementwise(
                '''T c_prev1, T a1, T i1, T f1, T o1,
                T c_prev2, T a2, T i2, T f2, T o2,
                T c, T gc, T gh''',
                '''T gc_prev1, T ga1, T gi1, T gf1, T go1,
                T gc_prev2, T ga2, T gi2, T gf2, T go2''',
                '''
                    COMMON_ROUTINE;
                    T co = tanh(c);
                    T temp = gh * ao * grad_tanh(co) + gc;
                    ga1 = temp * ai1 * grad_tanh(aa1);
                    gi1 = temp * aa1 * grad_sigmoid(ai1);
                    gf1 = temp * c_prev1 * grad_sigmoid(af1);
                    go1 = gh * co * grad_sigmoid(ao);
                    gc_prev1 = temp * af1;
                    ga2 = temp * ai2 * grad_tanh(aa2);
                    gi2 = temp * aa2 * grad_sigmoid(ai2);
                    gf2 = temp * c_prev2 * grad_sigmoid(af2);
                    go2 = gh * co * grad_sigmoid(ao);
                    gc_prev2 = temp * af2;
                ''',
                'lstm_bwd', preamble=_preamble)(
                    c_prev1, a1, i1, f1, o1,
                    c_prev2, a2, i2, f2, o2,
                    self.c, gc, gh,
                    gc_prev1, ga1, gi1, gf1, go1,
                    gc_prev2, ga2, gi2, gf2, go2)

        return gc_prev1, gc_prev2, gx1, gx2


def slstm(c_prev1, c_prev2, x1, x2):

    """S-LSTM units as an activation function.

    This function implements S-LSTM unit. It is an extension of LSTM unit
    applied to tree structures.
    The function is applied to binary trees. Each node has two child nodes.
    It gets four arguments, previous cell states :math:`c_1` and
    :math:`c_2`, and incoming signals :math:`x_1` and :math:`x_2`.

    First both input signals :math:`x_1` and :math:`x_2` are split into
    eight arrays :math:`a_1, i_1, f_1, o_1`, and :math:`a_2, i_2, f_2, o_2`.
    They have the same shape along the second axis.
    It means that :math:`x_1` and :math:`x_2` 's second axis must have 4 times
    the length of :math:`c_{1 \\text{prev}}` and :math:`c_{2 \\text{prev}}`.

    The split input signals are corresponding to:

        - :math:`a_i` : sources of cell input
        - :math:`i_i` : sources of input gate
        - :math:`f_i` : sources of forget gate
        - :math:`o_i` : sources of output gate

    It computes outputs as:

    .. math::

        c &= \\tanh(a_1 + a_2) \\sigma(i_1 + i_2)
           + c_{1 \\text{prev}} \\sigma(f_1)
           + c_{2 \\text{prev}} \\sigma(f_2), \\\\
        h &= \\tanh(c) \\sigma(o_1 + o_2),

    where :math:`\\sigma` is the elementwise sigmoid function.
    The function returns :math:`c` and :math:`h` as a tuple.

    Args:
        c_prev1 (~chainer.Variable): Variable that holds the previous cell
            state of the first child node. The cell state should be a zero
            array or the output of the  previous call of LSTM.
        c_prev2 (~chainer.Variable): Variable that holds the previous cell
            state of the second child node.
        x1 (~chainer.Variable): Variable that holds the incoming signal from
            the first child node. It must have the second dimension four times
            of that of the cell state,
        x2 (~chainer.Variable): Variable that holds the incoming signal from
            the second child node.

    Returns:
        tuple: Two :class:`~chainer.Variable` objects ``c`` and ``h``. ``c`` is
            the cell state. ``h`` indicates the outgoing signal.

    See detail in paper: `Long Short-Term Memory Over Tree Structures \
    <http://arxiv.org/abs/1503.04881>`_.

    """
    return SLSTM()(c_prev1, c_prev2, x1, x2)

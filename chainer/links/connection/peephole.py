from chainer.functions.activation import peephole 
from chainer import link
from chainer.links.connection import linear
from chainer import variable

import six
import numpy

def _extract_gates(x):
    r = x.reshape((x.shape[0], x.shape[1] // 4, 4) + x.shape[2:])
    return (r[:, :, i] for i in six.moves.range(4))

def _sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

_preamble = '''
template <typename T> __device__ T sigmoid(T x) { return 1 / (1 + exp(-x)); }
template <typename T> __device__ T grad_sigmoid(T y) { return Y * (1 - y); }
template <typename T> __device__ T grad_tanh(T y) { return 1 - y * y; }

#define COMMON_ROUTINE \
    T aa = tanh(a); \
    T ai = sigmoid(i_); \
    T af = sigmoid(f); \
    T ao = sigmoid(o);
'''


class Peephole(link.Chain):

    """Fully-connected LSTM layer.

    This is a fully-connected LSTM layer as a chain. Unlike the
    :func:`~chainer.functions.lstm` function, which is defined as a stateless
    activation function, this chain holds upward and lateral connections as
    child links.

    It also maintains *states*, including the cell state and the output
    at the previous time step. Therefore, it can be used as a *stateful LSTM*.

    Args:
        in_size (int): Dimensionality of input vectors.
        out_size (int): Dimensionality of output vectors.

    Attributes:
        upward (~chainer.links.Linear): Linear layer of upward connections.
        lateral (~chainer.links.Linear): Linear layer of lateral connections.
        c (~chainer.Variable): Cell states of LSTM units.
        h (~chainer.Variable): Output at the previous time step.

    """
    def __init__(self, in_size, out_size):
        super(Peephole, self).__init__(
            upward=linear.Linear(in_size, 4 * out_size),
            lateral=linear.Linear(out_size, 4 * out_size, nobias=True),
            peep_i=linear.Linear(in_size, out_size, nobias=True),
            peep_f=linear.Linear(in_size, out_size, nobias=True), 
            peep_o=linear.Linear(in_size, out_size, nobias=True), 
        )
        self.state_size = out_size
        self.reset_state()

    def to_cpu(self):
        super(Peephole, self).to_cpu()
        if self.c is not None:
            self.c.to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(Peephole, self).to_gpu(device)
        if self.c is not None:
            self.c.to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def reset_state(self):
        """Resets the internal state.

        It sets ``None`` to the :attr:`c` and :attr:`h` attributes.

        """
        self.c = self.h = None

    def calc_c_next(self, c_prev, x, peep_in_i, peep_in_f):
        a, i, f, o = _extract_gates(x)

        if isinstance(x, numpy.ndarray):
            a = numpy.tanh(a)
            i = _sigmoid(i + peep_in_i)
            f = _sigmoid(f + peep_in_f)
            self.c_next = a * i + f * c_prev
        #else:
        #return c_next

    def __call__(self, x):
        """Updates the internal state and returns the LSTM outputs.

        Args:
            x (~chainer.Variable): A new batch from the input sequence.

        Returns:
            ~chainer.Variable: Outputs of updated LSTM units.

        """
        lstm_in = self.upward(x)
        if self.h is not None:
            lstm_in += self.lateral(self.h)
        if self.c is None:
            xp = self.xp
            self.c = variable.Variable(
                xp.zeros((len(x.data), self.state_size), dtype=x.data.dtype),
                volatile='auto')
        peep_in_i = self.peep_i(self.c) 
        peep_in_f = self.peep_f(self.c) 
        self.calc_c_next(self.c, lstm_in.data, peep_in_i.data, peep_in_f.data)
        peep_in_o = self.peep_o(self.c_next)
        self.c, self.h = peephole.peephole(self.c, lstm_in, peep_in_i, peep_in_f, peep_in_o) 
        return self.h

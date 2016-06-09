from chainer.functions.activation import sigmoid
from chainer.functions.activation import tanh
from chainer.functions.array import reshape
from chainer.functions.array import split_axis
from chainer import link
from chainer.links.connection import linear
from chainer import variable


class Peephole(link.Chain):

    """Fully-connected LSTM with peephole connections layer.

    This is a fully-connected LSTM layer with peephole connections as a chain.
    Unlike the :link:`~chainer.links.lstm` link, this chain holds peep_i,
    peep_f and peep_o as child links besides upward and lateral.

    Given two inputs a previous hidden vector :math:`h` and an input vector
    :math:`x`, Peephole returns the next hidden vector :math:`h` defined as

    Args:
        in_size (int): Dimensionality of input vectors.
        out_size (int): Dimensionality of output vectors.

    Attributes:
        c (~chainer.Variable): Cell states of LSTM units.
        h (~chainer.Variable): Output at the previous time step.

    """
    def __init__(self, in_size, out_size):
        super(Peephole, self).__init__(
            upward=linear.Linear(in_size, 4 * out_size),
            lateral=linear.Linear(out_size, 4 * out_size, nobias=True),
            peep_i=linear.Linear(out_size, out_size, nobias=True),
            peep_f=linear.Linear(out_size, out_size, nobias=True),
            peep_o=linear.Linear(out_size, out_size, nobias=True),
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
        lstm_in = reshape.reshape(lstm_in, (lstm_in.data.shape[0],
                                            lstm_in.data.shape[1] // 4,
                                            4) + lstm_in.data.shape[2:])
        a, i, f, o = split_axis.split_axis(lstm_in, 4, 2)
        a = reshape.reshape(a, (a.data.shape[0], a.data.shape[1]))
        i = reshape.reshape(i, (i.data.shape[0], i.data.shape[1]))
        f = reshape.reshape(f, (f.data.shape[0], f.data.shape[1]))
        o = reshape.reshape(o, (o.data.shape[0], o.data.shape[1]))
        peep_in_i = self.peep_i(self.c)
        peep_in_f = self.peep_f(self.c)
        a = tanh.tanh(a)
        i = sigmoid.sigmoid(i + peep_in_i)
        f = sigmoid.sigmoid(f + peep_in_f)
        self.c = a * i + f * self.c
        peep_in_o = self.peep_o(self.c)
        o = sigmoid.sigmoid(o + peep_in_o)
        self.h = o * tanh.tanh(self.c)
        return self.h

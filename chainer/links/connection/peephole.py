from chainer import cuda
from chainer.functions.activation import sigmoid
from chainer.functions.activation import tanh
from chainer.functions.array import reshape
from chainer.functions.array import split_axis
from chainer import link
from chainer.links.connection import linear
from chainer import variable


class StatefulPeepholeLSTM(link.Chain):

    """Fully-connected LSTM layer with peephole connections.

    This is a fully-connected LSTM layer with peephole connections as a chain.
    Unlike the :class:`~chainer.links.LSTM` link, this chain holds ``peep_i``,
    ``peep_f`` and ``peep_o`` as child links besides ``upward`` and
    ``lateral``.

    Given a input vector :math:`x`, Peephole returns the next hidden vector
    :math:`h'` defined as

    .. math::

       a &=& \\tanh(upward x + lateral h), \\\\
       i &=& \\sigma(upward x + lateral h + peep_i c), \\\\
       f &=& \\sigma(upward x + lateral h + peep_f c), \\\\
       c' &=& a \\odot i + f \\odot c, \\\\
       o &=& \\sigma(upward x + lateral h + peep_o c'), \\\\
       h' &=& o \\tanh(c'),

    where :math:`\\sigma` is the sigmoid function, :math:`\\odot` is the
    element-wise product, :math:`c` is the current cell state, :math:`c'`
    is the next cell state and :math:`h` is the current hidden vector.

    Args:
        in_size(int): Dimension of the input vector :math:`x`.
        out_size(int): Dimension of the hidden vector :math:`h`.

    Attributes:
        upward (~chainer.links.Linear): Linear layer of upward connections.
        lateral (~chainer.links.Linear): Linear layer of lateral connections.
        peep_i (~chainer.links.Linear): Linear layer of peephole connections
                                        to the input gate.
        peep_f (~chainer.links.Linear): Linear layer of peephole connections
                                        to the forget gate.
        peep_o (~chainer.links.Linear): Linear layer of peephole connections
                                        to the output gate.
        c (~chainer.Variable): Cell states of LSTM units.
        h (~chainer.Variable): Output at the current time step.

    """

    def __init__(self, in_size, out_size):
        super(StatefulPeepholeLSTM, self).__init__(
            upward=linear.Linear(in_size, 4 * out_size),
            lateral=linear.Linear(out_size, 4 * out_size, nobias=True),
            peep_i=linear.Linear(out_size, out_size, nobias=True),
            peep_f=linear.Linear(out_size, out_size, nobias=True),
            peep_o=linear.Linear(out_size, out_size, nobias=True),
        )
        self.state_size = out_size
        self.reset_state()

    def to_cpu(self):
        super(StatefulPeepholeLSTM, self).to_cpu()
        if self.c is not None:
            self.c.to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(StatefulPeepholeLSTM, self).to_gpu(device)
        if self.c is not None:
            self.c.to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def reset_state(self):
        """Resets the internal states.

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
            with cuda.get_device(self._device_id):
                self.c = variable.Variable(
                    xp.zeros((x.shape[0], self.state_size), dtype=x.dtype),
                    volatile='auto')
        lstm_in = reshape.reshape(lstm_in, (len(lstm_in.data),
                                            lstm_in.shape[1] // 4,
                                            4))
        a, i, f, o = split_axis.split_axis(lstm_in, 4, 2)
        a = reshape.reshape(a, (len(a.data), a.shape[1]))
        i = reshape.reshape(i, (len(i.data), i.shape[1]))
        f = reshape.reshape(f, (len(f.data), f.shape[1]))
        o = reshape.reshape(o, (len(o.data), o.shape[1]))
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

from chainer.functions.activation import lstm
from chainer import initializations
from chainer import link
from chainer.links.connection import linear
from chainer import variable


class LSTM(link.Chain):

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

    def __init__(self, in_size, out_size,
                 lateral_init=initializations.orthogonal, upward_init=None,
                 bias_init=0, forget_bias_init=1):
        super(LSTM, self).__init__(
            upward=linear.Linear(in_size, 4 * out_size, initialW=0),
            lateral=linear.Linear(out_size, 4 * out_size,
                                  initialW=0, nobias=True),
        )
        self.state_size = out_size
        self.reset_state()

        for i in range(0, 4 * out_size, out_size):
            initializations.init_weight(
                self.lateral.W.data[i:i + out_size, :], lateral_init)
            initializations.init_weight(
                self.upward.W.data[i:i + out_size, :], upward_init)

        a, i, f, o = lstm._extract_gates(
            self.upward.b.data.reshape(1, 4 * out_size, 1))
        initializations.init_weight(a, bias_init)
        initializations.init_weight(i, bias_init)
        initializations.init_weight(f, forget_bias_init)
        initializations.init_weight(o, bias_init)

    def to_cpu(self):
        super(LSTM, self).to_cpu()
        if self.c is not None:
            self.c.to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(LSTM, self).to_gpu(device)
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
        self.c, self.h = lstm.lstm(self.c, lstm_in)
        return self.h

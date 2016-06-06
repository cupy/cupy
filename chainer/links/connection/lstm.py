import six

from chainer.functions.activation import lstm
from chainer.functions.array import concat
from chainer.functions.array import split_axis
from chainer import initializers
from chainer import link
from chainer.links.connection import linear
from chainer import variable


class LSTMBase(link.Chain):

    def __init__(self, in_size, out_size,
                 lateral_init=None, upward_init=None,
                 bias_init=0, forget_bias_init=0):
        super(LSTMBase, self).__init__(
            upward=linear.Linear(in_size, 4 * out_size, initialW=0),
            lateral=linear.Linear(out_size, 4 * out_size,
                                  initialW=0, nobias=True),
        )
        self.state_size = out_size

        for i in six.moves.range(0, 4 * out_size, out_size):
            initializers.init_weight(
                self.lateral.W.data[i:i + out_size, :], lateral_init)
            initializers.init_weight(
                self.upward.W.data[i:i + out_size, :], upward_init)

        a, i, f, o = lstm._extract_gates(
            self.upward.b.data.reshape(1, 4 * out_size, 1))
        initializers.init_weight(a, bias_init)
        initializers.init_weight(i, bias_init)
        initializers.init_weight(f, forget_bias_init)
        initializers.init_weight(o, bias_init)


class StatelessLSTM(LSTMBase):

    """Stateless LSTM layer.

    This is a fully-connected LSTM layer as a chain. Unlike the
    :func:`~chainer.functions.lstm` function, this chain holds upward and
    lateral connections as child links. This link doesn't keep cell and
    hidden states.

    Args:
        in_size (int): Dimensionality of input vectors.
        out_size (int): Dimensionality of output vectors.

    Attributes:
        upward (chainer.links.Linear): Linear layer of upward connections.
        lateral (chainer.links.Linear): Linear layer of lateral connections.

    """

    def __call__(self, c, h, x):
        """Returns new cell state and updated output of LSTM.

        Args:
            c (~chainer.Variable): Cell states of LSTM units.
            h (~chainer.Variable): Output at the previous timestep.
            x (~chainer.Variable): A new batch from the input sequence.

        Returns:
            tuple of ~chainer.Variable: Returns ``(c_new, h_new)``, where
                ``c_new`` represents new cell state, and ``h_new`` is updated
                output of LSTM units.

        """
        lstm_in = self.upward(x)
        if h is not None:
            lstm_in += self.lateral(h)
        if c is None:
            xp = self.xp
            c = variable.Variable(
                xp.zeros((len(x.data), self.state_size), dtype=x.data.dtype),
                volatile='auto')
        return lstm.lstm(c, lstm_in)


class LSTM(LSTMBase):

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
        lateral_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            It is used for initialization of the lateral connections.
            Maybe be ``None`` to use default initialization.
        upward_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            It is used for initialization of the upward connections.
            Maybe be ``None`` to use default initialization.
        bias_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value
            It is used for initialization of the biases of cell input,
            input gate and output gate.and gates of the upward connection.
            Maybe a scalar, in that case, the bias is
            initialized by this value.
            Maybe be ``None`` to use default initialization.
        forget_bias_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value
            It is used for initialization of the biases of the forget gate of
            the upward connection.
            Maybe a scalar, in that case, the bias is
            initialized by this value.
            Maybe be ``None`` to use default initialization.


    Attributes:
        upward (~chainer.links.Linear): Linear layer of upward connections.
        lateral (~chainer.links.Linear): Linear layer of lateral connections.
        c (~chainer.Variable): Cell states of LSTM units.
        h (~chainer.Variable): Output at the previous time step.

    """

    def __init__(self, in_size, out_size, **kwargs):
        super(LSTM, self).__init__(in_size, out_size, **kwargs)
        self.reset_state()

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
        batch = len(x.data)
        lstm_in = self.upward(x)
        h_rest = None
        if self.h is not None:
            if len(self.h.data) > batch:
                h_update, h_rest = split_axis.split_axis(
                    self.h, [batch], axis=0)
                lstm_in += self.lateral(h_update)
            else:
                lstm_in += self.lateral(self.h)
        if self.c is None:
            xp = self.xp
            self.c = variable.Variable(
                xp.zeros((len(x.data), self.state_size), dtype=x.data.dtype),
                volatile='auto')
        self.c, h = lstm.lstm(self.c, lstm_in)
        if h_rest is not None:
            h = concat.concat([h, h_rest], axis=0)
        self.h = h
        return self.h

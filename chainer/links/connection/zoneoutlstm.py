import numpy

import chainer
from chainer import cuda
from chainer.functions.activation import sigmoid
from chainer.functions.activation import tanh
from chainer.functions.array import reshape
from chainer.functions.array import split_axis
from chainer.functions.noise import zoneout

from chainer import link
from chainer.links.connection import linear
from chainer import variable


class StatefulZoneoutLSTM(link.Chain):

    def __init__(self, in_size, out_size,
                 c_ratio=0.5, h_ratio=0.5, train=True):
        super(StatefulZoneoutLSTM, self).__init__(
            upward=linear.Linear(in_size, 4 * out_size),
            lateral=linear.Linear(out_size, 4 * out_size, nobias=True),
        )
        self.state_size = out_size
        self.c_ratio = c_ratio
        self.h_ratio = h_ratio
        self.train = train
        self.reset_state()

    def to_cpu(self):
        super(StatefulZoneoutLSTM, self).to_cpu()
        if self.c is not None:
            self.c.to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(StatefulZoneoutLSTM, self).to_gpu(device)
        if self.c is not None:
            self.c.to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def set_state(self, c, h):
        """Sets the internal state.

        It sets the :attr:`c` and :attr:`h` attributes.

        Args:
            c (~chainer.Variable): A new cell states of LSTM units.
            h (~chainer.Variable): A new output at the previous time step.

        """
        assert isinstance(c, chainer.Variable)
        assert isinstance(h, chainer.Variable)
        c_ = c
        h_ = h
        if self.xp is numpy:
            c_.to_cpu()
            h_.to_cpu()
        else:
            c_.to_gpu(self._device_id)
            h_.to_gpu(self._device_id)
        self.c = c_
        self.h = h_

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
        else:
            xp = self.xp
            with cuda.get_device(self._device_id):
                self.h = variable.Variable(
                    xp.zeros((len(x.data), self.state_size),
                             dtype=x.data.dtype),
                    volatile='auto')
        if self.c is None:
            xp = self.xp
            with cuda.get_device(self._device_id):
                self.c = variable.Variable(
                    xp.zeros((len(x.data), self.state_size),
                             dtype=x.data.dtype),
                    volatile='auto')

        lstm_in = reshape.reshape(lstm_in, (len(lstm_in.data),
                                            lstm_in.data.shape[1] // 4,
                                            4))

        a, i, f, o = split_axis.split_axis(lstm_in, 4, 2)
        a = reshape.reshape(a, (len(a.data), self.state_size))
        i = reshape.reshape(i, (len(i.data), self.state_size))
        f = reshape.reshape(f, (len(f.data), self.state_size))
        o = reshape.reshape(o, (len(o.data), self.state_size))

        c_tmp = tanh.tanh(a) * sigmoid.sigmoid(i) + sigmoid.sigmoid(f) * self.c
        self.c = zoneout.zoneout(self.c, c_tmp, self.c_ratio, self.train)
        self.h = zoneout.zoneout(self.h,
                                 sigmoid.sigmoid(o) * tanh.tanh(c_tmp),
                                 self.h_ratio, self.train)
        return self.h

import numpy

import chainer
from chainer.functions.activation import sigmoid
from chainer.functions.activation import tanh
from chainer import link
from chainer.links.connection import linear


class GRU(link.Chain):

    """Gated Recurrent Unit function (GRU).

    GRU function has six parameters :math:`W_r`, :math:`W_z`, :math:`W`,
    :math:`U_r`, :math:`U_z`, and :math:`U`. All these parameters are
    :math:`n \\times n` matricies, where :math:`n` is the dimension of
    hidden vectors.

    Given two inputs a previous hidden vector :math:`h` and an input vector
    :math:`x`, GRU returns the next hidden vector :math:`h'` defined as

    .. math::

       r &=& \\sigma(W_r x + U_r h), \\\\
       z &=& \\sigma(W_z x + U_z h), \\\\
       \\bar{h} &=& \\tanh(W x + U (r \\odot h)), \\\\
       h' &=& (1 - z) \\odot h + z \\odot \\bar{h},

    where :math:`\\sigma` is the sigmoid function, and :math:`\\odot` is the
    element-wise product.

    Args:
        n_units(int): Dimension of input vector :math:`x`, and hidden vector
            :math:`h`.

    See:
        - `On the Properties of Neural Machine Translation: Encoder-Decoder
          Approaches <http://www.aclweb.org/anthology/W14-4012>`_
          [Cho+, SSST2014].
        - `Empirical Evaluation of Gated Recurrent Neural Networks on Sequence
          Modeling <http://arxiv.org/abs/1412.3555>`_
          [Chung+NIPS2014 DLWorkshop].

    """

    def __init__(self, n_units):
        super(GRU, self).__init__(
            W_r=linear.Linear(n_units, n_units),
            U_r=linear.Linear(n_units, n_units),
            W_z=linear.Linear(n_units, n_units),
            U_z=linear.Linear(n_units, n_units),
            W=linear.Linear(n_units, n_units),
            U=linear.Linear(n_units, n_units),
        )
        self.state_size = n_units
        self.reset_state()

    def to_cpu(self):
        super(GRU, self).to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(GRU, self).to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def set_state(self, h):
        if self.xp == numpy:
            h = chainer.cuda.to_cpu(h)
        else:
            h = chainer.cuda.to_gpu(h)
        self.h = chainer.Variable(h)

    def reset_state(self):
        self.h = None

    def __call__(self, x):
        if self.h is None:
            self.h = chainer.Variable(
                self.xp.zeros((len(x.data), self.state_size),
                              dtype=x.data.dtype), volatile='auto')
        r = sigmoid.sigmoid(self.W_r(x) + self.U_r(self.h))
        z = sigmoid.sigmoid(self.W_z(x) + self.U_z(self.h))
        h_bar = tanh.tanh(self.W(x) + self.U(r * self.h))
        self.h = (1 - z) * self.h + z * h_bar
        return self.h

import numpy

import chainer
from chainer.functions.activation import sigmoid
from chainer.functions.activation import tanh
from chainer import link
from chainer.links.connection import linear


class GRUBase(link.Chain):

    def __init__(self, n_units):
        super(GRUBase, self).__init__(
            W_r=linear.Linear(n_units, n_units),
            U_r=linear.Linear(n_units, n_units),
            W_z=linear.Linear(n_units, n_units),
            U_z=linear.Linear(n_units, n_units),
            W=linear.Linear(n_units, n_units),
            U=linear.Linear(n_units, n_units),
        )


class GRU(GRUBase):

    """Stateless Gated Recurrent Unit function (GRU).

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

    :class:`~chainer.links.GRU` does not hold the value of
    hidden vector :math:`h`. So this is *stateless*.
    Use :class:`~chainer.links.StatefulGRU` as a *stateful* GRU.

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


    .. seealso:: :class:`~chainer.links.StatefulGRU`
    """

    def __call__(self, h, x):
        r = sigmoid.sigmoid(self.W_r(x) + self.U_r(h))
        z = sigmoid.sigmoid(self.W_z(x) + self.U_z(h))
        h_bar = tanh.tanh(self.W(x) + self.U(r * h))
        h_new = (1 - z) * h + z * h_bar
        return h_new


class StatefulGRU(GRUBase):
    """Stateful Gated Recurrent Unit function (GRU).

    Stateful GRU function has six parameters :math:`W_r`, :math:`W_z`,
    :math:`W`, :math:`U_r`, :math:`U_z`, and :math:`U`.
    All these parameters are :math:`n \\times n` matricies,
    where :math:`n` is the dimension of hidden vectors.

    Given input vector :math:`x`, Stateful GRU returns the next
    hidden vector :math:`h'` defined as

    .. math::

       r &=& \\sigma(W_r x + U_r h), \\\\
       z &=& \\sigma(W_z x + U_z h), \\\\
       \\bar{h} &=& \\tanh(W x + U (r \\odot h)), \\\\
       h' &=& (1 - z) \\odot h + z \\odot \\bar{h},

    where :math:`h` is current hidden vector.

    As the name indicates, :class:`~chainer.links.StatefulGRU` is *stateful*,
    meaning that it also holds the next hidden vector `h'` as a state.
    Use :class:`~chainer.links.GRU` as a stateless version of GRU.

    Args:
        n_units(int): Dimension of input vector :math:`x`, and hidden vector.

    Attributes:
        h(~chainer.Variable): Hidden vector that indicates the state of
        :class:`~chainer.links.StatefulGRU`.

    .. seealso:: :class:`~chainer.functions.GRU`
    """

    def __init__(self, n_units):
        super(StatefulGRU, self).__init__(n_units)
        self.state_size = n_units
        self.reset_state()

    def to_cpu(self):
        super(StatefulGRU, self).to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(StatefulGRU, self).to_gpu(device)
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
        z = self.W_z(x)
        if self.h is not None:
            r = sigmoid.sigmoid(self.W_r(x) + self.U_r(self.h))
            z += self.U_z(self.h)
        z = sigmoid.sigmoid(z)

        h_bar = self.W(x)
        if self.h is not None:
            h_bar += self.U(r * self.h)
        h_bar = tanh.tanh(h_bar)

        h_new = z * h_bar
        if self.h is not None:
            h_new += (1 - z) * self.h
        self.h = h_new
        return self.h

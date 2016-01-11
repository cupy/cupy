from chainer.functions.activation import sigmoid
from chainer.functions.activation import tanh
from chainer import link
from chainer import initializations
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

    def __init__(self, n_units, init=None, inner_init=initializations.orthogonal, bias_init=0):
        super(GRU, self).__init__(
            W_r=linear.Linear(n_units, n_units, initialW=0),
            U_r=linear.Linear(n_units, n_units, initialW=0),
            W_z=linear.Linear(n_units, n_units, initialW=0),
            U_z=linear.Linear(n_units, n_units, initialW=0),
            W=linear.Linear(n_units, n_units, initialW=0),
            U=linear.Linear(n_units, n_units, initialW=0),
        )

		#initialize mats that process raw input        
        for mat in (self.W_r, self.W_z, self.W):
        	initializations.init_weight(mat.W.data, init)
        #initialize mats that take in recurrences 
        for mat in (self.U_r, self.U_z, self.U):
        	initializations.init_weight(mat.W.data, inner_init)
        #initialize bias terms
        for mat in (self.W_r, self.W_z, self.W, self.U_r, self.U_z, self.U):
        	initializations.init_weight(mat.b.data, bias_init)

    def __call__(self, h, x):
        r = sigmoid.sigmoid(self.W_r(x) + self.U_r(h))
        z = sigmoid.sigmoid(self.W_z(x) + self.U_z(h))
        h_bar = tanh.tanh(self.W(x) + self.U(r * h))
        h_new = (1 - z) * h + z * h_bar
        return h_new

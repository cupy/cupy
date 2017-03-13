import numpy
import six

import chainer
from chainer import cuda
from chainer.functions.array import permutate
from chainer.functions.array import transpose_sequence
from chainer.functions.connection import n_step_lstm as rnn
from chainer import link


def argsort_list_descent(lst):
    return numpy.argsort([-len(x.data) for x in lst]).astype('i')


def permutate_list(lst, indices, inv):
    ret = [None] * len(lst)
    if inv:
        for i, ind in enumerate(indices):
            ret[ind] = lst[i]
    else:
        for i, ind in enumerate(indices):
            ret[i] = lst[ind]
    return ret


class NStepLSTM(link.ChainList):

    """Stacked LSTM for sequnces.

    This link is stacked version of LSTM for sequences. It calculates hidden
    and cell states of all layer at end-of-string, and all hidden states of
    the last layer for each time.

    Unlike :func:`chainer.functions.n_step_lstm`, this function automatically
    sort inputs in descending order by length, and transpose the seuqnece.
    Users just need to call the link with a list of :class:`chainer.Variable`
    holding sequences.

    Args:
        n_layers (int): Number of layers.
        in_size (int): Dimensionality of input vectors.
        out_size (int): Dimensionality of hidden states and output vectors.
        dropout (float): Dropout ratio.
        use_cudnn (bool): Use cuDNN.

    .. seealso::
        :func:`chainer.functions.n_step_lstm`

    """

    def __init__(
            self, n_layers, in_size, out_size, dropout, use_cudnn=True):
        weights = []
        for i in six.moves.range(n_layers):
            weight = link.Link()
            for j in six.moves.range(8):
                if i == 0 and j < 4:
                    w_in = in_size
                else:
                    w_in = out_size
                weight.add_param('w%d' % j, (out_size, w_in))
                weight.add_param('b%d' % j, (out_size,))
                getattr(weight, 'w%d' % j).data[...] = numpy.random.normal(
                    0, numpy.sqrt(1. / w_in), (out_size, w_in))
                getattr(weight, 'b%d' % j).data[...] = 0
            weights.append(weight)

        super(NStepLSTM, self).__init__(*weights)

        self.n_layers = n_layers
        self.dropout = dropout
        self.use_cudnn = use_cudnn
        self.out_size = out_size

    def __call__(self, hx, cx, xs, train=True):
        """Calculate all hidden states and cell states.

        Args:
            hx (~chainer.Variable or None): Initial hidden states. If ``None``
                is specified zero-vector is used.
            cx (~chainer.Variable or None): Initial cell states. If ``None``
                is specified zero-vector is used.
            xs (list of ~chianer.Variable): List of input sequences.
                Each element ``xs[i]`` is a :class:`chainer.Variable` holding
                a sequence.
        """
        assert isinstance(xs, (list, tuple))
        indices = argsort_list_descent(xs)

        xs = permutate_list(xs, indices, inv=False)
        if hx is None:
            with cuda.get_device(self._device_id):
                hx = chainer.Variable(
                    self.xp.zeros(
                        (self.n_layers, len(xs), self.out_size),
                        dtype=xs[0].dtype),
                    volatile='auto')
        else:
            hx = permutate.permutate(hx, indices, axis=1, inv=False)

        if cx is None:
            with cuda.get_device(self._device_id):
                cx = chainer.Variable(
                    self.xp.zeros(
                        (self.n_layers, len(xs), self.out_size),
                        dtype=xs[0].dtype),
                    volatile='auto')
        else:
            cx = permutate.permutate(cx, indices, axis=1, inv=False)

        trans_x = transpose_sequence.transpose_sequence(xs)

        ws = [[w.w0, w.w1, w.w2, w.w3, w.w4, w.w5, w.w6, w.w7] for w in self]
        bs = [[w.b0, w.b1, w.b2, w.b3, w.b4, w.b5, w.b6, w.b7] for w in self]

        hy, cy, trans_y = rnn.n_step_lstm(
            self.n_layers, self.dropout, hx, cx, ws, bs, trans_x,
            train=train, use_cudnn=self.use_cudnn)

        hy = permutate.permutate(hy, indices, axis=1, inv=True)
        cy = permutate.permutate(cy, indices, axis=1, inv=True)
        ys = transpose_sequence.transpose_sequence(trans_y)
        ys = permutate_list(ys, indices, inv=True)

        return hy, cy, ys

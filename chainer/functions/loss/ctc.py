import numpy
import six

import chainer
from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check


def _logsumexp(a, xp, axis=None):
    vmax = xp.amax(a, axis=axis, keepdims=True)
    vmax += xp.log(xp.sum(xp.exp(a - vmax),
                          axis=axis, keepdims=True, dtype=a.dtype))
    return xp.squeeze(vmax, axis=axis)


def _softmax(x, xp):
    val = xp.exp(x - xp.amax(x, axis=1, keepdims=True))
    val /= xp.sum(val, axis=1, keepdims=True)
    return val


def _label_to_path(labels, blank_symbol, xp):
    path = xp.full((len(labels), labels.shape[1] * 2 + 1),
                   blank_symbol, dtype=numpy.int32)
    path[:, 1::2] = labels
    return path


def _log_dot(prob, rr, xp):
    return _logsumexp(prob + xp.swapaxes(rr, 1, 2), xp, axis=2)


def _activate(yseq, xp):
    return [_softmax(y, xp) for y in yseq]


def _move_label_to_front(path, path_length, xp):
    return (xp.take(path[:, ::-1],
                    (xp.arange(0, path.size, path.shape[1],
                               dtype=numpy.int32)[:, None]
                     + (xp.arange(path.size).reshape(path.shape)
                        - path_length[:, None])
                     % path.shape[1])))


def _move_label_to_back(path, path_length, xp):
    return (xp.take(path,
                    (xp.arange(0, path.size, path.shape[1],
                               dtype=numpy.int32)[:, None]
                     + (xp.arange(path.size).reshape(path.shape)
                        + path_length[:, None])[:, ::-1]
                     % path.shape[1])))


def _move_inputs(prob, input_length, xp):
    concatenated = xp.rollaxis(xp.dstack(prob), 2)
    rotate = (numpy.swapaxes((xp.arange(concatenated.shape[0]
                                        * concatenated.shape[1])
                              % concatenated.shape[0]).reshape(
                                  (concatenated.shape[1],
                                   concatenated.shape[0])),
                             1, 0)
              + input_length[None, :]) % concatenated.shape[0]
    index = concatenated.shape[2]\
        * (rotate * concatenated.shape[1]
           + xp.arange(0, concatenated.shape[1])[None, :])[:, :, None]\
        + xp.arange(concatenated.shape[2])[None, None, :]
    result = [xp.squeeze(a)
              for a in xp.vsplit(xp.take(concatenated, index), len(prob))]
    return result


class ConnectionistTemporalClassification(function.Function):

    """The implementation of Connectionist Temporal Classfication loss functions.

    To make it usable for real-world cases, this class has two policies below.
    1. This class computes forward and backward variables in the log domain.
    2. This class applies the softmax function to inputs. The Backward
    values of CTC loss is often overflows. This is avoided by computing
    backward values before the activation function is applied.
    """

    def __init__(self, blank_symbol):
        self.blank_symbol = blank_symbol
        self.zero_padding = -10000000000.0

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 3)
        l_type = in_types[2]
        type_check.expect(l_type.dtype == numpy.int32)

        x_basetype = in_types[3]

        for i in six.moves.range(3, len(in_types)):
            x_type = in_types[i]
            type_check.expect(
                x_type.dtype == numpy.float32,
                x_type.shape == x_basetype.shape,
            )

    def log_matrix(self, x, xp):
        if xp == numpy:
            res = numpy.ma.log(x).filled(fill_value=self.zero_padding)
        else:
            create_recurrence_relation = cuda.cupy.ElementwiseKernel(
                'T x, T e', 'T y',
                'y = x == 0 ? e : log(x)',
                'create_recurrence_relation')
            res = create_recurrence_relation(x, self.zero_padding)
        return res.astype(numpy.float32)

    def recurrence_relation(self, path_length, max_length, dtype, xp):
        """Transition in forword and backword algorithms is represented as matrix.

        See also
        https://blog.wtf.sg/2014/10/06/connectionist-temporal-classification-ctc-with-theano/
        """
        rr = (xp.eye(max_length, dtype=dtype) +
              xp.eye(max_length, k=1, dtype=dtype) +
              xp.eye(max_length, k=2, dtype=dtype) *
              (xp.arange(max_length, dtype=dtype) % dtype(2)))
        return self.log_matrix(rr[None, :]
                               * xp.greater(
                                   (path_length[:, None]
                                    - xp.arange(max_length)[None, :]),
                                   0).astype(numpy.int32)[:, :, None], xp)

    # path probablity to label probability
    def label_probability(self, label_size, path, path_length, multiply, xp):
        labels_prob = self.log_matrix(xp.zeros((len(path), label_size),
                                               dtype=multiply.dtype), xp)
        if xp == numpy:
            for b in six.moves.range(len(path)):
                target_path = path[b][0:path_length[b]]
                chars = {c for c in target_path}
                for c in chars:
                    labels_prob[b, c] = _logsumexp(
                        multiply[b, 0:path_length[b]][target_path == c], numpy)
        else:
            cuda.cupy.ElementwiseKernel(
                'raw T x, raw I y, raw I l, I b_max, I c_max',
                'T z',
                '''
                T value = z;
                I c = i % b_max, b = i / b_max;
                int ind[2] = {b, -1};
                for (int index = 0; index < c_max; ++index) {
                    ind[1] = index;
                    if (ind[1] < l[ind[0]] && y[ind] == c) {
                        T xvalue = x[ind];
                        if (value > xvalue) {
                            value = value + log(1 + exp(xvalue - value));
                        } else {
                            value = xvalue + log(1 + exp(value - xvalue));
                        }
                    }
                    z = value;
                }
                ''',
                'reduce_probability')(multiply, path, path_length,
                                      labels_prob.shape[1],
                                      path.shape[1], labels_prob)
        return labels_prob

    def calc_trans(self, path, yseq, xp):
        forward_prob = self.log_matrix(
            xp.eye(path.shape[1], dtype='f')[0], xp)[None, :]
        backward_prob = forward_prob
        offset = xp.arange(
            0, yseq[0].size, yseq[0].shape[1], dtype=path.dtype)[:, None]

        # prob[i] := forward[i] + backward[-i-1]
        prob = []
        index = offset + path
        frr = self.recurrence_relation(
            self.path_length, path.shape[1], numpy.float32, xp)
        # forward computation.
        for y in yseq:
            # calc forward probability in log scale
            forward_prob = xp.take(y, index) + _log_dot(
                forward_prob[:, None, :], frr, xp)
            prob.append(forward_prob)
        r_index = offset \
            + _move_label_to_back(path, self.path_length, xp)

        # rotate yseq with path_length
        yseq_inv = _move_inputs(yseq, self.input_length, xp)[::-1]
        brr = self.recurrence_relation(
            self.path_length, path.shape[1], numpy.float32, xp)

        # move to back.
        prob = _move_inputs(prob, self.input_length, xp)
        # backward computation.
        for i, y_inv in enumerate(yseq_inv):
            # calc backward probability
            backward_prob = _log_dot(backward_prob[:, None, :], brr, xp)
            prob[-i - 1] += _move_label_to_front(backward_prob,
                                                 self.path_length, xp)
            backward_prob = xp.take(y_inv, r_index) + backward_prob

        # move to front.
        prob = _move_inputs(prob, -self.input_length, xp)
        return prob

    def forward(self, inputs):
        xp = cuda.get_array_module(inputs[0])
        self.input_length = inputs[0]

        # The length of path is (2 * label_length + 1)
        self.path_length = 2 * inputs[1] + 1

        batch_size = len(inputs[2])
        self.yseq = _activate(inputs[3::], xp)
        log_yseq = [self.log_matrix(y, xp) for y in self.yseq]
        self.path = _label_to_path(inputs[2], self.blank_symbol, xp)
        self.prob_trans = self.calc_trans(self.path, log_yseq, xp)

        loss = utils.force_array(xp.sum(
            _logsumexp(self.prob_trans[0], xp, axis=1)))
        loss /= -batch_size
        return loss,

    def backward(self, inputs, grad_output):
        xp = cuda.get_array_module(inputs[0])
        batch_size = len(inputs[2])

        total_probability = _logsumexp(self.prob_trans[0], xp, axis=1)
        scale = grad_output[0] / batch_size
        for i, (y, prob) in enumerate(zip(self.yseq, self.prob_trans)):
            label_prob = self.label_probability(
                y.shape[1], self.path, self.path_length, prob, xp)
            y -= xp.exp(label_prob - total_probability[:, None])
            y *= scale
            # mask
            y *= xp.less((i - self.input_length),
                         0).astype(numpy.int32)[:, None]
        return (None, None, None) + tuple(self.yseq)


def connectionist_temporal_classification(x, t, blank_symbol,
                                          input_length=None,
                                          label_length=None):
    """Connectionist Temporal Classification loss function.

    Connectionist Temporal Classification(CTC) [Graves2006]_ is a loss function
    of sequence labeling where the alignment between the inputs and target is
    unknown. See also [Graves2012]_

    Args:
        x (Variable): RNN output at each time.
                      (ex. :math:`(y_1, y_2, ..., y_T)`)
        t (Variable): Expected label sequence.
        blank_symbol (int): Index of blank_symbol.
                            This value must be non-negative.
        input_length (Variable): Length of valid sequence for each of
                                 mini batch x (optional). If input_length
                                 is skipped, It regards that all of
                                 x is valid input.
        label_length (Variable): Length of valid sequence for each of
                                 mini batch t (optional). If label_length
                                 is skipped, It regards that all of
                                 t is valid input.

    Returns:
        Variable: A variable holding a scalar value of the CTC loss.

    .. note::
       You need to input ``x`` without applying to activation functions(e.g.
       softmax function), because this function applies softmax functions
       to ``x`` before calculating CTC loss to avoid numerical limitations.
       You also need to apply softmax function to fowarded values before you
       decode it.

    .. note::
       This function is differentiable only by ``x``.

    .. note::
       This function supports (batch, sequence, 1-dimensional input)-data.

    .. [Graves2006] Alex Graves, Santiago Fernandez,\
    Faustino Gomez, Jurgen Schmidhuber,\
    `Connectionist Temporal Classification: Labelling Unsegmented\
    Sequence Data with Recurrent Neural Networks\
    <ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf>`_

    .. [Graves2012] Alex Graves,\
    `Supervised Sequence Labelling with Recurrent Neural Networks\
    <http://www.cs.toronto.edu/~graves/preprint.pdf>`_

    """
    if not isinstance(blank_symbol, int):
        raise TypeError('blank_symbol must be non-negative integer.')
    assert blank_symbol >= 0
    assert blank_symbol < x[0].data.shape[1]
    # This implementation only supports 1-dimensional data.
    # TODO(jnishi): Support d(>1)-dimentinal inputs.
    assert(len(x[0].data.shape) == 2)

    if input_length is None:
        xp = cuda.get_array_module(x[0].data)
        input_length = chainer.Variable(xp.full((len(x[0].data),),
                                                len(x[0].data[0]),
                                                dtype=int))
        label_length = chainer.Variable(xp.full((len(t.data),),
                                                len(t.data[0]),
                                                dtype=int))

    # Batch size check.
    assert len(x[0].data) == len(t.data)
    assert len(x[0].data) == len(input_length.data)
    assert len(x[0].data) == len(label_length.data)

    # Length check.
    assert len(x) >= max(input_length.data)
    assert len(t.data[0]) >= max(label_length.data)

    return ConnectionistTemporalClassification(blank_symbol)(input_length,
                                                             label_length,
                                                             t, *x)

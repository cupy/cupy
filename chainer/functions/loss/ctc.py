from chainer import cuda
from chainer import function
from chainer import utils
import numpy
import six


def logsumexp(a, xp, axis=None):
    vmax = xp.amax(a, axis=axis, keepdims=True)
    res = xp.log(xp.sum(xp.exp(a - vmax), axis=axis, keepdims=False))
    vmax = xp.squeeze(vmax, axis=axis)
    return (res + vmax).astype(numpy.float32)


def softmax(x, xp, axis=1):
    val = x - xp.amax(x, axis=axis, keepdims=True)
    val = xp.exp(val)
    return val / xp.sum(val, axis=axis, keepdims=True)


class ConnectionistTemporalClassification(function.Function):

    '''To make it usable for real-world cases, this class has two policies below.

    1. This class computes forward and backward variables in the log domain.
    2. This class applies the softmax function to inputs. The Backward
    values of CTC loss is often overflows. This is avoided by computing
    backward values before the activation function is applied.
    '''

    def __init__(self, blank_symbol):
        self.blank_symbol = blank_symbol
        self.zero_padding = -10000000000.0

    def check_type_forward(self, in_types):
        utils.type_check.expect(in_types.size() > 1)
        l_type = in_types[0]
        utils.type_check.expect(l_type.dtype == numpy.int32)

        x_basetype = in_types[1]

        for i in six.moves.range(2, len(in_types)):
            x_type = in_types[1]
            utils.type_check.expect(
                x_type.dtype == numpy.float32,
                x_type.shape == x_basetype.shape,
            )

    def log_matrix(self, x, xp):
        if xp == numpy:
            res = numpy.ma.log(x).filled(fill_value=self.zero_padding)
        else:
            create_recurrence_relation = cuda.cupy.ElementwiseKernel(
                'T x, T e', 'T y',
                '''
                if(x == 0){
                      y = e;
                }else{
                      y = log(x);
                }
                ''',
                'create_recurrence_relation')
            res = create_recurrence_relation(x, self.zero_padding)
        return res

    '''
    Transition in forword and backword algorithms is represented as matrix.
    See also
    https://blog.wtf.sg/2014/10/06/connectionist-temporal-classification-ctc-with-theano/
    '''
    def recurrence_relation(self, size, xp):
        rr = (xp.eye(size) + xp.eye(size, k=1) +
              xp.eye(size, k=2) * (xp.arange(size) % 2))
        return self.log_matrix(rr, xp)

    def label_to_path(self, labels, xp):
        label_length = labels.shape[0]
        path = xp.full((label_length * 2 + 1,),
                       self.blank_symbol, dtype=int)
        for i in six.moves.range(label_length):
            path[i * 2 + 1] = labels[i]
        return path

    def log_dot(self, prob, rr, xp):
        rtrans = xp.swapaxes(rr, 1, 0)
        return logsumexp(prob + rtrans, xp, axis=1)

    # path probablity to label probability
    def label_probability(self, label_size, path, multiply, xp):
        labels_prob = self.log_matrix(xp.zeros((label_size,),
                                               dtype=numpy.float32), xp)
        if xp == numpy:
            chars = {c for c in path}
            for c in chars:
                pos = numpy.where(path == c)[0]
                labels_prob[c] = logsumexp(xp.take(multiply, pos), xp)
        else:
            cuda.cupy.ElementwiseKernel(
                'raw float32 x, raw I y, I s',
                'float32 z',
                '''
                float value = z;
                for(int index=0;index<s;++index){
                    if(y[index] == i){
                        float xvalue = x[index];
                        if(value > xvalue){
                            value = value + log(1 + exp(xvalue - value));
                        }else{
                            value = xvalue + log(1 + exp(value - xvalue));
                        }
                    }
                    z = value;
                }
                ''',
                'reduce_probability')(multiply.astype(numpy.float32),
                                      path, path.shape[0], labels_prob)
        return labels_prob

    def calc_trans(self, path, yseq, rr, xp):
        forward_prob = self.log_matrix(xp.eye(path.shape[0])[0], xp)
        backward_prob = self.log_matrix(xp.eye(path.shape[0])[0], xp)

        alpha = []
        beta = []

        for t in six.moves.range(len(yseq)):
            # calc forward probability in log scale
            y = yseq[t]
            forward_prob = xp.take(y, path) + self.log_dot(forward_prob,
                                                           rr, xp)
            alpha.append(forward_prob)

            # calc backward probability
            y_inv = yseq[len(yseq) - t - 1]
            backward_prob = self.log_dot(backward_prob, rr, xp)
            beta.append(backward_prob[::-1])
            backward_prob = xp.take(y_inv, path[::-1]) + backward_prob
        return alpha, beta[::-1]

    def convert_inputs(self, labels, yseq, index, xp):
        log_yseq = [self.log_matrix(y[index], xp) for y in yseq]
        return labels[index], log_yseq

    def calc_path_probability(self, label_prob, total_prob, xp):
        return xp.exp(label_prob - total_prob)

    def calc_totalprob(self, forward_prob, backward_prob, xp):
        return logsumexp(forward_prob[0] + backward_prob[0], xp)

    def activate(self, yseq, xp):
        return [softmax(y, xp) for y in yseq]

    def forward(self, inputs):
        xp = cuda.get_array_module(inputs[0])
        yseq = self.activate(inputs[1::], xp)
        loss = 0
        batch_size = yseq[0].shape[0]
        for i in six.moves.range(batch_size):
            label, y = self.convert_inputs(inputs[0], yseq, i, xp)
            path = self.label_to_path(label, xp)
            rr = self.recurrence_relation(path.shape[0], xp)
            forward_prob_trans, backward_prob_trans\
                = self.calc_trans(path, y, rr, xp)
            loss += - logsumexp(forward_prob_trans[-1]
                                + backward_prob_trans[-1], xp)
        loss /= batch_size
        return utils.force_array(loss).astype(numpy.float32),

    def backward(self, inputs, grad_output):
        xp = cuda.get_array_module(inputs[0])
        yseq = self.activate(inputs[1::], xp)
        batch_size = yseq[0].shape[0]
        for b in six.moves.range(batch_size):
            label, y = self.convert_inputs(inputs[0], yseq, b, xp)
            path = self.label_to_path(label, xp)
            rr = self.recurrence_relation(path.shape[0], xp)
            forward_prob_trans, backward_prob_trans\
                = self.calc_trans(path, y, rr, xp)
            total_probability = self.calc_totalprob(forward_prob_trans,
                                                    backward_prob_trans, xp)
            for t in six.moves.range(len(y)):
                multiply = forward_prob_trans[t] + backward_prob_trans[t]
                label_prob = self.label_probability(y[t].shape[0],
                                                    path, multiply, xp)
                yseq[t][b] -= self.calc_path_probability(label_prob,
                                                         total_probability, xp)
                yseq[t][b] /= batch_size
                yseq[t][b] *= grad_output[0]
        return (None,) + tuple(yseq)


def connectionist_temporal_classification(blank_symbol, t, x):
    """Connectionist Temporal Classification loss function.

    Connectionist Temporal Classification(CTC) [Graves2006]_ is a loss function
    of sequence labeling where the alignment between the inputs and target is
    unknown. See also [Graves2012]_

    Args:
        blank_symbol (int): Index of blank_symbol.
                            This value must be non-negative.
        t (Variable): Expected label sequence.
        x (Variable): RNN output at each time.
                      (ex. :math:`(y_1, y_2, ..., y_T)`)

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
    return ConnectionistTemporalClassification(blank_symbol)(t, *x)

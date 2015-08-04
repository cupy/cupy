from chainer import cuda
from chainer import function
from chainer import utils
import numpy


class ConnectionistTemporalClassification(function.Function):

    def __init__(self, blank_symbol):
        self.blank_symbol = blank_symbol
        self.epsilon = 1e-10

    '''
    Transtion in forword and backword algorithms is represented as matrix.
    See also
    https://blog.wtf.sg/2014/10/06/connectionist-temporal-classification-ctc-with-theano/
    '''
    def recurrence_relation(self, size):
        big_I = numpy.eye(size+2)
        return (numpy.eye(size) + big_I[2:, 1:-1] +
                big_I[2:, :-2] * (numpy.arange(size) % 2))

    def label_to_path(self, labels):
        label_length = labels.shape[0]
        path = numpy.full((label_length * 2 + 1,),
                          self.blank_symbol, dtype=int)
        for i in range(label_length):
            path[i * 2 + 1] = labels[i]
        return path

    def forward_cpu(self, inputs):
        t = inputs[0]
        yseq = inputs[1::]
        path = self.label_to_path(t)
        rr = self.recurrence_relation(path.shape[0])
        forward_prob_trans, backward_prob_trans \
            = self.calc_trans_cpu(path, yseq, rr)
        return utils.force_array(- numpy.log(
            numpy.sum(forward_prob_trans[-1]
                      * backward_prob_trans[-1]))).astype(numpy.float32),

    def calc_trans_cpu(self, path, yseq, rr):
        forward_prob = numpy.eye(path.shape[0])[0]
        backward_prob = numpy.eye(path.shape[0])[0]

        alpha = ()
        beta = ()

        for t in range(len(yseq)):
            # calc forward probability
            y = yseq[t]
            forward_prob = y[path] * numpy.dot(forward_prob, rr)
            alpha += forward_prob,

            # calc backward probability
            y_inv = yseq[len(yseq) - t - 1]
            backward_prob = numpy.dot(backward_prob, rr)
            beta += backward_prob[::-1],
            backward_prob = y_inv[path[::-1]] * backward_prob
        return alpha, beta[::-1]

    # path probablity to label probability
    def label_probability(self, label_size, path, multiply):
        labels_prob = numpy.zeros(label_size)
        chars = set([c for c in path])
        for c in chars:
            pos = numpy.where(path == c)[0]
            labels_prob[c] = numpy.sum(multiply[pos, ])
        return labels_prob

    def backward_cpu(self, inputs, grad_output):
        labels = inputs[0]
        yseq = inputs[1::]
        path = self.label_to_path(labels)
        rr = self.recurrence_relation(path.shape[0])
        result = (None,)
        forward_prob_trans, backward_prob_trans \
            = self.calc_trans_cpu(path, yseq, rr)
        total_probability = numpy.sum(forward_prob_trans[0]
                                      * backward_prob_trans[0])
        for t in range(len(yseq)):
            multiply = forward_prob_trans[t] * backward_prob_trans[t]
            label_prob = self.label_probability(yseq[t].shape[0],
                                                path, multiply)
            # add epsilon to avoid to devide by 0.
            result += (- label_prob /
                       ((yseq[t] + self.epsilon)
                        * total_probability)) * grad_output[0],
        return result

    def forward_gpu(self, inputs):
        t = cuda.to_cpu(inputs[0])
        yseq = inputs[1::]
        path = self.label_to_path(t)
        rr = cuda.to_gpu(self.recurrence_relation(path.shape[0]))
        forward_prob = cuda.to_gpu(numpy.eye(path.shape[0])[0])
        for t in range(len(yseq)):
            # calc forward probability
            y = cuda.to_cpu(yseq[t])
            forward_prob = cuda.to_gpu(y[path]) * cuda.culinalg.dot(
                forward_prob.reshape(1, forward_prob.shape[0]), rr)
        forward_prob = cuda.to_cpu(forward_prob)
        return utils.force_array(
            - numpy.log(forward_prob[-2]
                        + forward_prob[-1])).astype(numpy.float32),

    def calc_trans_gpu(self, path, yseq, rr):
        forward_prob = cuda.to_gpu(numpy.eye(path.shape[0])[0])
        backward_prob = cuda.to_gpu(numpy.eye(path.shape[0])[0])

        alpha = ()
        beta = ()

        for t in range(len(yseq)):
            # calc forward probability
            y = cuda.to_cpu(yseq[t])
            forward_prob = cuda.to_gpu(y[path]) * cuda.culinalg.dot(
                forward_prob.reshape(1, forward_prob.shape[0]), rr)
            alpha += forward_prob,

            # calc backward probability
            y_inv = cuda.to_cpu(yseq[len(yseq) - t - 1])
            backward_prob = cuda.culinalg.dot(
                backward_prob.reshape(1, backward_prob.shape[0]), rr)
            beta += cuda.to_gpu(cuda.to_cpu(backward_prob)[0][::-1].copy()),
            backward_prob = cuda.to_gpu(y_inv[path[::-1]]) * backward_prob
        return alpha, beta[::-1]

    def backward_gpu(self, inputs, grad_output):
        t = cuda.to_cpu(inputs[0])
        yseq = inputs[1::]
        path = self.label_to_path(t)
        rr = cuda.to_gpu(self.recurrence_relation(path.shape[0]))
        result = (None,)
        forward_prob_trans, backward_prob_trans\
            = self.calc_trans_gpu(path, yseq, rr)
        with cuda.using_cumisc():
            total_probability = cuda.to_cpu(cuda.cumisc.sum(
                forward_prob_trans[0] * backward_prob_trans[0]))
        for t in range(len(yseq)):
            y = cuda.to_cpu(yseq[t])
            multiply = cuda.to_cpu(
                forward_prob_trans[t] * backward_prob_trans[t])
            label_prob = self.label_probability(y.shape[0], path, multiply)
            # add epsilon to avoid to devide by 0.
            result += cuda.to_gpu(- label_prob /
                                  ((y + self.epsilon) * total_probability)
                                  * grad_output[0]),
        return result


def connectionist_temporal_classification(blank_symbol, t, x):
    """Connectionist Temporal Classification loss function.

    Connectionist Temporal Classification(CTC) [Graves2006]_ is a loss function
    of sequence labeling where where the alignment between the inputs
    and target is unknown. See also [Graves2012]_

    Args:
        blank_symbol (int): Index of blank_symbol.
                            This value must be non-negative.
        t (Variable): Expected label sequence.
        x (Variable): RNN output as probability of
                      each charactor at each time.
                      (ex. :math:`(y_1, y_2, ..., y_T)`)

    Returns:
        Variable: A variable holding a scalar value of the CTC loss.

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
    assert blank_symbol < x[0].data.shape[0]
    return ConnectionistTemporalClassification(blank_symbol)(t, *x)

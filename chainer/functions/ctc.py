from chainer import cuda
from chainer import function
from chainer import utils
import numpy


class ConnectionistTemporalClassification(function.Function):

    def __init__(self, blank_symbol):
        self.blank_symbol = blank_symbol
        self.epsilon = 1e-40

    '''
    Transtion in forword and backword algorithms is represented as matrix.
    See also
    https://blog.wtf.sg/2014/10/06/connectionist-temporal-classification-ctc-with-theano/
    '''
    def recurrence_relation(self, size, vtype):
        big_I = numpy.eye(size+2)
        rr = numpy.log((numpy.eye(size) + big_I[2:, 1:-1] +
                        big_I[2:, :-2] * (numpy.arange(size) % 2)) + self.epsilon)
        if vtype == numpy:
            return rr
        else:
            return cuda.to_gpu(rr)

    def label_to_path(self, labels):
        label_length = labels.shape[0]
        path = numpy.full((label_length * 2 + 1,),
                          self.blank_symbol, dtype=int)
        for i in range(label_length):
            path[i * 2 + 1] = labels[i]
        return path

    def log_dot(self, prob, rr):
        vtype = cuda.cupy.get_array_module(prob)
        res = vtype.zeros(prob.shape)
        rtrans = vtype.swapaxes(rr, 1, 0)
        for i in range(rtrans.shape[0]):
            res[i] = self.logsumexp(prob + rtrans[i])
        return res

    def logsumexp(self, a):
        vtype = cuda.cupy.get_array_module(a)
        vmax = vtype.amax(a)
        res = vtype.log(vtype.sum(vtype.exp(a - vmax)))
        return (res + vmax).astype(numpy.float32)

    # path probablity to label probability
    def label_probability(self, label_size, path, multiply):
        labels_prob = numpy.log(numpy.zeros(label_size) + self.epsilon)
        chars = set([c for c in path])
        for c in chars:
            pos = numpy.where(path == c)[0]
            labels_prob[c] = self.logsumexp(multiply[pos, ])
        return labels_prob

    def forward_cpu(self, inputs):
        t = inputs[0]
        yseq = numpy.log(inputs[1::])
        path = self.label_to_path(t)
        rr = self.recurrence_relation(path.shape[0],
                                      cuda.cupy.get_array_module(yseq))
        forward_prob_trans, backward_prob_trans \
            = self.calc_trans_cpu(path, yseq, rr)
        return utils.force_array(
            - self.logsumexp(forward_prob_trans[-1]
                             + backward_prob_trans[-1])).astype(numpy.float32),

    def calc_trans_cpu(self, path, yseq, rr):
        forward_prob = numpy.log(numpy.eye(path.shape[0])[0]+self.epsilon)
        backward_prob = numpy.log(numpy.eye(path.shape[0])[0]+self.epsilon)

        alpha = ()
        beta = ()

        for t in range(len(yseq)):
            # calc forward probability in log scale
            y = yseq[t]
            forward_prob = y[path] + self.log_dot(forward_prob, rr)
            alpha += forward_prob,

            # calc backward probability
            y_inv = yseq[len(yseq) - t - 1]
            backward_prob = self.log_dot(backward_prob, rr)
            beta += backward_prob[::-1],
            backward_prob = y_inv[path[::-1]] + backward_prob
        return alpha, beta[::-1]

    def backward_cpu(self, inputs, grad_output):
        labels = inputs[0]
        yseq = numpy.log(inputs[1::])
        path = self.label_to_path(labels)
        rr = self.recurrence_relation(path.shape[0],
                                      cuda.cupy.get_array_module(yseq))
        forward_prob_trans, backward_prob_trans \
            = self.calc_trans_cpu(path, yseq, rr)
        total_probability = self.logsumexp(forward_prob_trans[0]
                                           + backward_prob_trans[0])
        result = (None,)
        for t in range(len(yseq)):
            multiply = forward_prob_trans[t] + backward_prob_trans[t]
            label_prob = self.label_probability(yseq[t].shape[0],
                                                path, multiply)
            result += (- numpy.exp(label_prob
                                   - (yseq[t] + total_probability))
                       * grad_output[0]).astype(numpy.float32),
        return result

    def forward_gpu(self, inputs):
        t = cuda.to_cpu(inputs[0])
        yseq = inputs[1::]
        path = self.label_to_path(t)
        rr = self.recurrence_relation(path.shape[0],
                                      cuda.cupy.get_array_module(yseq[0]))
        forward_prob_trans, backward_prob_trans \
            = self.calc_trans_gpu(path, yseq, rr)
        return - self.logsumexp(forward_prob_trans[-1]
                                + backward_prob_trans[-1]),

    def calc_trans_gpu(self, path, yseq, rr):
        forward_prob = cuda.cupy.log(cuda.cupy.eye(
            path.shape[0])[0]+self.epsilon)
        backward_prob = cuda.cupy.log(cuda.cupy.eye(
            path.shape[0])[0]+self.epsilon)

        alpha = ()
        beta = ()

        for t in range(len(yseq)):
            # calc forward probability
            y = cuda.to_cpu(cuda.cupy.log(yseq[t]))
            forward_prob = cuda.to_gpu(y[path]) \
                + self.log_dot(forward_prob, rr)
            alpha += forward_prob,

            # calc backward probability
            y_inv = cuda.to_cpu(cuda.cupy.log(yseq[len(yseq) - t - 1]))
            backward_prob = self.log_dot(backward_prob, rr)
            beta += backward_prob[::-1],
            backward_prob = cuda.to_gpu(y_inv[path[::-1]]) + backward_prob
        return alpha, beta[::-1]

    def backward_gpu(self, inputs, grad_output):
        t = cuda.to_cpu(inputs[0])
        yseq = inputs[1::]
        path = self.label_to_path(t)
        rr = self.recurrence_relation(path.shape[0],
                                      cuda.cupy.get_array_module(yseq[0]))
        result = (None,)
        forward_prob_trans, backward_prob_trans\
            = self.calc_trans_gpu(path, yseq, rr)
        total_probability = cuda.to_cpu(
            self.logsumexp(forward_prob_trans[0]
                           + backward_prob_trans[0]))
        for t in range(len(yseq)):
            y = cuda.to_cpu(cuda.cupy.log(yseq[t]))
            multiply = cuda.to_cpu(forward_prob_trans[t]
                                   + backward_prob_trans[t])
            label_prob = self.label_probability(y.shape[0], path, multiply)
            result += (- cuda.to_gpu(
                numpy.exp(label_prob
                          - (y + total_probability))).astype(numpy.float32)
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

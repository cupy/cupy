from chainer import cuda
from chainer import function
from chainer import utils
import numpy


class ConnectionistTemporalClassification(function.Function):

    def __init__(self, blank_symbol):
        self.blank_symbol = blank_symbol
        self.zero_padding = -10000000000

    '''
    Transtion in forword and backword algorithms is represented as matrix.
    See also
    https://blog.wtf.sg/2014/10/06/connectionist-temporal-classification-ctc-with-theano/
    '''
    def recurrence_relation(self, size, xp):
        big_I = numpy.eye(size+2)
        rr = numpy.ma.log((numpy.eye(size) + big_I[2:, 1:-1] +
                           big_I[2:, :-2] * (numpy.arange(size) % 2)))\
                     .filled(fill_value=self.zero_padding)
        if xp == numpy:
            return rr
        else:
            return cuda.to_gpu(rr)

    def force_cpu(self, v):
        if cuda.get_array_module(v) == cuda.cupy:
            v = cuda.to_cpu(v)
        return v

    def force_gpu(self, v):
        if cuda.get_array_module(v) == numpy:
            v = cuda.to_gpu(v)
        return v

    def label_to_path(self, labels):
        label_length = labels.shape[0]
        path = numpy.full((label_length * 2 + 1,),
                          self.blank_symbol, dtype=int)
        for i in range(label_length):
            path[i * 2 + 1] = labels[i]
        return path

    def log_dot(self, prob, rr):
        xp = cuda.cupy.get_array_module(prob)
        res = xp.zeros(prob.shape)
        rtrans = xp.swapaxes(rr, 1, 0)
        for i in range(rtrans.shape[0]):
            res[i] = self.logsumexp(prob + rtrans[i])
        return res

    def logsumexp(self, a):
        xp = cuda.cupy.get_array_module(a)
        vmax = xp.amax(a)
        res = xp.log(xp.sum(xp.exp(a - vmax)))
        return (res + vmax).astype(numpy.float32)

    # path probablity to label probability
    def label_probability(self, label_size, path, multiply):
        labels_prob = numpy.ma.log(numpy.zeros(label_size))\
                              .filled(fill_value=self.zero_padding)
        chars = set([c for c in path])
        for c in chars:
            pos = numpy.where(path == c)[0]
            labels_prob[c] = self.logsumexp(self.force_cpu(multiply)[pos, ])
        return labels_prob

    def calc_trans(self, path, yseq, rr):
        forward_prob = numpy.ma.log(numpy.eye(path.shape[0])[0])\
                               .filled(fill_value=self.zero_padding)
        backward_prob = numpy.ma.log(numpy.eye(path.shape[0])[0])\
                                .filled(fill_value=self.zero_padding)
        xp = cuda.get_array_module(yseq[0])
        if xp == cuda.cupy:
            forward_prob = cuda.to_gpu(forward_prob)
            backward_prob = cuda.to_gpu(backward_prob)

        alpha = ()
        beta = ()

        for t in range(len(yseq)):
            # calc forward probability in log scale
            y = self.force_cpu(yseq[t])
            if xp == cuda.cupy:
                forward_prob = cuda.to_gpu(cuda.to_cpu(y[path])) \
                    + self.log_dot(forward_prob, rr)
            else:
                forward_prob = y[path] + self.log_dot(forward_prob, rr)
            alpha += forward_prob,

            # calc backward probability
            y_inv = self.force_cpu(yseq[len(yseq) - t - 1])
            backward_prob = self.log_dot(backward_prob, rr)
            beta += backward_prob[::-1],
            if xp == cuda.cupy:
                backward_prob = cuda.to_gpu(y_inv[path[::-1]]) + backward_prob
            else:
                backward_prob = y_inv[path[::-1]] + backward_prob
        return alpha, beta[::-1]

    def convert_inputs(self, inputs):
        xp = cuda.get_array_module(inputs[0])
        labels = inputs[0]
        yseq = inputs[1::]
        log_yseq = ()
        labels = self.force_cpu(labels)

        for y in yseq:
            log_y = numpy.ma.log(self.force_cpu(y))\
                            .filled(fill_value=self.zero_padding)
            if xp == cuda.cupy:
                log_y = cuda.to_gpu(log_y)
            log_yseq += log_y,
        return labels, log_yseq

    def calc_differential(self, label_prob, y, total_prob):
        xp = cuda.get_array_module(y)
        y = self.force_cpu(y)
        differential = utils.force_array(
            - numpy.exp(label_prob
                        - (y + total_prob)).astype(numpy.float32))
        if xp == cuda.cupy:
            differential = cuda.to_gpu(differential)
        return differential

    def calc_totalprob(self, forward_prob, backward_prob):
        return self.force_cpu(self.logsumexp(forward_prob[0]
                                             + backward_prob[0]))

    def forward(self, inputs):
        label, yseq = self.convert_inputs(inputs)
        path = self.label_to_path(label)
        rr = self.recurrence_relation(path.shape[0],
                                      cuda.cupy.get_array_module(yseq[0]))
        forward_prob_trans, backward_prob_trans\
            = self.calc_trans(path, yseq, rr)
        return utils.force_array(- self.logsumexp(forward_prob_trans[-1]
                                                  + backward_prob_trans[-1])),

    def backward(self, inputs, grad_output):
        label, yseq = self.convert_inputs(inputs)
        path = self.label_to_path(label)
        rr = self.recurrence_relation(path.shape[0],
                                      cuda.cupy.get_array_module(yseq[0]))
        result = (None,)
        forward_prob_trans, backward_prob_trans\
            = self.calc_trans(path, yseq, rr)
        total_probability = self.calc_totalprob(forward_prob_trans,
                                                backward_prob_trans)
        for t in range(len(yseq)):
            multiply = forward_prob_trans[t] + backward_prob_trans[t]
            label_prob = self.label_probability(yseq[t].shape[0],
                                                path, multiply)
            result += self.calc_differential(
                label_prob, yseq[t],
                total_probability) * grad_output[0],
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

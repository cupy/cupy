from chainer import cuda
from chainer import function
from chainer import utils
import numpy


class ConnectionistTemporalClassification(function.Function):

    def __init__(self, blank_symbol):
        self.blank_symbol = blank_symbol
        self.zero_padding = -10000000000.0

    '''
    Transtion in forword and backword algorithms is represented as matrix.
    See also
    https://blog.wtf.sg/2014/10/06/connectionist-temporal-classification-ctc-with-theano/
    '''

    def log_matrix(self, x):
        xp = cuda.get_array_module(x)
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

    def recurrence_relation(self, size, xp):
        rr = (xp.eye(size) + xp.eye(size, k=1) +
              xp.eye(size, k=2) * (xp.arange(size) % 2))
        return self.log_matrix(rr)

    def label_to_path(self, labels):
        xp = cuda.get_array_module(labels)
        label_length = labels.shape[0]
        path = xp.full((label_length * 2 + 1,),
                       self.blank_symbol, dtype=int)
        for i in range(label_length):
            path[i * 2 + 1] = labels[i]
        return path

    def log_dot(self, prob, rr):
        xp = cuda.get_array_module(prob)
        res = xp.zeros(prob.shape)
        rtrans = xp.swapaxes(rr, 1, 0)
        for i in range(rtrans.shape[0]):
            res[i] = self.logsumexp(prob + rtrans[i])
        return res

    def logsumexp(self, a):
        xp = cuda.get_array_module(a)
        vmax = xp.amax(a)
        res = xp.log(xp.sum(xp.exp(a - vmax)))
        return (res + vmax).astype(numpy.float32)

    # path probablity to label probability
    def label_probability(self, label_size, path, multiply):
        xp = cuda.get_array_module(path)
        chars = set([c for c in path])
        labels_prob = self.log_matrix(xp.zeros((label_size,),
                                               dtype=numpy.float32))
        if xp == numpy:
            for c in chars:
                pos = numpy.where(path == c)[0]
                labels_prob[c] = self.logsumexp(xp.take(multiply, pos))
        else:
            cuda.cupy.ElementwiseKernel(
                'raw float32 x, raw I y, I s',
                'float32 z',
                '''
                float value = z;
                for(int index=0;index<s;++index){
                    if(y[index] == i){
                        if(value > x[index]){
                            value = value + log(1 + exp(x[index] - value));
                        }else{
                            value = x[index] + log(1 + exp(value -x[index]));
                        }
                    }
                    atomicExch(&z, value);
                }
                ''',
                'reduce_probability')(multiply.astype(numpy.float32),
                                      path, path.shape[0], labels_prob)
        return labels_prob

    def calc_trans(self, path, yseq, rr):
        xp = cuda.get_array_module(yseq[0])
        forward_prob = self.log_matrix(xp.eye(path.shape[0])[0])
        backward_prob = self.log_matrix(xp.eye(path.shape[0])[0])

        alpha = ()
        beta = ()

        for t in range(len(yseq)):
            # calc forward probability in log scale
            y = yseq[t]
            forward_prob = xp.take(y, path) + self.log_dot(forward_prob, rr)
            alpha += forward_prob,

            # calc backward probability
            y_inv = yseq[len(yseq) - t - 1]
            backward_prob = self.log_dot(backward_prob, rr)
            beta += backward_prob[::-1],
            backward_prob = xp.take(y_inv, path[::-1]) + backward_prob
        return alpha, beta[::-1]

    def convert_inputs(self, labels, yseq, index):
        log_yseq = ()
        for y in yseq:
            log_yseq += self.log_matrix(y[index]),
        return labels[index], log_yseq

    def calc_path_probability(self, label_prob, total_prob):
        xp = cuda.get_array_module(label_prob)
        return xp.exp(label_prob - total_prob)

    def calc_totalprob(self, forward_prob, backward_prob):
        return self.logsumexp(forward_prob[0] + backward_prob[0])

    def softmax(self, x, axis=None):
        xp = cuda.get_array_module(x)
        val = x - xp.amax(x, axis=axis).reshape(x.shape[0], 1)
        val = xp.exp(val)
        return val / xp.sum(val, axis=axis).reshape(x.shape[0], 1)

    def activate(self, yseq):
        result = ()
        for y in yseq:
            result += self.softmax(y, axis=1),
        return result

    def forward(self, inputs):
        yseq = self.activate(inputs[1::])
        loss = 0
        for i in range(yseq[0].shape[0]):
            label, y = self.convert_inputs(inputs[0], yseq, i)
            path = self.label_to_path(label)
            rr = self.recurrence_relation(path.shape[0],
                                          cuda.get_array_module(y[0]))
            forward_prob_trans, backward_prob_trans\
                = self.calc_trans(path, y, rr)
            loss += - self.logsumexp(forward_prob_trans[-1]
                                     + backward_prob_trans[-1])
        return utils.force_array(loss).astype(numpy.float32),

    def backward(self, inputs, grad_output):
        yseq = self.activate(inputs[1::])
        delta = []
        for t in range(len(yseq)):
            delta.append(yseq[t])
        batch_size = yseq[0].shape[0]
        for b in range(batch_size):
            label, y = self.convert_inputs(inputs[0], yseq, b)
            path = self.label_to_path(label)
            rr = self.recurrence_relation(path.shape[0],
                                          cuda.get_array_module(yseq[0]))
            forward_prob_trans, backward_prob_trans\
                = self.calc_trans(path, y, rr)
            total_probability = self.calc_totalprob(forward_prob_trans,
                                                    backward_prob_trans)
            for t in range(len(y)):
                multiply = forward_prob_trans[t] + backward_prob_trans[t]
                label_prob = self.label_probability(y[t].shape[0],
                                                    path, multiply)
                # fixme: batch normarization of padded data.
                delta[t][b] -= self.calc_path_probability(label_prob,
                                                          total_probability)
                delta[t][b] *= grad_output[0]
        return (None,) + tuple(delta)


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
    assert blank_symbol < x[0].data.shape[1]
    return ConnectionistTemporalClassification(blank_symbol)(t, *x)

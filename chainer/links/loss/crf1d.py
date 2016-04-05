from chainer.functions.loss import crf1d
from chainer import link


class CRF1d(link.Link):

    def __init__(self, n_label):
        super(CRF1d, self).__init__(cost=(n_label, n_label))
        self.cost.data[...] = 0

    def __call__(self, xs, ys):
        return crf1d.crf1d(self.cost, xs, ys)

    def viterbi(self, xs):
        return crf1d.crf1d_viterbi(self.cost, xs)

from chainer.functions.connection import n_step_lstm
from chainer import link


class NStepLSTM(link.Link):

    def __init__(self, n_layers, in_size, out_size):
        super(NStepLSTM, self).__init__(
            w=(n_layers, 8, out_size, out_size),
            b=(n_layers, 8, out_size),
            #w=(n_layers * 8 * (out_size * out_size + out_size), 1, 1),
        )
        self.n_layers = n_layers

    def __call__(self, h, c, x):
        return n_step_lstm.NStepLSTM(n_layers=self.n_layers)(h, c, x, self.w, self.b)

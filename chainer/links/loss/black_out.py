import numpy

import chainer
from chainer import cuda
from chainer.functions.loss import black_out
from chainer import link
from chainer.utils import walker_alias


class BlackOut(link.Link):

    """BlackOut loss layer.

    .. seealso:: :func:`~chainer.functions.black_out` for more detail.

    Attributes:
        W (~chainer.Variable): Weight parameter matrix.

    """

    def __init__(self, in_size, counts, sample_size):
        vocab_size = len(counts)
        super(BlackOut, self).__init__(W=(vocab_size, in_size))
        p = numpy.array(counts, dtype=numpy.float32)
        self.sampler = walker_alias.WalkerAlias(p)
        self.sample_size = sample_size

    def to_cpu(self):
        super(BlackOut, self).to_cpu()
        self.sampler.to_cpu()

    def to_gpu(self, device=None):
        with cuda.get_device(device):
            super(BlackOut, self).to_gpu()
            self.sampler.to_gpu()

    def __call__(self, x, t):
        batch_size = x.data.shape[0]
        if hasattr(self, 'sample_data'):
            # for test
            sample_data = self.sample_data
        else:
            shape = (batch_size, self.sample_size)
            sample_data = self.sampler.sample(shape)
        samples = chainer.Variable(sample_data)
        return black_out.black_out(x, t, self.W, samples)

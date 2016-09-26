import numpy

from chainer import cuda
from chainer.functions.loss import negative_sampling
from chainer import link
from chainer.utils import walker_alias


class NegativeSampling(link.Link):

    """Negative sampling loss layer.

    This link wraps the :func:`~chainer.functions.negative_sampling` function.
    It holds the weight matrix as a parameter. It also builds a sampler
    internally given a list of word counts.

    Args:
        in_size (int): Dimension of input vectors.
        counts (int list): Number of each identifiers.
        sample_size (int): Number of negative samples.
        power (float): Power factor :math:`\\alpha`.

    .. seealso:: :func:`~chainer.functions.negative_sampling` for more detail.

    Attributes:
        W (~chainer.Variable): Weight parameter matrix.

    """

    def __init__(self, in_size, counts, sample_size, power=0.75):
        vocab_size = len(counts)
        super(NegativeSampling, self).__init__(W=(vocab_size, in_size))
        self.W.data.fill(0)

        self.sample_size = sample_size
        power = numpy.float32(power)
        p = numpy.array(counts, power.dtype)
        numpy.power(p, power, p)
        self.sampler = walker_alias.WalkerAlias(p)

    def to_cpu(self):
        super(NegativeSampling, self).to_cpu()
        self.sampler.to_cpu()

    def to_gpu(self, device=None):
        with cuda.get_device(device):
            super(NegativeSampling, self).to_gpu()
            self.sampler.to_gpu()

    def __call__(self, x, t):
        """Computes the loss value for given input and ground truth labels.

        Args:
            x (~chainer.Variable): Input of the weight matrix multiplication.
            t (~chainer.Variable): Batch of ground truth labels.

        Returns:
            ~chainer.Variable: Loss value.

        """
        return negative_sampling.negative_sampling(
            x, t, self.W, self.sampler.sample, self.sample_size)

import numpy

from chainer import initializer


class Orthogonal(initializer.Initializer):
    """From Lasagne.

    Reference: Saxe et al., http://arxiv.org/abs/1312.6120

    """

    def __init__(self, scale=1.1):
        self.scale = scale

    def __call__(self, shape):
        if not shape:
            return numpy.array(1)
        flat_shape = (shape[0], numpy.prod(shape[1:]))
        a = numpy.random.standard_normal(flat_shape)
        u, _, v = numpy.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return self.scale * q

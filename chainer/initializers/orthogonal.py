import numpy

from chainer import cuda
from chainer import initializer


# Original code forked from MIT licensed keras project
# https://github.com/fchollet/keras/blob/master/keras/initializations.py

class Orthogonal(initializer.Initializer):
    """From Lasagne.

    Reference: Saxe et al., http://arxiv.org/abs/1312.6120

    """

    def __init__(self, scale=1.1):
        self.scale = scale

    # TODO(Kenta Oono)
    # How do we treat overcomplete base-system case?
    def __call__(self, array):
        xp = cuda.get_array_module(array)
        if not array.shape:
            array[...] = self.scale
        elif array.size:
            flat_shape = (len(array), numpy.prod(array.shape[1:]))
            if flat_shape[0] > flat_shape[1]:
                raise ValueError('Cannot make orthogonal system because'
                                 '# of vectors ({}) is larger than'
                                 ' that of dimensions({})'.format(
                                     flat_shape[0], flat_shape[1]))
            a = numpy.random.standard_normal(flat_shape)
            # we do not have cupy.linalg.svd for now
            u, _, v = numpy.linalg.svd(a, full_matrices=False)
            # pick the one with the correct shape
            q = u if u.shape == flat_shape else v
            array[...] = xp.asarray(q.reshape(array.shape))
            array *= self.scale

import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _reverse_indices(indices):
    r = numpy.empty(len(indices), 'i')
    for i, ind in enumerate(indices):
        r[ind] = i
    return r


class Permutate(function.Function):

    def __init__(self, indices, axis=0, rev=False):
        self.indices = indices
        self.axis = axis
        self.rev = rev

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type = in_types[0]
        if self.axis < 0:
            type_check.expect(x_type.ndim >= -self.axis)
        else:
            type_check.expect(x_type.ndim > self.axis)

        type_check.expect(x_type.shape[self.axis] == len(self.indices))

    def _permutate(self, x, rev):
        xp = cuda.get_array_module(x)
        if rev:
            indices = _reverse_indices(self.indices)
        else:
            indices = self.indices

        if xp is not numpy:
            indices = xp.array(indices, 'i')
        return xp.take(x, indices, axis=self.axis)

    def forward(self, inputs):
        x = inputs[0]
        return self._permutate(x, self.rev),

    def backward(self, inputs, grads):
        g = grads[0]
        return self._permutate(g, not self.rev),


def permutate(x, indices, axis=0, rev=False):
    return Permutate(indices, axis=axis, rev=rev)(x)

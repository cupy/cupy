import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _transpose(xs):
    xp = cuda.get_array_module(*xs)
    lengths = numpy.zeros(len(xs[0]), dtype='i')
    for i, x in enumerate(xs):
        lengths[0:len(x)] = i + 1
    dtype = xs[0].dtype
    unit = xs[0].shape[1:]
    outs = tuple(xp.empty((l,) + unit, dtype=dtype) for l in lengths)

    for i, x in enumerate(xs):
        for p, xi in enumerate(x):
            outs[p][i] = xi

    return outs


class TransposeSequence(function.Function):

    def check_type_forward(self, xs_type):
        type_check.expect(xs_type.size() >= 1)
        for p, n in zip(xs_type, xs_type[1:]):
            type_check.expect(
                p.shape[0] >= n.shape[0],
                p.shape[1:] == n.shape[1:],
            )

    def forward(self, xs):
        return _transpose(xs)

    def backward(self, xs, gs):
        return _transpose(gs)


def transpose_sequence(xs):
    ys = TransposeSequence()(*xs)
    if not isinstance(ys, tuple):
        ys = (ys,)
    return ys

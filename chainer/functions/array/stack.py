from chainer.functions.array import concat
from chainer.functions.array import expand_dims


def stack(xs, axis=0):
    xs = [expand_dims.expand_dims(x, axis=axis) for x in xs]
    return concat.concat(xs, axis=axis)

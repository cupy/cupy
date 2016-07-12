from chainer.functions.array import reshape
from chainer.functions.array import split_axis


def separate(x, axis=0):
    shape = list(x.data.shape)
    del shape[axis]
    ys = split_axis.split_axis(x, x.data.shape[axis], axis, force_tuple=True)
    return tuple(reshape.reshape(y, shape) for y in ys)

from chainer import cuda
from chainer import function
from chainer import type_check


class Where(function.Fuction):

    def __init__(self, condition):
        self.condition = condition

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == self.condition.shape,
            in_types[1].shape == self.condition.shape,
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        return xp.where(self.condition, inputs[0], inputs[1]),

    def backward(self, inputs, grads):
        xp = cuda.get_array_module(*inputs)
        zeros = xp.zeros((), dtype=grads.dtype)
        gx = xp.where(self.condition, grads, zeros)
        gy = xp.where(~self.condition, grads, zeros)
        return gx, gy


def where(condition, x, y):
    return Where(condition)(x, y)

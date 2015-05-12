from chainer import Function

class Reshape(Function):
    """Reshapes an input array without copy."""

    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        return x[0].reshape(self.shape),

    def backward(self, x, gy):
        return gy[0].reshape(x[0].shape),

def reshape(x, shape):
    return Reshape(shape)(x)

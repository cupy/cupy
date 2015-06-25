from chainer import function


class Identity(function.Function):

    """Identity function."""

    def forward(self, xs):
        return xs

    def backward(self, xs, gys):
        return gys


def identity(*inputs):
    """Just returns input variables."""
    return Identity()(*inputs)

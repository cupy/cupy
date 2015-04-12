from chain import Function

class Identity(Function):
    """Identity function."""

    def forward(self, xs):
        return xs

    def backward(self, xs, gys):
        return gys

def identity(*inputs):
    return Identity()(*inputs)

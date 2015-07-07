from chainer import cuda
from chainer import function
from chainer.functions import linear as linear_module


class NonparameterizedLinear(function.Function):

    """Nonparameterized linear class.

    .. seealso:: :meth:`Linear`

    """

    def forward(self, x):
        W = x[1]
        b = None
        if len(x) == 3:
            b = x[2]
        out_size, in_size = W.shape
        func = linear_module.Linear(
            in_size, out_size, initialW=W, initial_bias=b)
        self.func = func
        if any(isinstance(i, cuda.GPUArray) for i in x):
            func.to_gpu()
        return func.forward(x[:1])

    def backward_cpu(self, x, gy):
        func = self.func
        func.zero_grads()
        gx = func.backward_cpu(x[:1], gy)
        if func.gb is None:
            return (gx[0], func.gW)
        return (gx[0], func.gW, func.gb)

    def backward_gpu(self, x, gy):
        func = self.func
        func.zero_grads()
        gx = func.backward_gpu(x[:1], gy)
        if func.gb is None:
            return (gx[0], func.gW)
        return (gx[0], func.gW, func.gb)


def linear(x, W, b=None, stride=1, pad=0, use_cudnn=True):
    """Nonparameterized linear function.

    Args:
        x (~chainer.Variable): Input variable.
        W (~chainer.Variable): Weight variable.
        b (~chainer.Variable): Bias variable.

    Returns:
        ~chainer.Variable: Output variable.

    .. seealso:: :meth:`Linear`

    """

    return NonparameterizedLinear()(x, W, b)

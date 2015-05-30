from chainer import Function, FunctionSet, Variable
from chainer.functions import concat, Convolution2D, max_pooling_2d, relu

class Inception(Function):
    """Inception module of GoogLeNet."""

    def __init__(self, in_channels, out1, proj3, out3, proj5, out5, proj_pool):
        self.f = FunctionSet(
            conv1 = Convolution2D(in_channels, out1,      1),
            proj3 = Convolution2D(in_channels, proj3,     1),
            conv3 = Convolution2D(proj3,       out3,      3, pad=1),
            proj5 = Convolution2D(in_channels, proj5,     1),
            conv5 = Convolution2D(proj5,       out5,      5, pad=2),
            projp = Convolution2D(in_channels, proj_pool, 1),
        )

    def forward(self, x):
        self.x = Variable(x[0])
        out1 = self.f.conv1(self.x)
        out3 = self.f.conv3(relu(self.f.proj3(self.x)))
        out5 = self.f.conv5(relu(self.f.proj5(self.x)))
        pool = self.f.projp(max_pooling_2d(self.x, 3, stride=1, pad=1))
        self.y = relu(concat((out1, out3, out5, pool), axis=1))

        return self.y.data,

    def backward(self, x, gy):
        self.y.grad = gy[0]
        self.y.backward()
        return self.x.grad,

    def to_gpu(self, device=None):
        return self.f.to_gpu(device)

    def to_cpu(self):
        return self.f.to_cpu()

    @property
    def parameters(self):
        return self.f.parameters

    @parameters.setter
    def parameters(self, params):
        self.f.parameters = params

    @property
    def gradients(self):
        return self.f.gradients

    @gradients.setter
    def gradients(self, grads):
        self.f.gradients = grads

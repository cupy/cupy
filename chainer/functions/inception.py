from chainer import Function, FunctionSet
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

    def __call__(self, x):
        f = self.f
        out1 = f.conv1(x)
        out3 = f.conv3(relu(f.proj3(x)))
        out5 = f.conv5(relu(f.proj5(x)))
        pool = f.projp(max_pooling_2d(x, 3, stride=1, pad=1))
        return relu(concat((out1, out3, out5, pool), axis=1))

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

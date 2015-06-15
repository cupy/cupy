from chainer import Function, FunctionSet, Variable
from chainer.functions import concat, Convolution2D, BatchNormalization, relu
from chainer.functions.pooling_2d import MaxPooling2D, AveragePooling2D

class InceptionBN(Function):
    """Inception module in new GoogLeNet with BN."""

    def __init__(self, in_channels, out1, proj3, out3, proj33, out33,
                 pooltype, proj_pool=None, stride=1):
        if out1 > 0:
            assert stride == 1
            assert proj_pool is not None

        self.f = FunctionSet(
            proj3    = Convolution2D(in_channels,  proj3, 1, nobias=True),
            conv3    = Convolution2D(      proj3,   out3, 3, pad=1, stride=stride, nobias=True),
            proj33   = Convolution2D(in_channels, proj33, 1, nobias=True),
            conv33a  = Convolution2D(     proj33,  out33, 3, pad=1, nobias=True),
            conv33b  = Convolution2D(      out33,  out33, 3, pad=1, stride=stride, nobias=True),
            proj3n   = BatchNormalization(proj3),
            conv3n   = BatchNormalization(out3),
            proj33n  = BatchNormalization(proj33),
            conv33an = BatchNormalization(out33),
            conv33bn = BatchNormalization(out33),
        )

        if out1 > 0:
            self.f.conv1  = Convolution2D(in_channels, out1, 1, stride=stride, nobias=True)
            self.f.conv1n = BatchNormalization(out1)

        if proj_pool is not None:
            self.f.poolp  = Convolution2D(in_channels, proj_pool, 1, nobias=True)
            self.f.poolpn = BatchNormalization(proj_pool)

        if pooltype == 'max':
            self.f.pool = MaxPooling2D(3, stride=stride, pad=1)
        elif pooltype == 'avg':
            self.f.pool = AveragePooling2D(3, stride=stride, pad=1)
        else:
            raise NotImplementedError()

    def forward(self, x):
        f = self.f

        self.x = Variable(x[0])
        outs = []

        if hasattr(f, 'conv1'):
            h1 = f.conv1(self.x)
            h1 = f.conv1n(h1)
            h1 = relu(h1)
            outs.append(h1)

        h3 = relu(f.proj3n(f.proj3(self.x)))
        h3 = relu(f.conv3n(f.conv3(h3)))
        outs.append(h3)

        h33 = relu(f.proj33n(f.proj33(self.x)))
        h33 = relu(f.conv33an(f.conv33a(h33)))
        h33 = relu(f.conv33bn(f.conv33b(h33)))
        outs.append(h33)

        p = f.pool(self.x)
        if hasattr(f, 'poolp'):
            p = relu(f.poolpn(f.poolp(p)))
        outs.append(p)

        self.y = concat(outs, axis=1)
        return self.y.data,

    def backward(self, x, gy):
        self.y.grad = gy[0]
        self.y.backward()
        return self.x.grad,

    def to_gpu(self, device=None):
        super(InceptionBN, self).to_gpu(device)
        self.f.to_gpu(device)

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

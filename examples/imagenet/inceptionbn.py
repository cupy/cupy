from chainer import Function, FunctionSet, Variable
import chainer.functions as F
from chainer.functions.pooling_2d import AveragePooling2D, MaxPooling2D

class InceptionBN(Function):
    """Inception module in new GoogLeNet with BN."""

    def __init__(self, in_channels, out1, proj3, out3, proj33, out33,
                 pooltype, proj_pool=None, stride=1):
        if out1 > 0:
            assert stride == 1
            assert proj_pool is not None

        self.f = FunctionSet(
            proj3    = F.Convolution2D(in_channels,  proj3, 1, nobias=True),
            conv3    = F.Convolution2D(      proj3,   out3, 3, pad=1, stride=stride, nobias=True),
            proj33   = F.Convolution2D(in_channels, proj33, 1, nobias=True),
            conv33a  = F.Convolution2D(     proj33,  out33, 3, pad=1, nobias=True),
            conv33b  = F.Convolution2D(      out33,  out33, 3, pad=1, stride=stride, nobias=True),
            proj3n   = F.BatchNormalization(proj3),
            conv3n   = F.BatchNormalization(out3),
            proj33n  = F.BatchNormalization(proj33),
            conv33an = F.BatchNormalization(out33),
            conv33bn = F.BatchNormalization(out33),
        )

        if out1 > 0:
            self.f.conv1  = F.Convolution2D(in_channels, out1, 1, stride=stride, nobias=True)
            self.f.conv1n = F.BatchNormalization(out1)

        if proj_pool is not None:
            self.f.poolp  = F.Convolution2D(in_channels, proj_pool, 1, nobias=True)
            self.f.poolpn = F.BatchNormalization(proj_pool)

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
            h1 = F.relu(h1)
            outs.append(h1)

        h3 = F.relu(f.proj3n(f.proj3(self.x)))
        h3 = F.relu(f.conv3n(f.conv3(h3)))
        outs.append(h3)

        h33 = F.relu(f.proj33n(f.proj33(self.x)))
        h33 = F.relu(f.conv33an(f.conv33a(h33)))
        h33 = F.relu(f.conv33bn(f.conv33b(h33)))
        outs.append(h33)

        p = f.pool(self.x)
        if hasattr(f, 'poolp'):
            p = F.relu(f.poolpn(f.poolp(p)))
        outs.append(p)

        self.y = F.concat(outs, axis=1)
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


class GoogLeNetBN(FunctionSet):
    """New GoogLeNet of BatchNormalization version."""

    insize = 224

    def __init__(self):
        super(GoogLeNetBN, self).__init__(
            conv1 = F.Convolution2D( 3,  64, 7, stride=2, pad=3, nobias=True),
            norm1 = F.BatchNormalization(64),
            conv2 = F.Convolution2D(64, 192, 3, pad=1, nobias=True),
            norm2 = F.BatchNormalization(192),
            inc3a = InceptionBN( 192,  64,  64,  64,  64,  96, 'avg',  32),
            inc3b = InceptionBN( 256,  64,  64,  96,  64,  96, 'avg',  64),
            inc3c = InceptionBN( 320,   0, 128, 160,  64,  96, 'max', stride=2),
            inc4a = InceptionBN( 576, 224,  64,  96,  96, 128, 'avg', 128),
            inc4b = InceptionBN( 576, 192,  96, 128,  96, 128, 'avg', 128),
            inc4c = InceptionBN( 576, 128, 128, 160, 128, 160, 'avg', 128),
            inc4d = InceptionBN( 576,  64, 128, 192, 160, 192, 'avg', 128),
            inc4e = InceptionBN( 576,   0, 128, 192, 192, 256, 'max', stride=2),
            inc5a = InceptionBN(1024, 352, 192, 320, 160, 224, 'avg', 128),
            inc5b = InceptionBN(1024, 352, 192, 320, 192, 224, 'max', 128),
            out   = F.Linear(1024, 1000),

            conva  = F.Convolution2D(576, 128, 1, nobias=True),
            norma  = F.BatchNormalization(128),
            lina   = F.Linear(2048, 1024, nobias=True),
            norma2 = F.BatchNormalization(1024),
            outa   = F.Linear(1024, 1000),

            convb  = F.Convolution2D(576, 128, 1, nobias=True),
            normb  = F.BatchNormalization(128),
            linb   = F.Linear(2048, 1024, nobias=True),
            normb2 = F.BatchNormalization(1024),
            outb   = F.Linear(1024, 1000),
        )

    def forward(self, x_data, y_data, train=True):
        x = Variable(x_data, volatile=not train)
        t = Variable(y_data, volatile=not train)

        h = F.max_pooling_2d(
            F.relu(self.norm1(self.conv1(x))),  3, stride=2, pad=1)
        h = F.max_pooling_2d(
            F.relu(self.norm2(self.conv2(h))), 3, stride=2, pad=1)

        h = self.inc3a(h)
        h = self.inc3b(h)
        h = self.inc3c(h)
        h = self.inc4a(h)

        a = F.average_pooling_2d(h, 5, stride=3)
        a = F.relu(self.norma(self.conva(a)))
        a = F.relu(self.norma2(self.lina(a)))
        a = self.outa(a)
        a = F.softmax_cross_entropy(a, t)

        h = self.inc4b(h)
        h = self.inc4c(h)
        h = self.inc4d(h)

        b = F.average_pooling_2d(h, 5, stride=3)
        b = F.relu(self.normb(self.convb(b)))
        b = F.relu(self.normb2(self.linb(b)))
        b = self.outb(b)
        b = F.softmax_cross_entropy(b, t)

        h = self.inc4e(h)
        h = self.inc5a(h)
        h = F.average_pooling_2d(self.inc5b(h), 7)
        h = self.out(h)
        return 0.3 * (a + b) + F.softmax_cross_entropy(h, t), F.accuracy(h, t)

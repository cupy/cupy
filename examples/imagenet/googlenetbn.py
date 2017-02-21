import numpy as np

import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L


class GoogLeNetBN(chainer.Chain):

    """New GoogLeNet of BatchNormalization version."""

    insize = 224

    def __init__(self):
        super(GoogLeNetBN, self).__init__(
            conv1=L.Convolution2D(None, 64, 7, stride=2, pad=3, nobias=True),
            norm1=L.BatchNormalization(64),
            conv2=L.Convolution2D(None, 192, 3, pad=1, nobias=True),
            norm2=L.BatchNormalization(192),
            inc3a=L.InceptionBN(None, 64, 64, 64, 64, 96, 'avg', 32),
            inc3b=L.InceptionBN(None, 64, 64, 96, 64, 96, 'avg', 64),
            inc3c=L.InceptionBN(None, 0, 128, 160, 64, 96, 'max', stride=2),
            inc4a=L.InceptionBN(None, 224, 64, 96, 96, 128, 'avg', 128),
            inc4b=L.InceptionBN(None, 192, 96, 128, 96, 128, 'avg', 128),
            inc4c=L.InceptionBN(None, 128, 128, 160, 128, 160, 'avg', 128),
            inc4d=L.InceptionBN(None, 64, 128, 192, 160, 192, 'avg', 128),
            inc4e=L.InceptionBN(None, 0, 128, 192, 192, 256, 'max', stride=2),
            inc5a=L.InceptionBN(None, 352, 192, 320, 160, 224, 'avg', 128),
            inc5b=L.InceptionBN(None, 352, 192, 320, 192, 224, 'max', 128),
            out=L.Linear(None, 1000),

            conva=L.Convolution2D(None, 128, 1, nobias=True),
            norma=L.BatchNormalization(128),
            lina=L.Linear(None, 1024, nobias=True),
            norma2=L.BatchNormalization(1024),
            outa=L.Linear(None, 1000),

            convb=L.Convolution2D(None, 128, 1, nobias=True),
            normb=L.BatchNormalization(128),
            linb=L.Linear(None, 1024, nobias=True),
            normb2=L.BatchNormalization(1024),
            outb=L.Linear(None, 1000),
        )
        self._train = True

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, value):
        self._train = value
        self.inc3a.train = value
        self.inc3b.train = value
        self.inc3c.train = value
        self.inc4a.train = value
        self.inc4b.train = value
        self.inc4c.train = value
        self.inc4d.train = value
        self.inc4e.train = value
        self.inc5a.train = value
        self.inc5b.train = value

    def __call__(self, x, t):
        test = not self.train

        h = F.max_pooling_2d(
            F.relu(self.norm1(self.conv1(x), test=test)),  3, stride=2, pad=1)
        h = F.max_pooling_2d(
            F.relu(self.norm2(self.conv2(h), test=test)), 3, stride=2, pad=1)

        h = self.inc3a(h)
        h = self.inc3b(h)
        h = self.inc3c(h)
        h = self.inc4a(h)

        a = F.average_pooling_2d(h, 5, stride=3)
        a = F.relu(self.norma(self.conva(a), test=test))
        a = F.relu(self.norma2(self.lina(a), test=test))
        a = self.outa(a)
        loss1 = F.softmax_cross_entropy(a, t)

        h = self.inc4b(h)
        h = self.inc4c(h)
        h = self.inc4d(h)

        b = F.average_pooling_2d(h, 5, stride=3)
        b = F.relu(self.normb(self.convb(b), test=test))
        b = F.relu(self.normb2(self.linb(b), test=test))
        b = self.outb(b)
        loss2 = F.softmax_cross_entropy(b, t)

        h = self.inc4e(h)
        h = self.inc5a(h)
        h = F.average_pooling_2d(self.inc5b(h), 7)
        h = self.out(h)
        loss3 = F.softmax_cross_entropy(h, t)

        loss = 0.3 * (loss1 + loss2) + loss3
        accuracy = F.accuracy(h, t)

        chainer.report({
            'loss': loss,
            'loss1': loss1,
            'loss2': loss2,
            'loss3': loss3,
            'accuracy': accuracy,
        }, self)
        return loss


class GoogLeNetBNFp16(GoogLeNetBN):

    """New GoogLeNet of BatchNormalization version."""

    insize = 224

    def __init__(self):
        self.dtype = dtype = np.float16
        W = initializers.HeNormal(1 / np.sqrt(2), self.dtype)
        bias = initializers.Zero(self.dtype)

        chainer.Chain.__init__(
            self,
            conv1=L.Convolution2D(None, 64, 7, stride=2, pad=3, initialW=W,
                                  nobias=True),
            norm1=L.BatchNormalization(64, dtype=dtype),
            conv2=L.Convolution2D(None, 192, 3,
                                  pad=1, initialW=W, nobias=True),
            norm2=L.BatchNormalization(192, dtype=dtype),
            inc3a=L.InceptionBN(None, 64, 64, 64, 64, 96, 'avg', 32,
                                conv_init=W, dtype=dtype),
            inc3b=L.InceptionBN(None, 64, 64, 96, 64, 96, 'avg', 64,
                                conv_init=W, dtype=dtype),
            inc3c=L.InceptionBN(None, 0, 128, 160, 64, 96, 'max', stride=2,
                                conv_init=W, dtype=dtype),
            inc4a=L.InceptionBN(None, 224, 64, 96, 96, 128, 'avg', 128,
                                conv_init=W, dtype=dtype),
            inc4b=L.InceptionBN(None, 192, 96, 128, 96, 128, 'avg', 128,
                                conv_init=W, dtype=dtype),
            inc4c=L.InceptionBN(None, 128, 128, 160, 128, 160, 'avg', 128,
                                conv_init=W, dtype=dtype),
            inc4d=L.InceptionBN(None, 64, 128, 192, 160, 192, 'avg', 128,
                                conv_init=W, dtype=dtype),
            inc4e=L.InceptionBN(None, 0, 128, 192, 192, 256, 'max', stride=2,
                                conv_init=W, dtype=dtype),
            inc5a=L.InceptionBN(None, 352, 192, 320, 160, 224, 'avg', 128,
                                conv_init=W, dtype=dtype),
            inc5b=L.InceptionBN(None, 352, 192, 320, 192, 224, 'max', 128,
                                conv_init=W, dtype=dtype),
            out=L.Linear(None, 1000, initialW=W, bias=bias),

            conva=L.Convolution2D(None, 128, 1, initialW=W, nobias=True),
            norma=L.BatchNormalization(128, dtype=dtype),
            lina=L.Linear(None, 1024, initialW=W, nobias=True),
            norma2=L.BatchNormalization(1024, dtype=dtype),
            outa=L.Linear(None, 1000, initialW=W, bias=bias),

            convb=L.Convolution2D(None, 128, 1, initialW=W, nobias=True),
            normb=L.BatchNormalization(128, dtype=dtype),
            linb=L.Linear(None, 1024, initialW=W, nobias=True),
            normb2=L.BatchNormalization(1024, dtype=dtype),
            outb=L.Linear(None, 1000, initialW=W, bias=bias),
        )
        self._train = True

    def __call__(self, x, t):
        return GoogLeNetBN.__call__(self, F.cast(x, self.dtype), t)

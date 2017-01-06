import numpy as np

import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L


class Alex(chainer.Chain):

    """Single-GPU AlexNet without partition toward the channel axis."""

    insize = 227

    def __init__(self):
        super(Alex, self).__init__(
            conv1=L.Convolution2D(None,  96, 11, stride=4),
            conv2=L.Convolution2D(None, 256,  5, pad=2),
            conv3=L.Convolution2D(None, 384,  3, pad=1),
            conv4=L.Convolution2D(None, 384,  3, pad=1),
            conv5=L.Convolution2D(None, 256,  3, pad=1),
            fc6=L.Linear(None, 4096),
            fc7=L.Linear(None, 4096),
            fc8=L.Linear(None, 1000),
        )
        self.train = True

    def __call__(self, x, t):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)), train=self.train)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train)
        h = self.fc8(h)

        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss


class AlexFp16(Alex):

    """Single-GPU AlexNet without partition toward the channel axis."""

    insize = 227

    def __init__(self):
        self.dtype = np.float16
        W = initializers.HeNormal(1 / np.sqrt(2), self.dtype)
        bias = initializers.Zero(self.dtype)
        chainer.Chain.__init__(
            self,
            conv1=L.Convolution2D(None, 96, 11,
                                  stride=4, initialW=W, bias=bias),
            conv2=L.Convolution2D(None, 256, 5, pad=2, initialW=W, bias=bias),
            conv3=L.Convolution2D(None, 384, 3, pad=1, initialW=W, bias=bias),
            conv4=L.Convolution2D(None, 384, 3, pad=1, initialW=W, bias=bias),
            conv5=L.Convolution2D(None, 256, 3, pad=1, initialW=W, bias=bias),
            fc6=L.Linear(None, 4096, initialW=W, bias=bias),
            fc7=L.Linear(None, 4096, initialW=W, bias=bias),
            fc8=L.Linear(None, 1000, initialW=W, bias=bias),
        )
        self.train = True

    def __call__(self, x, t):
        return Alex.__call__(self, F.cast(x, self.dtype), t)

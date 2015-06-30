import math

import chainer
import chainer.functions as F


class NIN(chainer.FunctionSet):

    """Network-in-Network example model."""

    insize = 227

    def __init__(self):
        w = math.sqrt(2)  # MSRA scaling
        super(NIN, self).__init__(
            conv1=F.Convolution2D(3,   96, 11, wscale=w, stride=4),
            conv1a=F.Convolution2D(96,   96,  1, wscale=w),
            conv1b=F.Convolution2D(96,   96,  1, wscale=w),
            conv2=F.Convolution2D(96,  256,  5, wscale=w, pad=2),
            conv2a=F.Convolution2D(256,  256,  1, wscale=w),
            conv2b=F.Convolution2D(256,  256,  1, wscale=w),
            conv3=F.Convolution2D(256,  384,  3, wscale=w, pad=1),
            conv3a=F.Convolution2D(384,  384,  1, wscale=w),
            conv3b=F.Convolution2D(384,  384,  1, wscale=w),
            conv4=F.Convolution2D(384, 1024,  3, wscale=w, pad=1),
            conv4a=F.Convolution2D(1024, 1024,  1, wscale=w),
            conv4b=F.Convolution2D(1024, 1000,  1, wscale=w),
        )

    def forward(self, x_data, y_data, train=True):
        x = chainer.Variable(x_data, volatile=not train)
        t = chainer.Variable(y_data, volatile=not train)

        h = F.relu(self.conv1(x))
        h = F.relu(self.conv1a(h))
        h = F.relu(self.conv1b(h))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv2a(h))
        h = F.relu(self.conv2b(h))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv3a(h))
        h = F.relu(self.conv3b(h))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.dropout(h, train=train)
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv4a(h))
        h = F.relu(self.conv4b(h))
        h = F.reshape(F.average_pooling_2d(h, 6), (x_data.shape[0], 1000))
        return F.softmax_cross_entropy(h, t), F.accuracy(h, t)

import math

import chainer
import chainer.functions as F
import chainer.links as L


class NIN(chainer.Chain):

    """Network-in-Network example model."""

    insize = 227

    def __init__(self):
        w = math.sqrt(2)  # MSRA scaling
        super(NIN, self).__init__(
            mlpconv1=L.MLPConvolution2D(
                None, (96, 96, 96), 11, stride=4, wscale=w),
            mlpconv2=L.MLPConvolution2D(
                None, (256, 256, 256), 5, pad=2, wscale=w),
            mlpconv3=L.MLPConvolution2D(
                None, (384, 384, 384), 3, pad=1, wscale=w),
            mlpconv4=L.MLPConvolution2D(
                None, (1024, 1024, 1000), 3, pad=1, wscale=w),
        )
        self.train = True

    def __call__(self, x, t):
        h = F.max_pooling_2d(F.relu(self.mlpconv1(x)), 3, stride=2)
        h = F.max_pooling_2d(F.relu(self.mlpconv2(h)), 3, stride=2)
        h = F.max_pooling_2d(F.relu(self.mlpconv3(h)), 3, stride=2)
        h = self.mlpconv4(F.dropout(h, train=self.train))
        h = F.reshape(F.average_pooling_2d(h, 6), (x.data.shape[0], 1000))

        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss

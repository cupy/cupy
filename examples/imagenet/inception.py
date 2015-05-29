from chainer import Function, FunctionSet, Variable
import chainer.functions as F
from chainer.functions.pooling_2d import AveragePooling2D, MaxPooling2D

class GoogLeNet(FunctionSet):

    insize = 224

    def __init__(self):
        super(GoogLeNet, self).__init__(
            conv1        = F.Convolution2D( 3,  64, 7, stride=2, pad=3),
            conv2_reduce = F.Convolution2D(64,  64, 1),
            conv2        = F.Convolution2D(64, 192, 3, stride=1, pad=1),
            inc3a = F.Inception( 192,  64,  96, 128, 16,  32,  32),
            inc3b = F.Inception( 256, 128, 128, 192, 32,  96,  64),
            inc4a = F.Inception( 480, 192,  96, 208, 16,  48,  64),
            inc4b = F.Inception( 512, 160, 112, 224, 24,  64,  64),
            inc4c = F.Inception( 512, 128, 128, 256, 24,  64,  64),
            inc4d = F.Inception( 512, 112, 144, 288, 32,  64,  64),
            inc4e = F.Inception( 528, 256, 160, 320, 32, 128, 128),
            inc5a = F.Inception( 832, 256, 160, 320, 32, 128, 128),
            inc5b = F.Inception( 832, 384, 192, 384, 48, 128, 128),
            loss3_fc   = F.Linear(1024, 1000),

            loss1_conv = F.Convolution2D(512, 128, 1),
            loss1_fc1  = F.Linear(4*4*128, 1024),
            loss1_fc2  = F.Linear(1024   , 1000),

            loss2_conv = F.Convolution2D(528, 128, 1),
            loss2_fc1 = F.Linear(4*4*128, 1024),
            loss2_fc2 = F.Linear(1024   , 1000)
        )

    def forward(self, x_data, y_data, train=True):
        x = Variable(x_data, volatile=not train)
        t = Variable(y_data, volatile=not train)

        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(F.local_response_normalization(h, n=5), 3, stride=2, pad=1)

        h = F.relu(self.conv2_reduce(h))
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(F.local_response_normalization(h, n=5), 3, stride=2, pad=1)

        h = self.inc3a(h)
        h = self.inc3b(h)
        h = F.max_pooling_2d(h, 3, stride=2, pad=1)
        h = self.inc4a(h)

        if train:
            loss1 = F.average_pooling_2d(h, 5, stride=3)
            loss1 = F.relu(self.loss_conv(loss1))
            loss1 = F.relu(self.loss1_fc1(loss1, 4*4*128))
            loss1 = self.loss1_fc2(loss1)
            loss1 = F.softmax_cross_entropy(loss1, t)

        h = self.inc4b(h)
        h = self.inc4c(h)
        h = self.inc4d(h)

        if train:
            loss2 = F.average_pooling_2d(h, 5, stride=3)
            loss2 = F.relu(self.loss_conv(1))
            loss2 = F.relu(self.loss2_fc1(loss2, 4*4*128))
            loss2 = self.loss2_fc2(loss2)
            loss2 = F.softmax_cross_entropy(loss2, t)

        h = self.inc4e(h)
        h = F.max_pooling_2d(h, 3, stride=2, pad=1)
        h = self.inc5a(h)
        h = self.inc5b(h)

        loss3 = F.dropout(F.average_pooling_2d(h, 7, stride=1). 0.4)
        loss3 = F.loss3_fc(loss3)
        loss3 = F.softmax_cross_entropy(loss3, t)

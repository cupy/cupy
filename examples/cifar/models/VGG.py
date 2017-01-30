from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L


class Block(chainer.Chain):

    """A convolution, batch norm, ReLU block.

    A block in a feedforward network that performs a
    convolution followed by batch normalization followed
    by a ReLU activation.

    For the convolution operation, a square filter size is used.

    Args:
        out_channels (int): The number of output channels.
        ksize (int): The size of the filter is ksize x ksize.
        pad (int): The padding to use for the convolution.

    """

    def __init__(self, out_channels, ksize, pad=1):
        super(Block, self).__init__(
            conv=L.Convolution2D(None, out_channels, ksize, pad=pad,
                                 nobias=True),
            bn=L.BatchNormalization(out_channels)
        )

    def __call__(self, x, train=True):
        h = self.conv(x)
        h = self.bn(h, test=not train)
        return F.relu(h)


class VGG(chainer.Chain):

    """A VGG-style network for very small images.

    This model is based on the VGG-style model from
    http://torch.ch/blog/2015/07/30/cifar.html
    which is based on the network architecture from the paper:
    https://arxiv.org/pdf/1409.1556v6.pdf

    This model is intended to be used with either RGB or greyscale input
    images that are of size 32x32 pixels, such as those in the CIFAR10
    and CIFAR100 datasets.

    On CIFAR10, it achieves approximately 89% accuracy on the test set with
    no data augmentation.

    On CIFAR100, it achieves approximately 63% accuracy on the test set with
    no data augmentation.

    Args:
        class_labels (int): The number of class labels.

    """

    def __init__(self, class_labels=10):
        super(VGG, self).__init__(
            block1_1=Block(64, 3),
            block1_2=Block(64, 3),
            block2_1=Block(128, 3),
            block2_2=Block(128, 3),
            block3_1=Block(256, 3),
            block3_2=Block(256, 3),
            block3_3=Block(256, 3),
            block4_1=Block(512, 3),
            block4_2=Block(512, 3),
            block4_3=Block(512, 3),
            block5_1=Block(512, 3),
            block5_2=Block(512, 3),
            block5_3=Block(512, 3),
            fc1=L.Linear(None, 512, nobias=True),
            bn_fc1=L.BatchNormalization(512),
            fc2=L.Linear(None, class_labels, nobias=True)
        )
        self.train = True

    def __call__(self, x):
        # 64 channel blocks:
        h = self.block1_1(x, self.train)
        h = F.dropout(h, ratio=0.3, train=self.train)
        h = self.block1_2(h, self.train)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 128 channel blocks:
        h = self.block2_1(h, self.train)
        h = F.dropout(h, ratio=0.4, train=self.train)
        h = self.block2_2(h, self.train)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 256 channel blocks:
        h = self.block3_1(h, self.train)
        h = F.dropout(h, ratio=0.4, train=self.train)
        h = self.block3_2(h, self.train)
        h = F.dropout(h, ratio=0.4, train=self.train)
        h = self.block3_3(h, self.train)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 512 channel blocks:
        h = self.block4_1(h, self.train)
        h = F.dropout(h, ratio=0.4, train=self.train)
        h = self.block4_2(h, self.train)
        h = F.dropout(h, ratio=0.4, train=self.train)
        h = self.block4_3(h, self.train)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 512 channel blocks:
        h = self.block5_1(h, self.train)
        h = F.dropout(h, ratio=0.4, train=self.train)
        h = self.block5_2(h, self.train)
        h = F.dropout(h, ratio=0.4, train=self.train)
        h = self.block5_3(h, self.train)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.dropout(h, ratio=0.5, train=self.train)
        h = self.fc1(h)
        h = self.bn_fc1(h, test=not self.train)
        h = F.relu(h)
        h = F.dropout(h, ratio=0.5, train=self.train)
        return self.fc2(h)

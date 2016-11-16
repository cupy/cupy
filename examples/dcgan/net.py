#!/usr/bin/env python

from __future__ import print_function

import numpy

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L


def add_noise(h, test, sigma=0.2):
    xp = cuda.get_array_module(h.data)
    if test:
        return h
    else:
        return h + sigma * xp.random.randn(*h.data.shape)


class Generator(chainer.Chain):

    def __init__(self, n_hidden, bottom_width=4, ch=512, wscale=0.02):
        self.n_hidden = n_hidden
        self.ch = ch
        self.bottom_width = bottom_width
        w = chainer.initializers.Normal(wscale)
        super(Generator, self).__init__(
            l0=L.Linear(self.n_hidden, bottom_width *
                        bottom_width * ch, initialW=w),
            dc1=L.Deconvolution2D(ch, ch // 2, 4, 2, 1, initialW=w),
            dc2=L.Deconvolution2D(ch // 2, ch // 4, 4, 2, 1, initialW=w),
            dc3=L.Deconvolution2D(ch // 4, ch // 8, 4, 2, 1, initialW=w),
            dc4=L.Deconvolution2D(ch // 8, 3, 3, 1, 1, initialW=w),
            bn0=L.BatchNormalization(bottom_width * bottom_width * ch),
            bn1=L.BatchNormalization(ch // 2),
            bn2=L.BatchNormalization(ch // 4),
            bn3=L.BatchNormalization(ch // 8),
        )

    def make_hidden(self, batchsize):
        return numpy.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1))\
            .astype(numpy.float32)

    def __call__(self, z, test=False):
        h = F.reshape(F.relu(self.bn0(self.l0(z), test=test)), (z.data.shape[
                      0], self.ch, self.bottom_width, self.bottom_width))
        h = F.relu(self.bn1(self.dc1(h), test=test))
        h = F.relu(self.bn2(self.dc2(h), test=test))
        h = F.relu(self.bn3(self.dc3(h), test=test))
        x = F.sigmoid(self.dc4(h))
        return x


class Discriminator(chainer.Chain):

    def __init__(self, bottom_width=4, ch=512, wscale=0.02):
        w = chainer.initializers.Normal(wscale)
        super(Discriminator, self).__init__(
            c0_0=L.Convolution2D(3, ch // 8, 3, 1, 1, initialW=w),
            c0_1=L.Convolution2D(ch // 8, ch // 4, 4, 2, 1, initialW=w),
            c1_0=L.Convolution2D(ch // 4, ch // 4, 3, 1, 1, initialW=w),
            c1_1=L.Convolution2D(ch // 4, ch // 2, 4, 2, 1, initialW=w),
            c2_0=L.Convolution2D(ch // 2, ch // 2, 3, 1, 1, initialW=w),
            c2_1=L.Convolution2D(ch // 2, ch // 1, 4, 2, 1, initialW=w),
            c3_0=L.Convolution2D(ch // 1, ch // 1, 3, 1, 1, initialW=w),
            l4=L.Linear(bottom_width * bottom_width * ch, 1, initialW=w),
            bn0_1=L.BatchNormalization(ch // 4, use_gamma=False),
            bn1_0=L.BatchNormalization(ch // 4, use_gamma=False),
            bn1_1=L.BatchNormalization(ch // 2, use_gamma=False),
            bn2_0=L.BatchNormalization(ch // 2, use_gamma=False),
            bn2_1=L.BatchNormalization(ch // 1, use_gamma=False),
            bn3_0=L.BatchNormalization(ch // 1, use_gamma=False),
        )

    def __call__(self, x, test=False):
        h = add_noise(x, test=test)
        h = F.leaky_relu(add_noise(self.c0_0(h), test=test))
        h = F.leaky_relu(add_noise(self.bn0_1(
            self.c0_1(h), test=test), test=test))
        h = F.leaky_relu(add_noise(self.bn1_0(
            self.c1_0(h), test=test), test=test))
        h = F.leaky_relu(add_noise(self.bn1_1(
            self.c1_1(h), test=test), test=test))
        h = F.leaky_relu(add_noise(self.bn2_0(
            self.c2_0(h), test=test), test=test))
        h = F.leaky_relu(add_noise(self.bn2_1(
            self.c2_1(h), test=test), test=test))
        h = F.leaky_relu(add_noise(self.bn3_0(
            self.c3_0(h), test=test), test=test))
        return self.l4(h)

import unittest

import numpy

import chainer
from chainer import cuda
from chainer.testing import attr
import chainer.training.updaters.multiprocess_parallel_updater as mpu

import copy


class SimpleNet(chainer.Chain):
    insize = 5

    def __init__(self):
        super(SimpleNet, self).__init__(
            conv=chainer.links.Convolution2D(2, 2, 3),
            fc=chainer.links.Linear(18, 2),
        )
        self.train = True

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x, t):
        h = chainer.functions.relu(self.conv(x))
        y = self.fc(h)

        self.loss = chainer.functions.softmax_cross_entropy(y, t)
        self.accuracy = chainer.functions.accuracy(y, t)

        return self.loss


class TestGatherScatter(unittest.TestCase):

    def setUp(self):
        pass

    @attr.gpu
    def test_gather_scatter_grads(self):
        cupy = cuda.cupy
        model0 = SimpleNet()
        model1 = copy.deepcopy(model0)

        model0.to_gpu()
        model1.to_gpu()

        optimizer0 = chainer.optimizers.SGD(lr=1.0)
        optimizer0.setup(model0)

        optimizer1 = chainer.optimizers.SGD(lr=1.0)
        optimizer1.setup(model1)

        bsize = 8

        x = numpy.random.uniform(0, 1, (bsize, 2, 5, 5)).astype(numpy.float32)
        t = numpy.empty(bsize, dtype=numpy.int32)
        for i in range(bsize):
            t[i] = i % 2

        x = chainer.Variable(chainer.cuda.to_gpu(x))
        t = chainer.Variable(chainer.cuda.to_gpu(t))

        loss0 = model0(x, t)
        model1(x, t)

        model0.cleargrads()
        model1.cleargrads()

        loss0.backward()
        gg0 = mpu.gather_grads(model0)
        mpu.scatter_grads(model1, gg0)

        cupy.testing.assert_array_equal(model0.conv.W.grad, model1.conv.W.grad)
        cupy.testing.assert_array_equal(model0.conv.b.grad, model1.conv.b.grad)
        cupy.testing.assert_array_equal(model0.fc.W.grad, model1.fc.W.grad)
        cupy.testing.assert_array_equal(model0.fc.b.grad, model1.fc.b.grad)

        optimizer0.update()
        optimizer1.update()

        cupy.testing.assert_array_equal(model0.conv.W.data, model1.conv.W.data)
        cupy.testing.assert_array_equal(model0.conv.b.data, model1.conv.b.data)
        cupy.testing.assert_array_equal(model0.fc.W.data, model1.fc.W.data)
        cupy.testing.assert_array_equal(model0.fc.b.data, model1.fc.b.data)

    def test_gather_grads_raise_on_cpu(self):
        model = SimpleNet()
        with self.assertRaises(RuntimeError):
            mpu.gather_grads(model)

    @attr.gpu
    def test_gather_scatter_params(self):
        cupy = cuda.cupy
        model0 = SimpleNet()
        model1 = SimpleNet()

        model0.to_gpu()
        model1.to_gpu()

        gp0 = mpu.gather_params(model0)
        mpu.scatter_params(model1, gp0)

        cupy.testing.assert_array_equal(model0.conv.W.data, model1.conv.W.data)
        cupy.testing.assert_array_equal(model0.conv.b.data, model1.conv.b.data)
        cupy.testing.assert_array_equal(model0.fc.W.data, model1.fc.W.data)
        cupy.testing.assert_array_equal(model0.fc.b.data, model1.fc.b.data)

    def test_gather_params_raise_on_cpu(self):
        model = SimpleNet()
        with self.assertRaises(RuntimeError):
            mpu.gather_params(model)

from unittest import TestCase

import numpy

from chainer import cuda, Variable
from chainer.cuda import to_cpu, to_gpu, GPUArray
from chainer.gradient_check import assert_allclose, numerical_grad
from chainer.functions import NegativeSampling
from chainer.functions.negative_sampling import WalkerAlias

cuda.init()

class TestWalkerAlias(TestCase):
    def setUp(self):
        self.ps = [5, 3, 4, 1, 2]
        self.sampler = WalkerAlias(self.ps)

    def check_sample(self):
        counts = numpy.zeros(len(self.ps), numpy.float32)
        for _ in range(1000):
            vs = self.sampler.sample((4, 3))
            numpy.add.at(counts, to_cpu(vs), 1)
        counts /= (1000 * 12)
        counts *= sum(self.ps)
        assert_allclose(self.ps, counts, atol=0.1, rtol=0.1)

    def test_sample_cpu(self):
        self.check_sample()

    def test_sample_gpu(self):
        self.sampler.to_gpu()
        self.check_sample()


class TestNegativeSampling(TestCase):
    def setUp(self):
        self.func = NegativeSampling(3, [10, 5, 2, 5, 2], 2)
        self.x = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        self.t = numpy.array([0, 2])
        self.gy = numpy.random.uniform(-1, 1, (1, 1)).astype(numpy.float32)

    def check_backward(self, x_data, t_data, y_grad, use_cudnn=True):
        x = Variable(x_data)
        t = Variable(t_data)
        y = self.func(x, t)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data, t.data))
        gx, _, gW = numerical_grad(f, (x.data, t.data, func.W), (y.grad,), eps=1e-2)

        assert_allclose(to_cpu(gx), to_cpu(x.grad), atol=1.e-4)
        assert_allclose(to_cpu(gW), to_cpu(func.gW), atol=1.e-4)

    def test_forward_gpu(self):
        x = Variable(self.x)
        t = Variable(self.t)
        self.func._make_samples(self.t)
        y = self.func(x, t)

        self.func.to_gpu()
        y_g = self.func(Variable(to_gpu(self.x)),
                        Variable(to_gpu(self.t)))

        assert_allclose(y.data, y_g.data, atol=1.e-4)


    def test_backward_cpu(self):
        self.check_backward(self.x, self.t, self.gy)

    def test_backward_gpu(self):
        self.func.to_gpu()
        self.check_backward(to_gpu(self.x), to_gpu(self.t), to_gpu(self.gy))


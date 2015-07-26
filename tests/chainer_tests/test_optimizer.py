import unittest

import numpy as np

from chainer import cuda
from chainer import gradient_check
from chainer import optimizer
from chainer import optimizers
from chainer import testing
from chainer.testing import attr


if cuda.available:
    cuda.init()


class TestOptimizerUtility(unittest.TestCase):

    def setUp(self):
        self.x = np.linspace(-1.0, 1.5, num=6).astype(np.float32).reshape(2, 3)
        self.a = np.array(2.0)

    def test_sqnorm_cpu(self):
        # \Sum_{n=0}^{5} (-1.0+0.5n)**2 = 4.75
        self.assertAlmostEqual(optimizer._sqnorm(self.x), 4.75)

    def test_sqnorm_scalar_cpu(self):
        self.assertAlmostEqual(optimizer._sqnorm(self.a), 4)

    @attr.gpu
    def test_sqnorm_gpu(self):
        x = cuda.to_gpu(self.x)
        self.assertAlmostEqual(optimizer._sqnorm(x), 4.75)

    @attr.gpu
    def test_sqnorm_scalar_gpu(self):
        a = cuda.to_gpu(self.a)
        self.assertAlmostEqual(optimizer._sqnorm(a), 4)


class TestOptimizer(unittest.TestCase):

    def setUp(self):
        self.w = np.arange(6, dtype=np.float32).reshape(2, 3)
        self.g = np.arange(3, -3, -1, dtype=np.float32).reshape(2, 3)

    def check_weight_decay(self, w, g):
        decay = 0.2
        expect = w - g - decay * w

        opt = optimizers.SGD(lr=1)
        opt.setup((w, g))
        opt.weight_decay(decay)
        opt.update()

        gradient_check.assert_allclose(expect, w)

    def test_weight_decay_cpu(self):
        self.check_weight_decay(self.w, self.g)

    @attr.gpu
    def test_weight_decay_gpu(self):
        self.check_weight_decay(cuda.to_gpu(self.w), cuda.to_gpu(self.g))


testing.run_module(__name__, __file__)

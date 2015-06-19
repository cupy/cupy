from unittest import TestCase
import numpy as np
from chainer import cuda, Optimizer
from chainer.optimizer import _sqnorm
from chainer.testing import attr

cuda.init()

class TestOptimizerUtility(TestCase):
    def setUp(self):
        self.x = np.linspace(-1.0, 1.5, num=6).astype(np.float32).reshape(2, 3)
        self.a = np.array(2.0)

    def test_sqnorm_cpu(self):
        # \Sum_{n=0}^{5} (-1.0+0.5n)**2 = 4.75
        self.assertAlmostEqual(_sqnorm(self.x), 4.75)

    def test_sqnorm_scalar_cpu(self):
        self.assertAlmostEqual(_sqnorm(self.a), 4)

    @attr.gpu
    def test_sqnorm_gpu(self):
        x = cuda.to_gpu(self.x)
        self.assertAlmostEqual(_sqnorm(x), 4.75)

    @attr.gpu
    def test_sqnorm_scalar_gpu(self):
        a = cuda.to_gpu(self.a)
        self.assertAlmostEqual(_sqnorm(a), 4)

class TestOptimizer(TestCase):
    def setUp(self):
        pass

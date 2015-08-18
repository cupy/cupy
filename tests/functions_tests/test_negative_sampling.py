import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


if cuda.available:
    cuda.init()


class TestNegativeSampling(unittest.TestCase):
    def setUp(self):
        self.func = chainer.functions.NegativeSampling(3, [10, 5, 2, 5, 2], 2)
        self.x = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        self.t = numpy.array([0, 2]).astype(numpy.int32)
        self.gy = numpy.random.uniform(-1, 1, ()).astype(numpy.float32)

    def check_backward(self, x_data, t_data, y_grad, use_cudnn=True):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        y = self.func(x, t)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data, t.data))
        gx, _, gW = gradient_check.numerical_grad(f, (x.data, t.data, func.W),
                                                  (y.grad,), eps=1e-2)

        gradient_check.assert_allclose(cuda.to_cpu(gx), cuda.to_cpu(x.grad),
                                       atol=1.e-4)
        gradient_check.assert_allclose(cuda.to_cpu(gW), cuda.to_cpu(func.gW),
                                       atol=1.e-4)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        x = chainer.Variable(self.x)
        t = chainer.Variable(self.t)
        self.func._make_samples(self.t)
        y = self.func(x, t)

        self.assertEqual(y.data.dtype, numpy.float32)
        self.assertEqual(y.data.shape, ())

        self.func.to_gpu()
        y_g = self.func(chainer.Variable(cuda.to_gpu(self.x)),
                        chainer.Variable(cuda.to_gpu(self.t)))

        self.assertEqual(y_g.data.dtype, numpy.float32)
        self.assertEqual(y_g.data.shape, ())

        gradient_check.assert_allclose(y.data, y_g.data, atol=1.e-4)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.t, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.func.to_gpu()
        self.check_backward(cuda.to_gpu(self.x),
                            cuda.to_gpu(self.t),
                            cuda.to_gpu(self.gy))

    @attr.gpu
    def test_to_cpu(self):
        self.func.to_gpu()
        self.assertTrue(self.func.sampler.use_gpu)
        self.func.to_cpu()
        self.assertFalse(self.func.sampler.use_gpu)


testing.run_module(__name__, __file__)

import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
}))
class TestGaussian(unittest.TestCase):

    def setUp(self):
        self.m = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        self.v = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)

    def check_forward(self, m_data, v_data):
        m = chainer.Variable(m_data)
        v = chainer.Variable(v_data)
        n = functions.gaussian(m, v)

        # Only checks dtype and shape because its result contains noise
        self.assertEqual(n.dtype, numpy.float32)
        self.assertEqual(n.shape, m.shape)

    def test_forward_cpu(self):
        self.check_forward(self.m, self.v)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.m), cuda.to_gpu(self.v))

    def check_backward(self, m_data, v_data, y_grad):
        gradient_check.check_backward(
            functions.Gaussian(), (m_data, v_data), y_grad,
            atol=1e-4, rtol=1e-3)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.m, self.v, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.m),
                            cuda.to_gpu(self.v),
                            cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)

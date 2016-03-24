import unittest

import numpy

import chainer
from chainer import functions
from chainer import gradient_check
from chainer import testing


class TestForget(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (3, 4)).astype(numpy.float32)
        self.y = numpy.random.uniform(-1, 1, (3, 4)).astype(numpy.float32)
        self.gz = numpy.random.uniform(-1, 1, (3, 4)).astype(numpy.float32)

    def check_forward(self, x_data, y_data):
        x = chainer.Variable(x_data)
        y = chainer.Variable(y_data)
        z = functions.forget(lambda x, y: (x + y + x,), x, y)
        gradient_check.assert_allclose(x_data + y_data + x_data, z.data)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.y)

    def check_backward(self, x_data, y_data, gz_data):
        x = chainer.Variable(x_data)
        y = chainer.Variable(y_data)
        z = functions.forget(lambda x, y: (x + y + x,), x, y)
        z.grad = gz_data
        z.backward()

        gradient_check.assert_allclose(x.grad, gz_data * 2)
        gradient_check.assert_allclose(y.grad, gz_data)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.y, self.gz)


class TestForgetError(unittest.TestCase):

    def test_not_callable(self):
        with self.assertRaises(TypeError):
            functions.forget(1)

    def test_invalid_type(self):
        with self.assertRaisesRegexp(RuntimeError, 'int'):
            functions.forget(lambda: 1)

    def test_invalid_tuple_type(self):
        with self.assertRaisesRegexp(RuntimeError, '1-th.*int'):
            functions.forget(lambda: (1,))


testing.run_module(__name__, __file__)

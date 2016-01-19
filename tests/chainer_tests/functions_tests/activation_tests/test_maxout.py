import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer.functions.activation import maxout
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


def _maxout(x, pool_size, axis):
    shape = (x.shape[:axis] + (x.shape[axis] // pool_size, pool_size) +
             x.shape[axis + 1:])
    x = x.reshape(shape)
    return x.max(axis=axis + 1)


@testing.parameterize(
    {'x_shape': (7, 12), 'pool_size': 2, 'axis': 1, 'y_shape': (7, 6)},
    {'x_shape': (7, 12), 'pool_size': 12, 'axis': 1, 'y_shape': (7, 1)},
    {'x_shape': (7, 3, 4), 'pool_size': 7, 'axis': 0, 'y_shape': (1, 3, 4)},
    {'x_shape': (7, 3, 4), 'pool_size': 3, 'axis': 1, 'y_shape': (7, 1, 4)},
    {'x_shape': (7, 3, 4), 'pool_size': 4, 'axis': 2, 'y_shape': (7, 3, 1)},
    {'x_shape': (7, 2, 3, 4), 'pool_size': 2,
     'axis': 3, 'y_shape': (7, 2, 3, 2)},
)
class TestNonparameterizedMaxout(unittest.TestCase):

    def setUp(self):
        x_size = numpy.prod(self.x_shape)
        self.x = numpy.arange(x_size).reshape(
            self.x_shape).astype(numpy.float32)
        numpy.random.shuffle(self.x.ravel())

        self.y = _maxout(self.x, self.pool_size, self.axis)
        self.gy = numpy.random.uniform(
            -1, 1, self.y.shape).astype(numpy.float32)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.maxout(x, self.pool_size, self.axis)
        gradient_check.assert_allclose(self.y, y.data)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            lambda x: functions.maxout(x, self.pool_size, self.axis),
            x_data, y_grad, atol=1e-2)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)

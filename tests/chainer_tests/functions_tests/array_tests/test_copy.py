import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


def _to_gpu(x, device_id):
    if device_id >= 0:
        return cuda.to_gpu(x, device_id)
    else:
        return x


class Copy(unittest.TestCase):

    def setUp(self):
        self.x_data = numpy.random.uniform(
            -1, 1, (10, 5)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (10, 5)).astype(numpy.float32)

    def check_forward(self, src_id, dst_id):
        x_data = _to_gpu(self.x_data, src_id)
        x = chainer.Variable(x_data)

        y = functions.copy(x, dst_id)

        y_data = _to_gpu(y.data, dst_id)
        gradient_check.assert_allclose(self.x_data, y_data, atol=0, rtol=0)

    def check_backward(self, src_id, dst_id):
        x_data = _to_gpu(self.x_data, src_id)
        x = chainer.Variable(x_data)

        y = functions.copy(x, dst_id)
        gy = _to_gpu(self.gy, dst_id)
        y.grad = gy

        y.backward()

        x_grad = x.grad
        if src_id >= 0:
            x_grad = x_grad.get()
        gradient_check.assert_allclose(x_grad, self.gy, atol=0, rtol=0)

    def test_forward_cpu(self):
        self.check_forward(-1, -1)

    def test_backward_cpu(self):
        self.check_backward(-1, -1)

    @attr.gpu
    def test_forward_gpu(self):
        device_id = cuda.Device().id
        self.check_forward(device_id, device_id)

    @attr.gpu
    def test_check_backward_gpu(self):
        device_id = cuda.Device().id
        self.check_forward(device_id, device_id)

    @attr.multi_gpu(2)
    def test_forward_multigpu(self):
        self.check_forward(0, 1)

    @attr.multi_gpu(2)
    def test_backward_multigpu(self):
        self.check_backward(0, 1)


testing.run_module(__name__, __file__)

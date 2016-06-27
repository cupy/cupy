import unittest

import numpy


import chainer
from chainer import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr


def _to_gpu(x, device_id):
    if device_id >= 0:
        return cuda.to_gpu(x, device_id)
    else:
        return x


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class Copy(unittest.TestCase):

    def setUp(self):
        self.x_data = numpy.random.uniform(
            -1, 1, (10, 5)).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, (10, 5)).astype(self.dtype)

    def check_forward(self, src_id, dst_id):
        x_data = _to_gpu(self.x_data, src_id)
        x = chainer.Variable(x_data)
        y = functions.copy(x, dst_id)

        self.assertEqual(self.x_data.dtype, self.dtype)
        numpy.testing.assert_array_equal(self.x_data, cuda.to_cpu(y.data))

    def check_backward(self, src_id, dst_id):
        x_data = _to_gpu(self.x_data, src_id)
        x = chainer.Variable(x_data)

        y = functions.copy(x, dst_id)
        gy = _to_gpu(self.gy, dst_id)
        y.grad = gy

        y.backward()

        x_grad = x.grad
        numpy.testing.assert_array_equal(
            cuda.to_cpu(x_grad), self.gy)

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

    @attr.gpu
    def test_forward_cpu_to_gpu(self):
        device_id = cuda.Device().id
        self.check_forward(-1, device_id)

    @attr.gpu
    def test_backward_cpu_to_gpu(self):
        device_id = cuda.Device().id
        self.check_backward(-1, device_id)

    @attr.gpu
    def test_forward_gpu_to_cpu(self):
        device_id = cuda.Device().id
        self.check_forward(device_id, -1)

    @attr.gpu
    def test_backward_gpu_to_cpu(self):
        device_id = cuda.Device().id
        self.check_backward(device_id, -1)

    @attr.multi_gpu(2)
    def test_forward_multigpu(self):
        self.check_forward(0, 1)

    @attr.multi_gpu(2)
    def test_backward_multigpu(self):
        self.check_backward(0, 1)


testing.run_module(__name__, __file__)

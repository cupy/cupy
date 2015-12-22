import math
import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(
    {'shape': (8, 7), 'normalize': True},
    {'shape': (8, 7), 'normalize': False},
    # too large shape causes int32 -> float64 issue
    {'shape': (65536, 1), 'normalize': False},
)
class TestSigmoidCrossEntropy(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        self.t = numpy.random.randint(-1, 2, self.shape).astype(numpy.int32)

    def check_forward(self, x_data, t_data, use_cudnn=True):
        x_val = chainer.Variable(x_data)
        t_val = chainer.Variable(t_data)
        loss = functions.sigmoid_cross_entropy(x_val, t_val,
                                               use_cudnn, self.normalize)
        self.assertEqual(loss.data.shape, ())
        self.assertEqual(loss.data.dtype, numpy.float32)
        loss_value = float(cuda.to_cpu(loss.data))

        # Compute expected value
        loss_expect = 0
        non_ignore_count = 0
        for i in six.moves.range(self.x.shape[0]):
            for j in six.moves.range(self.x.shape[1]):
                xd, td = self.x[i, j], self.t[i, j]
                if td == -1:
                    continue
                loss_expect -= xd * (td - (xd >= 0)) \
                    - math.log(1 + math.exp(-numpy.abs(xd)))
                non_ignore_count += 1
        if self.normalize:
            loss_expect /= non_ignore_count
        else:
            loss_expect /= self.t.shape[0]
        self.assertAlmostEqual(loss_expect, loss_value, places=5)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.t)

    @attr.cudnn
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t), False)

    def check_backward(self, x_data, t_data, use_cudnn=True):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        loss = functions.sigmoid_cross_entropy(x, t, use_cudnn)
        loss.backward()
        self.assertEqual(None, t.grad)

        # Skip too large case. That requires a long time.
        if self.shape[0] == 65536:
            return

        func = loss.creator
        f = lambda: func.forward((x.data, t.data))
        gx, = gradient_check.numerical_grad(f, (x.data,), (1,), eps=0.01)

        gradient_check.assert_allclose(gx, x.grad)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.t)

    @attr.cudnn
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.t), False)


testing.run_module(__name__, __file__)

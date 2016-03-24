import math
import unittest

import mock
import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class TestSoftmaxCrossEntropy(unittest.TestCase):

    shape = (4, 3)
    backward_atol = 1e-4
    store_forward = True

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        out_shape = (self.shape[0],) + self.shape[2:]
        self.t = numpy.random.randint(0, 3, out_shape).astype(numpy.int32)

    def check_forward(self, x_data, t_data, use_cudnn=True):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        loss = functions.softmax_cross_entropy(
            x, t, use_cudnn=use_cudnn, store_forward=self.store_forward)
        self.assertEqual(loss.data.shape, ())
        self.assertEqual(loss.data.dtype, numpy.float32)
        loss_value = float(cuda.to_cpu(loss.data))

        # Compute expected value
        loss_expect = 0.0
        count = 0
        x = numpy.rollaxis(self.x, 1, self.x.ndim).reshape(
            (self.t.size, self.x.shape[1]))
        t = self.t.reshape(self.t.size)
        for i in six.moves.range(x.shape[0]):
            if t[i] == -1:
                continue
            log_z = numpy.ufunc.reduce(numpy.logaddexp, x[i])
            loss_expect -= (x[i] - log_z)[t[i]]
            count += 1

        if count == 0:
            loss_expect = 0.0
        else:
            loss_expect /= count

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
        gradient_check.check_backward(
            functions.SoftmaxCrossEntropy(
                use_cudnn=use_cudnn, store_forward=self.store_forward),
            (x_data, t_data), None, eps=0.02, atol=self.backward_atol)

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


class TestSoftmaxCrossEntropyRemoveForward(TestSoftmaxCrossEntropy):

    store_forward = False


class TestSoftmaxCrossEntropyUnstable(TestSoftmaxCrossEntropy):

    backward_atol = 1e-3

    def setUp(self):
        self.x = numpy.array([[-1000, 1]], dtype=numpy.float32)
        self.t = numpy.array([0], dtype=numpy.int32)


class TestReplicatedSoftmaxCrossEntropy1(TestSoftmaxCrossEntropy):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (4, 3, 2)).astype(numpy.float32)
        self.t = numpy.random.randint(0, 3, (4, 2)).astype(numpy.int32)


class TestReplicatedSoftmaxCrossEntropy2(TestSoftmaxCrossEntropy):

    def setUp(self):
        self.x = numpy.random.uniform(
            -1, 1, (4, 3, 2, 5)).astype(numpy.float32)
        self.t = numpy.random.randint(0, 3, (4, 2, 5)).astype(numpy.int32)


class TestSoftmaxCrossEntropyWithIgnoreLabel(TestSoftmaxCrossEntropy):

    def setUp(self):
        super(TestSoftmaxCrossEntropyWithIgnoreLabel, self).setUp()
        self.t[2] = -1


class TestSoftmaxCrossEntropyIgnoreAll(TestSoftmaxCrossEntropy):

    def setUp(self):
        super(TestSoftmaxCrossEntropyIgnoreAll, self).setUp()
        self.t[:] = -1


class TestReplicatedSoftmaxCrossEntropy1IgnoreLabel(
        TestReplicatedSoftmaxCrossEntropy1):

    def setUp(self):
        super(TestReplicatedSoftmaxCrossEntropy1IgnoreLabel, self).setUp()
        self.t[0, 1] = -1


class TestReplicatedSoftmaxCrossEntropy2IgnoreLabel(
        TestReplicatedSoftmaxCrossEntropy2):

    def setUp(self):
        super(TestReplicatedSoftmaxCrossEntropy2IgnoreLabel, self).setUp()
        self.t[0, 1, 2] = -1


class TestReplicatedSoftmaxCrossEntropy1IgnoreAll(
        TestReplicatedSoftmaxCrossEntropy1):

    def setUp(self):
        super(TestReplicatedSoftmaxCrossEntropy1IgnoreAll, self).setUp()
        self.t[:] = -1


class TestReplicatedSoftmaxCrossEntropy2IgnoreAll(
        TestReplicatedSoftmaxCrossEntropy2):

    def setUp(self):
        super(TestReplicatedSoftmaxCrossEntropy2IgnoreAll, self).setUp()
        self.t[:] = -1


@testing.parameterize(
    {'t_value': -2, 'valid': False},
    {'t_value': 3,  'valid': False},
    {'t_value': -1, 'valid': True},  # -1 is ignore_label
)
class TestSoftmaxCrossEntropyValueCheck(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 2)).astype(numpy.float32)
        # `0` is required to avoid NaN
        self.t = numpy.array([self.t_value, 0], dtype=numpy.int32)
        self.original_debug = chainer.is_debug()
        chainer.set_debug(True)

    def tearDown(self):
        chainer.set_debug(self.original_debug)

    def check_value_check(self, x_data, t_data, use_cudnn):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)

        if self.valid:
            # Check if it throws nothing
            functions.softmax_cross_entropy(x, t, use_cudnn)
        else:
            with self.assertRaises(ValueError):
                functions.softmax_cross_entropy(x, t, use_cudnn)

    def test_value_check_cpu(self):
        self.check_value_check(self.x, self.t, False)

    @attr.gpu
    def test_value_check_gpu(self):
        self.check_value_check(self.x, self.t, False)

    @attr.cudnn
    def test_value_check_gpu_cudnn(self):
        self.check_value_check(cuda.to_gpu(self.x), cuda.to_gpu(self.t), True)


@testing.parameterize(
    {'use_cudnn': True},
    {'use_cudnn': False},
)
@attr.cudnn
class TestSoftmaxCrossEntropyCudnnCall(unittest.TestCase):

    def setUp(self):
        self.x = cuda.cupy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)
        self.t = cuda.cupy.random.randint(0, 3, (4,)).astype(numpy.int32)

    def forward(self):
        x = chainer.Variable(self.x)
        t = chainer.Variable(self.t)
        return functions.softmax_cross_entropy(x, t, self.use_cudnn)

    @unittest.skipIf(cuda.cudnn_enabled and
                     cuda.cudnn.cudnn.getVersion() < 3000,
                     'Only cudnn ver>=3 supports softmax-log')
    def test_call_cudnn_forward(self):
        with mock.patch('cupy.cudnn.cudnn.softmaxForward') as func:
            self.forward()
            self.assertEqual(func.called, self.use_cudnn)

    # Note that SoftmaxCrossEntropy does not use cudnn on backward


testing.run_module(__name__, __file__)

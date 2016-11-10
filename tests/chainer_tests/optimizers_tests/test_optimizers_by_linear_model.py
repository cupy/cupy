import unittest

import numpy
import six

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import initializers
import chainer.links as L
from chainer import optimizers
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class LinearModel(object):

    UNIT_NUM = 10
    BATCH_SIZE = 32
    EPOCH = 100

    def __init__(self, optimizer, dtype, use_placeholder):
        self.dtype = dtype
        weight = initializers.HeNormal(1 / numpy.sqrt(2), dtype)
        bias = initializers.Constant(0, dtype)
        in_size = None if use_placeholder else self.UNIT_NUM
        self.model = L.Linear(in_size, 2, initialW=weight, initial_bias=bias)

        self.optimizer = optimizer
        # true parameters
        self.w = numpy.random.uniform(
            -1, 1, (self.UNIT_NUM, 1)).astype(dtype)
        self.b = numpy.random.uniform(-1, 1, (1, )).astype(dtype)

    def _train_linear_classifier(self, model, optimizer, gpu):
        def _make_label(x):
            a = (numpy.dot(x, self.w) + self.b).reshape((self.BATCH_SIZE, ))
            t = numpy.empty_like(a).astype(numpy.int32)
            t[a >= 0] = 0
            t[a < 0] = 1
            return t

        def _make_dataset(batch_size, unit_num, gpu, dtype):
            x_data = numpy.random.uniform(
                -1, 1, (batch_size, unit_num)).astype(dtype)
            t_data = _make_label(x_data)
            if gpu:
                x_data = cuda.to_gpu(x_data)
                t_data = cuda.to_gpu(t_data)
            x = chainer.Variable(x_data)
            t = chainer.Variable(t_data)
            return x, t

        for _ in six.moves.range(self.EPOCH):
            x, t = _make_dataset(self.BATCH_SIZE, self.UNIT_NUM, gpu,
                                 self.dtype)
            model.cleargrads()
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            loss.backward()
            optimizer.update()

        x_test, t_test = _make_dataset(self.BATCH_SIZE, self.UNIT_NUM, gpu,
                                       self.dtype)
        y_test = model(x_test)
        return F.accuracy(y_test, t_test)

    def accuracy_cpu(self):
        self.optimizer.setup(self.model)
        return self._train_linear_classifier(self.model, self.optimizer, False)

    def accuracy_gpu(self, device=None):
        model = self.model
        optimizer = self.optimizer
        model.to_gpu(device=device)
        optimizer.setup(model)
        with cuda.get_device(device):
            return self._train_linear_classifier(model, optimizer, True)


class OptimizerTestBase(object):

    def create(self):
        raise NotImplementedError()

    def setUp(self):
        self.model = LinearModel(self.create(), self.dtype,
                                 self.use_placeholder)

    @condition.retry(10)
    def test_linear_model_cpu(self):
        self.assertGreater(self.model.accuracy_cpu().data, 0.9)

    @attr.gpu
    @condition.retry(10)
    def test_linear_model_gpu(self):
        self.assertGreater(cuda.to_cpu(self.model.accuracy_gpu().data), 0.9)

    @attr.multi_gpu(2)
    @condition.retry(10)
    def test_linear_model_multi_gpu(self):
        with cuda.Device(0):
            self.assertGreater(
                cuda.to_cpu(self.model.accuracy_gpu(1).data), 0.9)

    @attr.multi_gpu(2)
    def test_model_setup_multi_gpu(self):
        with cuda.Device(0):
            model = self.model.model
            optimizer = self.model.optimizer
            model.to_gpu(1)
            optimizer.setup(model)
        for name, param in optimizer.target.namedparams():
            for v in six.itervalues(optimizer._states[name]):
                self.assertEqual(int(param.data.device), int(v.device))

    def test_initialize(self):
        model = self.model.model
        assert isinstance(model, chainer.Link)
        optimizer = self.create()
        optimizer.setup(model)

        msg = 'optimization target must be a link'
        with six.assertRaisesRegex(self, TypeError, msg):
            optimizer.setup('xxx')


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'use_placeholder': [False, True],
}))
class TestAdaDelta(OptimizerTestBase, unittest.TestCase):

    def create(self):
        return optimizers.AdaDelta(eps=1e-5)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'use_placeholder': [False, True],
}))
class TestAdaGrad(OptimizerTestBase, unittest.TestCase):

    def create(self):
        return optimizers.AdaGrad(0.1)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'use_placeholder': [False, True],
}))
class TestAdam(OptimizerTestBase, unittest.TestCase):

    def create(self):
        return optimizers.Adam(0.05)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'use_placeholder': [False, True],
}))
class TestMomentumSGD(OptimizerTestBase, unittest.TestCase):

    def create(self):
        return optimizers.MomentumSGD(0.1)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'use_placeholder': [False, True],
}))
class NesterovAG(OptimizerTestBase, unittest.TestCase):

    def create(self):
        return optimizers.NesterovAG(0.1)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'use_placeholder': [False, True],
}))
class TestRMSprop(OptimizerTestBase, unittest.TestCase):

    def create(self):
        return optimizers.RMSprop(0.1)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'use_placeholder': [False, True],
}))
class TestRMSpropGraves(OptimizerTestBase, unittest.TestCase):

    def create(self):
        return optimizers.RMSpropGraves(0.1)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'use_placeholder': [False, True],
}))
class TestSGD(OptimizerTestBase, unittest.TestCase):

    def create(self):
        return optimizers.SGD(0.1)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'use_placeholder': [False, True],
}))
class TestSMORMS3(OptimizerTestBase, unittest.TestCase):

    def create(self):
        return optimizers.SMORMS3(0.1)


testing.run_module(__name__, __file__)

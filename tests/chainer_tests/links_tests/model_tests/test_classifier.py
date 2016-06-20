import unittest

import numpy

import chainer
from chainer.functions.evaluation import accuracy
from chainer import links
from chainer import testing
from chainer.testing import attr


class MockLink(chainer.Link):

    def __call__(self, *xs):
        return xs[0]


@testing.parameterize(*testing.product({
    'compute_accuracy': [True, False],
    'x_num': [1, 2]
}))
class TestClassifier(unittest.TestCase):

    def setUp(self):
        self.link = links.Classifier(MockLink())
        self.link.compute_accuracy = self.compute_accuracy

        self.x = numpy.random.uniform(-1, 1, (5, 10)).astype(numpy.float32)
        self.t = numpy.random.randint(3, size=(5)).astype(numpy.int32)

    def check_call(self):
        xp = self.link.xp
        x = chainer.Variable(xp.asarray(self.x))
        t = chainer.Variable(xp.asarray(self.t))
        if self.x_num == 1:
            loss = self.link(x, t)
        elif self.x_num == 2:
            x_ = chainer.Variable(xp.asarray(self.x.copy()))
            loss = self.link(x, x_, t)

        self.assertTrue(hasattr(self.link, 'y'))
        self.assertIsNotNone(self.link.y)

        self.assertTrue(hasattr(self.link, 'loss'))
        xp.testing.assert_allclose(self.link.loss.data, loss.data)

        self.assertTrue(hasattr(self.link, 'accuracy'))
        if self.compute_accuracy:
            self.assertIsNotNone(self.link.accuracy)
        else:
            self.assertIsNone(self.link.accuracy)

    def test_call_cpu(self):
        self.check_call()

    @attr.gpu
    def test_call_gpu(self):
        self.link.to_gpu()
        self.check_call()


class TestInvalidArgument(unittest.TestCase):

    def setUp(self):
        self.link = links.Classifier(MockLink())
        self.x = numpy.random.uniform(-1, 1, (5, 10)).astype(numpy.float32)
        self.debug = chainer.is_debug()
        chainer.set_debug(True)

    def tearDown(self):
        chainer.set_debug(self.debug)

    def check_invalid_argument(self):
        with self.assertRaises(AssertionError):
            x = chainer.Variable(self.link.xp.asarray(self.x))
            self.link(x)

    def test_invalid_argument_cpu(self):
        self.check_invalid_argument()

    @attr.gpu
    def test_invalid_argument_gpu(self):
        self.link.to_gpu()
        self.check_invalid_argument()


testing.run_module(__name__, __file__)

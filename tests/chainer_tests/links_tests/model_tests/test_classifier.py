import unittest


import mock
import numpy

import chainer
from chainer import functions
from chainer import links
from chainer import testing
from chainer.testing import attr


# testing.parameterize takes a list of dictionaries.
# Currently, we cannot set a function to the value of the dictionaries.
# As a workaround, we wrap the function and invoke it in __call__ method.
# See issue #1337 for detail.
class AccuracyWithIgnoreLabel(object):

    def __call__(self, y, t):
        return functions.accuracy(y, t, ignore_label=1)


@testing.parameterize(*testing.product({
    'accfun': [AccuracyWithIgnoreLabel(), None],
    'compute_accuracy': [True, False],
    'x_num': [1, 2]
}))
class TestClassifier(unittest.TestCase):

    def setUp(self):
        if self.accfun is None:
            self.link = links.Classifier(chainer.Link())
        else:
            self.link = links.Classifier(chainer.Link(),
                                         accfun=self.accfun)
        self.link.compute_accuracy = self.compute_accuracy

        self.x = numpy.random.uniform(-1, 1, (5, 10)).astype(numpy.float32)
        self.t = numpy.random.randint(3, size=5).astype(numpy.int32)

    def check_call(self):
        xp = self.link.xp

        y = chainer.Variable(xp.random.uniform(
            -1, 1, (5, 7)).astype(numpy.float32))
        self.link.predictor = mock.MagicMock(return_value=y)

        x = chainer.Variable(xp.asarray(self.x))
        t = chainer.Variable(xp.asarray(self.t))
        if self.x_num == 1:
            loss = self.link(x, t)
            self.link.predictor.assert_called_with(x)
        elif self.x_num == 2:
            x_ = chainer.Variable(xp.asarray(self.x.copy()))
            loss = self.link(x, x_, t)
            self.link.predictor.assert_called_with(x, x_)

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
        self.link = links.Classifier(links.Linear(10, 3))
        self.x = numpy.random.uniform(-1, 1, (5, 10)).astype(numpy.float32)

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

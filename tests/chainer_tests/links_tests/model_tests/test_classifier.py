import unittest

import numpy

import chainer
from chainer.functions.evaluation import accuracy
from chainer import links
from chainer import testing
from chainer.testing import attr


@testing.parameterize(
    {'compute_accuracy': True},
    {'compute_accuracy': False}
)
class TestClassifier(unittest.TestCase):

    def setUp(self):
        predictor = links.Linear(10, 3)
        self.link = links.Classifier(predictor)
        self.link.compute_accuracy = self.compute_accuracy

        self.x = numpy.random.uniform(-1, 1, (5, 10)).astype(numpy.float32)
        self.t = numpy.random.randint(3, size=(5)).astype(numpy.int32)

    def check_call(self):
        xp = self.link.xp
        x = chainer.Variable(xp.asarray(self.x))
        t = chainer.Variable(xp.asarray(self.t))
        loss = self.link(x, t)

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


testing.run_module(__name__, __file__)

import unittest

import numpy

import chainer
from chainer import link
from chainer import links
from chainer import testing
from chainer.testing import attr


class MockPredictor(link.Chain):

    def __init__(self, return_shape):
        self.return_shape = return_shape
        super(MockPredictor, self).__init__()

    def __call__(self, *xs):
        batchsize = len(xs[0].data)
        y_shape = (batchsize,) + self.return_shape
        y = self.xp.empty(y_shape, dtype=numpy.float32)
        return chainer.Variable(y)


class TestClassifier(unittest.TestCase):

    def setUp(self):
        self.classifier = links.Classifier(MockPredictor((3,)))
        self.x = numpy.random.uniform(-1, 1, (10, 3)).astype(numpy.float32)
        self.t = numpy.random.randint(3, size=(10,)).astype(numpy.int32)

    def check_call(self):
        xp = self.classifier.xp
        x = chainer.Variable(xp.asarray(self.x))
        t = chainer.Variable(xp.asarray(self.t))
        self.classifier(x, t)

    def test_call(self):
        self.check_call()

    @attr.gpu
    def test_gpu(self):
        self.classifier.to_gpu()
        self.check_call()


class TestClassifier2(unittest.TestCase):

    def setUp(self):
        self.classifier = links.Classifier(MockPredictor((3,)))
        self.xs = [numpy.random.uniform(-1, 1, (10, 3)).astype(numpy.float32),
                   numpy.random.uniform(-1, 1, (10, 2)).astype(numpy.float32)]
        self.t = numpy.random.randint(3, size=(10,)).astype(numpy.int32)

    def check_call(self):
        xp = self.classifier.xp
        xs = [chainer.Variable(xp.asarray(x)) for x in self.xs]
        t = chainer.Variable(xp.asarray(self.t))
        self.classifier(*(xs + [t]))

    def test_call(self):
        self.check_call()

    @attr.gpu
    def test_gpu(self):
        self.classifier.to_gpu()
        self.check_call()


testing.run_module(__name__, __file__)

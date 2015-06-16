from unittest import TestCase
import numpy as np

from chainer import cuda, Variable, FunctionSet
from chainer.functions import Linear, softmax_cross_entropy, accuracy
from chainer.optimizers import SGD

cuda.init()

class TestSGD(TestCase):
    def setUp(self):
        self.model = FunctionSet(
            l = Linear(10, 2))
        self.optimizer = SGD()
        self.w = np.random.uniform(-1, 1, (10, 1)).astype(np.float32)
        self.b = np.random.uniform(-1, 1, (1, )).astype(np.float32)
        self.x = np.random.uniform(-1, 1, (32, 10)).astype(np.float32)
        self.x_test = np.random.uniform(-1, 1, (32, 10)).astype(np.float32)

        def _make_label(x):
            a = np.dot(x, self.w) + self.b
            t = np.empty_like(a).astype(np.int32)
            t[a>=0] = 0
            t[a< 0] = 1
            return t.reshape((32, ))

        self.t = _make_label(self.x)
        self.t_test = _make_label(self.x_test)

    def setup_cpu(self):
        self.optimizer.setup(self.model.collect_parameters())

    def setup_gpu(self):
        model.to_gpu()
        optimizer.setup(model.collect_parameters())

    def test_train_linear_classifier(self):
        self.setup_cpu()
        x = Variable(self.x)
        t = Variable(self.t)
        for epoch in xrange(300):
            self.optimizer.zero_grads()
            y = self.model.l(x)
            loss = softmax_cross_entropy(y, t)
            loss.backward()
            self.optimizer.update()
        x_test = Variable(self.x_test)
        t_test = Variable(self.t_test)
        y_test = self.model.l(x_test)
        acc = accuracy(y_test, t_test)
        self.assertGreater(acc.data, 0.8)

from unittest import TestCase
import numpy as np

from chainer import cuda, Variable, FunctionSet
from chainer.functions import Linear, softmax_cross_entropy, accuracy
from chainer.optimizers import SGD

cuda.init()

class TestSGD(TestCase):
    UNIT_NUM = 10
    BATCH_SIZE = 32
    EPOCH = 100

    def _make_label(self, x):
        a = (np.dot(x, self.w) + self.b).reshape((TestSGD.BATCH_SIZE, ))
        t = np.empty_like(a).astype(np.int32)
        t[a>=0] = 0
        t[a< 0] = 1
        return t

    def setUp(self):
        self.model = FunctionSet(
            l = Linear(TestSGD.UNIT_NUM, 2)
        )
        self.optimizer = SGD(0.1)
        self.w      = np.random.uniform(-1, 1, (TestSGD.UNIT_NUM, 1)).astype(np.float32)
        self.b      = np.random.uniform(-1, 1, (1, )).astype(np.float32)
        self.x_test = np.random.uniform(-1, 1, (TestSGD.BATCH_SIZE, 10)).astype(np.float32)
        self.t_test = self._make_label(self.x_test)

    def setup_cpu(self):
        self.optimizer.setup(self.model.collect_parameters())

    def setup_gpu(self):
        model.to_gpu()
        optimizer.setup(model.collect_parameters())

    def test_train_linear_classifier(self):
        self.setup_cpu()
        for epoch in xrange(TestSGD.EPOCH):
            x_data = np.random.uniform(-1, 1, (TestSGD.BATCH_SIZE, TestSGD.UNIT_NUM)).astype(np.float32)
            x = Variable(x_data)
            t = Variable(self._make_label(x_data))

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

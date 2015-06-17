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

    def _make_dataset(self, batch_size, unit_num, gpu):
        def _make_label(x):
            a = (np.dot(x, self.w) + self.b).reshape((TestSGD.BATCH_SIZE, ))
            t = np.empty_like(a).astype(np.int32)
            t[a>=0] = 0
            t[a< 0] = 1
            return t

        x_data = np.random.uniform(-1, 1, (batch_size, unit_num)).astype(np.float32)
        t_data = _make_label(x_data)
        if gpu:
            x_data = cuda.to_gpu(x_data)
            t_data = cuda.to_gpu(t_data)
        x = Variable(x_data)
        t = Variable(t_data)
        return x, t

    def setUp(self):
        self.model = FunctionSet(
            l = Linear(TestSGD.UNIT_NUM, 2)
        )
        self.optimizer = SGD(0.1)
        self.w      = np.random.uniform(-1, 1, (TestSGD.UNIT_NUM, 1)).astype(np.float32)
        self.b      = np.random.uniform(-1, 1, (1, )).astype(np.float32)

    def check_train_linear_classifier(self, model, optimizer, gpu):
        for epoch in xrange(TestSGD.EPOCH):
            x, t = self._make_dataset(TestSGD.BATCH_SIZE, TestSGD.UNIT_NUM, gpu)
            optimizer.zero_grads()
            y = model.l(x)
            loss = softmax_cross_entropy(y, t)
            loss.backward()
            optimizer.update()

        x_test, t_test = self._make_dataset(TestSGD.BATCH_SIZE, 10, gpu)
        y_test = model.l(x_test)
        acc = accuracy(y_test, t_test)
        self.assertGreater(cuda.to_cpu(acc.data), 0.8)

    def test_train_linear_classifier_cpu(self):
        self.optimizer.setup(self.model.collect_parameters())
        self.check_train_linear_classifier(self.model, self.optimizer, False)

    def test_train_linear_classifier_gpu(self):
        model = self.model
        optimizer = self.optimizer
        model.to_gpu()
        optimizer.setup(model.collect_parameters())
        self.check_train_linear_classifier(model, optimizer, True)

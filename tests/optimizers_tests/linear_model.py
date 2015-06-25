import numpy as np
from chainer import FunctionSet, Variable, cuda
from chainer.functions import Linear, accuracy, softmax_cross_entropy

from six.moves import range

if cuda.available:
    cuda.init()


class LinearModel(object):
    UNIT_NUM = 10
    BATCH_SIZE = 32
    EPOCH = 100

    def __init__(self, optimizer):
        self.model = FunctionSet(
            l=Linear(self.UNIT_NUM, 2)
        )
        self.optimizer = optimizer
        # true parameters
        self.w = np.random.uniform(-1, 1,
                                   (self.UNIT_NUM, 1)).astype(np.float32)
        self.b = np.random.uniform(-1, 1, (1, )).astype(np.float32)

    def _train_linear_classifier(self, model, optimizer, gpu):
        def _make_label(x):
            a = (np.dot(x, self.w) + self.b).reshape((self.BATCH_SIZE, ))
            t = np.empty_like(a).astype(np.int32)
            t[a >= 0] = 0
            t[a < 0] = 1
            return t

        def _make_dataset(batch_size, unit_num, gpu):
            x_data = np.random.uniform(-1, 1,
                                       (batch_size, unit_num)).astype(np.float32)
            t_data = _make_label(x_data)
            if gpu:
                x_data = cuda.to_gpu(x_data)
                t_data = cuda.to_gpu(t_data)
            x = Variable(x_data)
            t = Variable(t_data)
            return x, t

        for epoch in range(self.EPOCH):
            x, t = _make_dataset(self.BATCH_SIZE, self.UNIT_NUM, gpu)
            optimizer.zero_grads()
            y = model.l(x)
            loss = softmax_cross_entropy(y, t)
            loss.backward()
            optimizer.update()

        x_test, t_test = _make_dataset(self.BATCH_SIZE, self.UNIT_NUM, gpu)
        y_test = model.l(x_test)
        return accuracy(y_test, t_test)

    def _accuracy_cpu(self):
        self.optimizer.setup(self.model.collect_parameters())
        return self._train_linear_classifier(self.model, self.optimizer, False)

    def _accuracy_gpu(self):
        model = self.model
        optimizer = self.optimizer
        model.to_gpu()
        optimizer.setup(model.collect_parameters())
        return self._train_linear_classifier(model, optimizer, True)

    def accuracy(self, gpu):
        if gpu:
            return cuda.to_cpu(self._accuracy_gpu().data)
        else:
            return self._accuracy_cpu().data

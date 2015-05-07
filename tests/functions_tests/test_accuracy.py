from unittest import TestCase
import numpy
from chainer import cuda, Variable
from chainer.cuda import to_cpu, to_gpu
from chainer.gradient_check import assert_allclose
from chainer.functions import accuracy

cuda.init()

class TestAccuracy(TestCase):
    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (10, 3)).astype(numpy.float32)
        self.t = numpy.random.randint(3, size=(10,)).astype(numpy.int32)

    def check_forward(self, x_data, t_data):
        x = Variable(x_data)
        t = Variable(t_data)
        y = accuracy(x, t)

        count = 0
        for i in xrange(self.t.size):
            pred = self.x[i].argmax()
            if pred == self.t[i]:
                count += 1

        expected = float(count) / self.t.size
        assert_allclose(expected, to_cpu(y.data))

    def test_forward_cpu(self):
        self.check_forward(self.x, self.t)

    def test_forward_gpu(self):
        self.check_forward(to_gpu(self.x), to_gpu(self.t))

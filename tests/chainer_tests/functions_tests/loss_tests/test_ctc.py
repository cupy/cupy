import math
import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class TestCTC(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (4, 2, 3)).astype(numpy.float32)
        self.t = numpy.array([[0, 1], [1, 0]]).astype(numpy.int32)
        self.l = numpy.array([[2, 0, 2, 1, 2],
                              [2, 1, 2, 0, 2]]).astype(numpy.int32)
        self.blank_symbol = 2
        self.x_length = numpy.full((len(self.x[0]),), len(self.x), dtype='i')
        self.l_length = numpy.full((len(self.t),), len(self.t[0]), dtype='i')
        self.use_length = True

    # recursive forward computation.
    def alpha(self, x, l, t, u):
        if u < 0:
            return 0.0
        if t == 0:
            if u == 0:
                return x[0][self.blank_symbol]
            elif u == 1:
                return x[0][l[1]]
            else:
                return 0.0
        elif l[u] == self.blank_symbol or l[u] == l[u - 2]:
            return (x[t][l[u]] *
                    (self.alpha(x, l, t - 1, u - 1) +
                     self.alpha(x, l, t - 1, u)))
        else:
            return (x[t][l[u]] *
                    (self.alpha(x, l, t - 1, u - 2) +
                     self.alpha(x, l, t - 1, u - 1) +
                     self.alpha(x, l, t - 1, u)))

    def check_forward(self, t_data, xs_data, l_length, x_length):
        x = tuple(chainer.Variable(x_data) for x_data in xs_data)
        t = chainer.Variable(t_data)

        args = (x, t, self.blank_symbol)
        if self.use_length:
            args += (chainer.Variable(x_length), chainer.Variable(l_length))
        loss = functions.connectionist_temporal_classification(*args)
        loss_value = float(loss.data)

        # compute expected value by recursive computation.
        xp = cuda.get_array_module(self.x)
        xt = xp.swapaxes(self.x, 0, 1)
        for b in range(xt.shape[0]):
            for t in range(xt.shape[1]):
                xt[b][t] = numpy.exp(xt[b][t]) / numpy.sum(numpy.exp(xt[b][t]))
        loss_expect = 0
        batch_size = xt.shape[0]
        path_length = 2 * l_length + 1
        for xtb, lb, xlb, plb in zip(xt, self.l, x_length, path_length):
            loss_expect += -math.log(
                self.alpha(xtb, lb, int(xlb - 1), int(plb - 1)) +
                self.alpha(xtb, lb, int(xlb - 1), int(plb - 2)))
        loss_expect /= batch_size
        self.assertAlmostEqual(loss_expect, loss_value, places=5)

    def test_forward_cpu(self):
        self.check_forward(self.t, tuple(self.x),
                           self.l_length, self.x_length)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.t),
                           tuple(cuda.to_gpu(x_data) for x_data in self.x),
                           cuda.to_gpu(self.l_length),
                           cuda.to_gpu(self.x_length))

    # expected value(via numerical differentiation) from t_data
    def check_backward(self, t_data, xs_data, l_length, x_length):
        gradient_check.check_backward(
            functions.ConnectionistTemporalClassification(self.blank_symbol),
            (x_length, l_length, t_data) + xs_data, None, eps=1e-2, atol=1e-4)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.t, tuple(self.x),
                            self.l_length, self.x_length)

    @condition.retry(3)
    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.t),
                            tuple(cuda.to_gpu(x_data) for x_data in self.x),
                            cuda.to_gpu(self.l_length),
                            cuda.to_gpu(self.x_length))


class TestCTCWithoutLength(TestCTC):

    def setUp(self):
        super(TestCTCWithoutLength, self).setUp()
        self.use_length = False


class TestCTCWithLabelPadding(TestCTC):

    def setUp(self):
        super(TestCTCWithLabelPadding, self).setUp()
        self.l_length[0] = 1


class TestCTCWithInputPadding(TestCTC):

    def setUp(self):
        super(TestCTCWithInputPadding, self).setUp()
        self.x_length[0] = 3


class TestCTCWithAllPadding(TestCTC):

    def setUp(self):
        super(TestCTCWithAllPadding, self).setUp()
        self.x_length[...] = 3
        self.l_length[...] = 1


class TestCTCBlankSymbol(TestCTC):

    def setUp(self):
        super(TestCTCBlankSymbol, self).setUp()
        self.x = numpy.random.uniform(-1, 1, (4, 2, 4)).astype(numpy.float32)
        self.l = numpy.array([[3, 0, 3, 1, 3],
                              [3, 1, 3, 0, 3]]).astype(numpy.int32)
        self.blank_symbol = 3


class TestCTCUseVolatile(unittest.TestCase):

    def test_volatile(self):
        xs_data = numpy.random.uniform(-1, 1, (4, 2, 3)).astype(numpy.float32)
        t_data = numpy.array([[0, 1], [1, 0]]).astype(numpy.int32)
        x = [chainer.Variable(x_data, volatile=True) for x_data in xs_data]
        t = chainer.Variable(t_data, volatile=True)
        functions.connectionist_temporal_classification(x, t, 2)


class TestCTCError(unittest.TestCase):

    def test_not_iterable(self):
        x = chainer.Variable(numpy.zeros((4, 2, 3), numpy.float32))
        t = chainer.Variable(numpy.zeros((2, 2), numpy.int32))
        with self.assertRaises(TypeError):
            functions.connectionist_temporal_classification(x, t, 0)


testing.run_module(__name__, __file__)

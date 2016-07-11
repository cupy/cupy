import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(
    {'shape': (), 'lengths': [4, 2, 1], 'trans_lengths': [3, 2, 1, 1]},
    {'shape': (3,), 'lengths': [4, 2, 1], 'trans_lengths': [3, 2, 1, 1]},
    {'shape': (), 'lengths': [0, 0], 'trans_lengths': []},
    {'shape': (3,), 'lengths': [4, 2, 0], 'trans_lengths': [2, 2, 1, 1]},
    {'shape': (3,), 'lengths': [], 'trans_lengths': []},
)
class TestTransposeSequence(unittest.TestCase):

    def setUp(self):
        self.xs = [numpy.random.uniform(-1, 1, (length,) + self.shape)
                   for length in self.lengths]
        self.gs = [numpy.random.uniform(-1, 1, (length,) + self.shape)
                   for length in self.trans_lengths]

    def check_forward(self, xs_data):
        xs = [chainer.Variable(x) for x in xs_data]
        ys = functions.transpose_sequence(xs)
        self.assertEqual(len(ys), len(self.trans_lengths))
        for y, l in zip(ys, self.trans_lengths):
            self.assertEqual(len(y.data), l)

        for i, l in enumerate(self.trans_lengths):
            for j in six.moves.range(l):
                testing.assert_allclose(ys[i].data[j], self.xs[j][i])

    def test_forward_cpu(self):
        self.check_forward(self.xs)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward([cuda.to_gpu(x) for x in self.xs])

    def check_backward(self, xs_data, gs_data):
        # In this situation the function returns no result
        if len(self.trans_lengths) == 0:
            return

        def f(*xs):
            return functions.transpose_sequence(xs)

        gradient_check.check_backward(
            f, tuple(xs_data), tuple(gs_data))

    def test_backward_cpu(self):
        self.check_backward(self.xs, self.gs)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(
            [cuda.to_gpu(x) for x in self.xs],
            [cuda.to_gpu(g) for g in self.gs])


testing.run_module(__name__, __file__)

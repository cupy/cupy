import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(
    {'shape': (), 'lengths': [4, 2, 1], 'trans_lengths': [3, 2, 1, 1]},
    {'shape': (3,), 'lengths': [4, 2, 1], 'trans_lengths': [3, 2, 1, 1]},
)
class TestTransposeSequence(unittest.TestCase):

    def setUp(self):
        self.xs = [numpy.random.uniform(-1, 1, (length,) + self.shape)
                   for length in self.lengths]
        self.gs = [numpy.random.uniform(-1, 1, (length,) + self.shape)
                   for length in self.trans_lengths]

    def check_forward(self, xs_data):
        xs = [chainer.Variable(x) for x in xs_data]
        ys = functions.transpose_sequence(*xs)
        self.assertEqual(len(ys), len(self.trans_lengths))
        for y, l in zip(ys, self.trans_lengths):
            self.assertEqual(len(y.data), l)

        zs = functions.transpose_sequence(*ys)
        self.assertEqual(len(xs), len(zs))
        for x, z in zip(xs, zs):
            gradient_check.assert_allclose(x.data, z.data)

    def test_forward_cpu(self):
        self.check_forward(self.xs)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward([cuda.to_gpu(x) for x in self.xs])

    def check_backward(self, xs_data, gs_data):
        gradient_check.check_backward(
            functions.transpose_sequence, tuple(xs_data), tuple(gs_data))

    def test_backward_cpu(self):
        self.check_backward(self.xs, self.gs)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(
            [cuda.to_gpu(x) for x in self.xs],
            [cuda.to_gpu(g) for g in self.gs])



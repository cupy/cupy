import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer.testing import condition


class TestHuffmanTree(unittest.TestCase):

    def test_empty(self):
        with self.assertRaises(ValueError):
            functions.create_huffman_tree({})

    def test_simple(self):
        tree = functions.create_huffman_tree(
            {'x': 8, 'y': 6, 'z': 5, 'w': 4, 'v': 3})
        expect = (('z', 'y'), (('v', 'w'), 'x'))
        self.assertEqual(expect, tree)


class TestBinaryHierarchicalSoftmax(unittest.TestCase):

    def setUp(self):
        tree = ((0, 1), ((2, 3), 4))
        self.func = functions.BinaryHierarchicalSoftmax(3, tree)
        self.x = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        self.t = numpy.array([0, 2])
        self.gy = numpy.random.uniform(-1, 1, (1, 1)).astype(numpy.float32)

        self.W = self.func.W.copy()

    @condition.retry(3)
    def test_sum(self):
        x = numpy.array([[1.0, 2.0, 3.0]], numpy.float32)
        total = 0
        for i in range(5):
            t = numpy.array([i])
            loss, = self.func.forward_cpu((x, t))
            self.assertEqual(loss.dtype, numpy.float32)
            self.assertEqual(loss.shape, ())
            total += numpy.exp(-loss)
        self.assertAlmostEqual(1.0, float(total))

    def check_backward(self, x_data, t_data, y_grad, use_cudnn=True):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        y = self.func(x, t)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data, t.data))
        gx, _, gW = gradient_check.numerical_grad(
            f, (x.data, t.data, func.W), (y.grad,), eps=1e-2)

        gradient_check.assert_allclose(cuda.to_cpu(gx), cuda.to_cpu(x.grad))
        gradient_check.assert_allclose(cuda.to_cpu(gW), cuda.to_cpu(func.gW))

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.t, self.gy)

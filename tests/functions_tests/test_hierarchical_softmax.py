from unittest import TestCase

import numpy

from chainer import Variable
from chainer.cuda import to_cpu
from chainer.gradient_check import assert_allclose, numerical_grad
from chainer.functions import BinaryHierarchicalSoftmax, create_huffman_tree

class TestHuffmanTree(TestCase):
    def test_empty(self):
        with self.assertRaises(ValueError):
            create_huffman_tree({})

    def test_simple(self):
        tree = create_huffman_tree({'x': 8, 'y': 6, 'z': 5, 'w': 4, 'v': 3})
        expect = (('z', 'y'), (('v', 'w'), 'x'))
        self.assertEqual(expect, tree)

class TestBinaryHierarchicalSoftmax(TestCase):
    def setUp(self):
        tree = ((0, 1), ((2, 3), 4))
        self.func = BinaryHierarchicalSoftmax(3, tree)
        self.x = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        self.t = numpy.array([0, 2])
        self.gy = numpy.random.uniform(-1, 1, (1, 1)).astype(numpy.float32)

        self.W  = self.func.W.copy()

    def test_sum(self):
        x = numpy.array([[1.0, 2.0, 3.0]])
        total = 0
        for i in range(5):
            t = numpy.array([i])
            loss, = self.func.forward_cpu((x, t))
            total += numpy.exp(-loss)
        self.assertAlmostEqual(1.0, float(total))

    def check_backward(self, x_data, t_data, y_grad, use_cudnn=True):
        x = Variable(x_data)
        t = Variable(t_data)
        y = self.func(x, t)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data, t.data))
        gx, _, gW = numerical_grad(f, (x.data, t.data, func.W), (y.grad,), eps=1e-2)

        assert_allclose(to_cpu(gx), to_cpu(x.grad))
        assert_allclose(to_cpu(gW), to_cpu(func.gW))

    def test_backward_cpu(self):
        self.check_backward(self.x, self.t, self.gy)

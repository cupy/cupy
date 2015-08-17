import copy
import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


if cuda.available:
    cuda.init()


class TestHuffmanTree(unittest.TestCase):

    def test_empty(self):
        with self.assertRaises(ValueError):
            functions.create_huffman_tree({})

    def test_simple(self):
        tree = functions.create_huffman_tree(
            {'x': 8, 'y': 6, 'z': 5, 'w': 4, 'v': 3})
        expect = (('z', 'y'), (('v', 'w'), 'x'))
        self.assertEqual(expect, tree)

    def test_same_count(self):
        tree = functions.create_huffman_tree(
            {'x': 1, 'y': 2, 'z': 3})
        # Order of the same items are not defined.
        self.assertTrue((('x', 'y'), 'z') == tree or
                        ('z', ('x', 'y')) == tree)


class TestBinaryHierarchicalSoftmax(unittest.TestCase):

    def setUp(self):
        tree = ((0, 1), ((2, 3), 4))
        self.func = functions.BinaryHierarchicalSoftmax(3, tree)
        self.x = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        self.t = numpy.array([0, 2]).astype(numpy.int32)
        self.gy = numpy.random.uniform(-1, 1, ()).astype(numpy.float32)

        self.W = self.func.W.copy()

    def check_sum(self, x, gpu=False):
        total = 0
        for i in range(5):
            t = numpy.array([i])
            if gpu:
                t = cuda.to_gpu(t)
            loss, = self.func.forward((x, t))
            self.assertEqual(loss.dtype, numpy.float32)
            self.assertEqual(loss.shape, ())
            total += numpy.exp(-cuda.to_cpu(loss))
        self.assertAlmostEqual(1.0, float(total), delta=1.0e-5)

    @condition.retry(3)
    def test_sum_cpu(self):
        x = numpy.array([[1.0, 2.0, 3.0]], numpy.float32)
        self.check_sum(x)

    @attr.gpu
    @condition.retry(3)
    def test_sum_gpu(self):
        x = numpy.array([[1.0, 2.0, 3.0]], numpy.float32)
        self.func.to_gpu()
        self.check_sum(cuda.to_gpu(x), gpu=True)

    @attr.gpu
    def test_forward(self):
        # TODO(unno): We need to test return values of forward function.
        cpu_loss, = self.func.forward((self.x, self.t))
        self.func.to_gpu()
        gpu_loss, = self.func.forward((cuda.to_gpu(self.x),
                                       cuda.to_gpu(self.t)))
        gradient_check.assert_allclose(
            cpu_loss, cuda.to_cpu(gpu_loss))

    def check_backward(self, x_data, t_data, y_grad):
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

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.func.to_gpu()
        self.check_backward(cuda.to_gpu(self.x),
                            cuda.to_gpu(self.t),
                            cuda.to_gpu(self.gy))

    @attr.gpu
    def test_to_cpu(self):
        f = copy.deepcopy(self.func)
        self.func.to_gpu()
        self.func.to_cpu()

        self.assertTrue((f.begins == self.func.begins).all())
        self.assertTrue((f.paths == self.func.paths).all())
        self.assertTrue((f.codes == self.func.codes).all())


testing.run_module(__name__, __file__)

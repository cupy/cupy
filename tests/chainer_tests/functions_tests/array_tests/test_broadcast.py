import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.utils import type_check


@testing.parameterize(
    {'in_shapes': [(3, 1, 5), (1, 2, 5)], 'out_shape': (3, 2, 5)},
    {'in_shapes': [(3, 2, 5), (5,)],      'out_shape': (3, 2, 5)},
    {'in_shapes': [(3, 2, 5), ()],        'out_shape': (3, 2, 5)},
    {'in_shapes': [(3, 2, 5), (3, 2, 5)], 'out_shape': (3, 2, 5)},
    {'in_shapes': [(), ()],               'out_shape': ()},
    {'in_shapes': [(1, 1, 1), (1,)],      'out_shape': (1, 1, 1)},
    {'in_shapes': [(1, 1, 1), ()],        'out_shape': (1, 1, 1)},
    {'in_shapes': [(3, 2, 5)],            'out_shape': (3, 2, 5)},
    {'in_shapes': [(3, 1, 5), (1, 2, 5), (3, 2, 1)],
     'out_shape': (3, 2, 5)},
)
class TestBroadcast(unittest.TestCase):

    def setUp(self):
        uniform = numpy.random.uniform
        self.data = [uniform(0, 1, shape).astype(numpy.float32)
                     for shape in self.in_shapes]
        self.grads = [uniform(0, 1, self.out_shape).astype(numpy.float32)
                      for _ in range(len(self.in_shapes))]

    def check_forward(self, data):
        xs = [chainer.Variable(x) for x in data]
        bxs = functions.broadcast(*xs)

        # When len(xs) == 1, function returns a Variable object
        if isinstance(bxs, chainer.Variable):
            bxs = (bxs,)

        for bx in bxs:
            self.assertEqual(bx.data.shape, self.out_shape)

    def test_forward_cpu(self):
        self.check_forward(self.data)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward([cuda.to_gpu(x) for x in self.data])

    def check_backward(self, data, grads):
        xs = [chainer.Variable(x) for x in data]
        bxs = functions.broadcast(*xs)

        # When len(xs) == 1, function returns a Variable object
        if isinstance(bxs, chainer.Variable):
            bxs = (bxs,)

        func = bxs[0].creator
        f = lambda: func.forward(data)

        for i, (bx, grad) in enumerate(zip(bxs, grads)):
            bx.grad = grad
            bx.backward()
            gxs = gradient_check.numerical_grad(
                f, data, tuple(bx.grad for bx in bxs))
            gradient_check.assert_allclose(gxs[i], xs[i].grad)

    def test_backward_cpu(self):
        self.check_backward(self.data, self.grads)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward([cuda.to_gpu(x) for x in self.data],
                            [cuda.to_gpu(x) for x in self.grads])


class TestBroadcastTypeError(unittest.TestCase):

    def test_invalid_shape(self):
        x_data = numpy.zeros((3, 2, 5), dtype=numpy.int32)
        y_data = numpy.zeros((1, 3, 4), dtype=numpy.float32)
        x = chainer.Variable(x_data)
        y = chainer.Variable(y_data)

        with self.assertRaises(type_check.InvalidType):
            functions.broadcast(x, y)

    def test_invalid_shape_fill(self):
        x_data = numpy.zeros((3, 2, 5), dtype=numpy.int32)
        y_data = numpy.zeros((4), dtype=numpy.float32)
        x = chainer.Variable(x_data)
        y = chainer.Variable(y_data)

        with self.assertRaises(type_check.InvalidType):
            functions.broadcast(x, y)

    def test_no_args(self):
        with self.assertRaises(type_check.InvalidType):
            functions.broadcast()


testing.run_module(__name__, __file__)

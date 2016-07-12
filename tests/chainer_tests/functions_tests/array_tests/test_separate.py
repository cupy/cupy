import unittest

import numpy

import chainer
from chainer import cuda
from chainer.functions.array import separate
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product_dict(
    [
        {'shape': (2, 3, 4), 'axis': 0},
        {'shape': (2, 3, 4), 'axis': 1},
        {'shape': (2, 3, 4), 'axis': 2},
        {'shape': (2, 3, 4), 'axis': -1},
        {'shape': (2,), 'axis': 0},
        {'shape': (2,), 'axis': -1},
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ]
))
class TestSeparate(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        yshape = list(self.shape)
        del yshape[self.axis]
        self.gys = [numpy.random.uniform(-1, 1, yshape).astype(self.dtype)
                    for _ in range(self.shape[self.axis])]
        self.check_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_backward_options = {
                'eps': 2 ** -5, 'atol': 1e-3, 'rtol': 1e-2}

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        ys = separate.separate(x, self.axis)

        self.assertIsInstance(ys, tuple)
        self.assertEqual(len(ys), self.shape[self.axis])
        for i in range(self.shape[self.axis]):
            expect = self.x.take(i, axis=self.axis)
            testing.assert_allclose(ys[i].data, expect)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, gys_data):
        def f(x):
            return separate.separate(x, self.axis)

        gradient_check.check_backward(
            f, x_data, gys_data, **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gys)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x),
                            [cuda.to_gpu(g) for g in self.gys])


testing.run_module(__name__, __file__)

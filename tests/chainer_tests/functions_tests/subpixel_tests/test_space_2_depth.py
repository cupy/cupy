import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class TestSpace2Depth(unittest.TestCase):

    def setUp(self):
        self.depth = numpy.arange(96).reshape(2, 8, 3, 2).astype(numpy.float32)
        self.rand_array = (numpy.random.randn(96)
                                .reshape(2, 2, 6, 4)
                                .astype(numpy.float32)
                           )
        self.rand_grad_array = (numpy.random.randn(96)
                                .reshape(2, 8, 3, 2)
                                .astype(numpy.float32)
                                )
        self.space = numpy.array([[[[0.,  12.,   1.,  13.],
                                    [24.,  36.,  25.,  37.],
                                    [2.,  14.,   3.,  15.],
                                    [26.,  38.,  27.,  39.],
                                    [4.,  16.,   5.,  17.],
                                    [28.,  40.,  29.,  41.]],
                                   [[6.,  18.,   7.,  19.],
                                    [30.,  42.,  31.,  43.],
                                    [8.,  20.,   9.,  21.],
                                    [32.,  44.,  33.,  45.],
                                    [10.,  22.,  11.,  23.],
                                    [34.,  46.,  35.,  47.]]],
                                  [[[48.,  60.,  49.,  61.],
                                    [72.,  84.,  73.,  85.],
                                    [50.,  62.,  51.,  63.],
                                    [74.,  86.,  75.,  87.],
                                    [52.,  64.,  53.,  65.],
                                    [76.,  88.,  77.,  89.]],
                                   [[54.,  66.,  55.,  67.],
                                    [78.,  90.,  79.,  91.],
                                    [56.,  68.,  57.,  69.],
                                    [80.,  92.,  81.,  93.],
                                    [58.,  70.,  59.,  71.],
                                    [82.,  94.,  83.,  95.]]]]
                                 ).astype(numpy.float32)

        self.r = 2

    def check_forward(self, space_data, depth_data):
        space = chainer.Variable(space_data)
        s2d = functions.space2depth(space, self.r)
        s2d_value = cuda.to_cpu(s2d.data)

        self.assertEqual(s2d_value.dtype, numpy.float32)
        self.assertEqual(s2d_value.shape, (2, 8, 3, 2))

        s2d_expect = depth_data

        testing.assert_allclose(s2d_value, s2d_expect)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.space, self.depth)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.space), cuda.to_gpu(self.depth))

    def check_backward(self, random_array, random_grad_array):
        x = chainer.Variable(random_array)
        y = functions.space2depth(x, 2)
        y.grad = random_grad_array
        y.backward()

        def func():
            return (functions.space2depth(x, 2).data,)
        gx, = gradient_check.numerical_grad(func, (x.data,), (y.grad,))

        testing.assert_allclose(x.grad, gx, rtol=0.0001)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.rand_array, self.rand_grad_array)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.rand_array),
                            cuda.to_gpu(self.rand_grad_array)
                            )


testing.run_module(__name__, __file__)

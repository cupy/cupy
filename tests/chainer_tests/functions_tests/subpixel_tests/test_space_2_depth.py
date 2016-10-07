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
        self.depth = numpy.arange(96).reshape(2, 3, 2, 8).astype(numpy.float32)
        self.depth = self.depth.transpose((0, 3, 1, 2))
        self.rand_array = (numpy.random.randn(96)
                                .reshape(2, 2, 6, 4)
                                .astype(numpy.float32)
                           )
        self.space = numpy.array([[[[0.,   4.,   8.,  12.],
                                    [1.,   5.,   9.,  13.],
                                    [16.,  20.,  24.,  28.],
                                    [17.,  21.,  25.,  29.],
                                    [32.,  36.,  40.,  44.],
                                    [33.,  37.,  41.,  45.]],
                                   [[48.,  52.,  56.,  60.],
                                    [49.,  53.,  57.,  61.],
                                    [64.,  68.,  72.,  76.],
                                    [65.,  69.,  73.,  77.],
                                    [80.,  84.,  88.,  92.],
                                    [81.,  85.,  89.,  93.]]],
                                  [[[2.,   6.,  10.,  14.],
                                    [3.,   7.,  11.,  15.],
                                    [18.,  22.,  26.,  30.],
                                    [19.,  23.,  27.,  31.],
                                    [34.,  38.,  42.,  46.],
                                    [35.,  39.,  43.,  47.]],

                                   [[50.,  54.,  58.,  62.],
                                    [51.,  55.,  59.,  63.],
                                    [66.,  70.,  74.,  78.],
                                    [67.,  71.,  75.,  79.],
                                    [82.,  86.,  90.,  94.],
                                    [83.,  87.,  91.,  95.]]]]
                                 )
        self.space = numpy.transpose(self.space, (1, 2, 3, 0))
        self.space = self.space.astype(numpy.float32)
        self.space = self.space.transpose((0, 3, 1, 2))
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

    def check_backward(self, random_array):
        x = chainer.Variable(random_array)
        y = functions.space2depth(x, 2)
        y.grad = numpy.random.randn(*y.data.shape).astype(numpy.float32)
        y.backward()

        def func():
            return (functions.space2depth(x, 2).data,)
        gx, = gradient_check.numerical_grad(func, (x.data,), (y.grad,))

        testing.assert_allclose(x.grad, gx, rtol=0.0001)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.rand_array)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.rand_array))


testing.run_module(__name__, __file__)

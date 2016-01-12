import unittest

import numpy

import chainer
from chainer import functions
from chainer import links
from chainer import testing


@testing.parameterize(
    {'use_cudnn': True},
    {'use_cudnn': False},
)
class TestMLPConvolution2D(unittest.TestCase):

    def setUp(self):
        self.mlp = links.MLPConvolution2D(
            3, (96, 96, 96), 11,
            activation=functions.sigmoid,
            use_cudnn=self.use_cudnn)
        self.x = numpy.zeros((10, 3, 20, 20), dtype=numpy.float32)

    def test_init(self):
        self.assertIs(self.mlp.activation, functions.sigmoid)

        self.assertEqual(len(self.mlp), 3)
        for i, conv in enumerate(self.mlp):
            self.assertIsInstance(conv, links.Convolution2D)
            self.assertEqual(conv.use_cudnn, self.use_cudnn)
            if i == 0:
                self.assertEqual(conv.W.data.shape, (96, 3, 11, 11))
            else:
                self.assertEqual(conv.W.data.shape, (96, 96, 1, 1))

    def check_call(self, x_data):
        x = chainer.Variable(x_data)
        actual = self.mlp(x)
        act = functions.sigmoid
        expect = self.mlp[2](act(self.mlp[1](act(self.mlp[0](x)))))
        cuda.cupy.testing.assert_array_equal(expect.data, actual.data)

    def test_call_cpu(self):
        self.check_call(self.x)

    @attr.gpu
    def test_call_gpu(self):
        self.mlp.to_gpu()
        self.check_call(cuda.to_gpu(self.x))


testing.run_module(__name__, __file__)

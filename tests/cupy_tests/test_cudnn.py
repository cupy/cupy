import unittest

import mock
import numpy

import cupy
try:
    import cupy.cuda.cudnn as libcudnn
    cudnn_enabled = True
    modes = [
        libcudnn.CUDNN_ACTIVATION_SIGMOID,
        libcudnn.CUDNN_ACTIVATION_RELU,
        libcudnn.CUDNN_ACTIVATION_TANH,
    ]
    import cupy.cudnn
except ImportError:
    cudnn_enabled = False
    modes = []
from cupy import testing


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
    'mode': modes,
}))
@unittest.skipUnless(
    cudnn_enabled and libcudnn.getVersion() >= 3000,
    'cuDNN >= 3.0 is supported')
class TestCudnnActivation(unittest.TestCase):

    def setUp(self):
        self.x = testing.shaped_arange((3, 4), cupy, self.dtype)
        self.y = testing.shaped_arange((3, 4), cupy, self.dtype)
        self.g = testing.shaped_arange((3, 4), cupy, self.dtype)

    def test_activation_forward_version(self):
        if libcudnn.getVersion() >= 4000:
            patch = 'cupy.cuda.cudnn.activationForward_v4'
        else:
            patch = 'cupy.cuda.cudnn.activationForward_v3'
        with mock.patch(patch) as func:
            cupy.cudnn.activation_forward(self.x, self.mode)
            self.assertEqual(func.called, True)

    def test_activation_backward_version(self):
        if libcudnn.getVersion() >= 4000:
            patch = 'cupy.cuda.cudnn.activationBackward_v4'
        else:
            patch = 'cupy.cuda.cudnn.activationBakward_v3'
        with mock.patch(patch) as func:
            cupy.cudnn.activation_backward(self.x, self.y, self.g, self.mode)
            self.assertEqual(func.called, True)

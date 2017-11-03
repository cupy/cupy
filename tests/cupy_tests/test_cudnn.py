import unittest

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
    coef_modes = [
        libcudnn.CUDNN_ACTIVATION_CLIPPED_RELU,
    ]
    if libcudnn.getVersion() >= 6000:
        coef_modes.append(libcudnn.CUDNN_ACTIVATION_ELU)

    from cupy import cudnn
except ImportError:
    cudnn_enabled = False
    modes = []
    coef_modes = []
from cupy import testing


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
    'mode': modes,
}))
@unittest.skipUnless(cudnn_enabled, 'cuDNN is not available')
class TestCudnnActivation(unittest.TestCase):

    def setUp(self):
        self.x = testing.shaped_arange((3, 4), cupy, self.dtype)
        self.y = testing.shaped_arange((3, 4), cupy, self.dtype)
        self.g = testing.shaped_arange((3, 4), cupy, self.dtype)

    def test_activation_forward(self):
        cudnn.activation_forward(self.x, self.mode)

    def test_activation_backward(self):
        cudnn.activation_backward(self.x, self.y, self.g, self.mode)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
    'mode': coef_modes,
}))
@unittest.skipUnless(
    cudnn_enabled and libcudnn.getVersion() >= 5000,
    'cuDNN >= 5.0 is supported')
class TestCudnnActivationCoef(unittest.TestCase):

    def setUp(self):
        self.x = testing.shaped_arange((3, 4), cupy, self.dtype)
        self.y = testing.shaped_arange((3, 4), cupy, self.dtype)
        self.g = testing.shaped_arange((3, 4), cupy, self.dtype)
        self.coef = self.dtype(0.75)

    def test_activation_forward(self):
        cudnn.activation_forward(self.x, self.mode, self.coef)

    def test_activation_backward(self):
        cudnn.activation_backward(self.x, self.y, self.g, self.mode,
                                       self.coef)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
    'ratio': [0.0, 0.1, 0.2, 0.5],
    'seed': [0, 100]
}))
@unittest.skipUnless(
    cudnn_enabled and libcudnn.getVersion() >= 5000,
    'cuDNN >= 5.0 is supported')
class TestCudnnDropout(unittest.TestCase):

    def setUp(self):
        self.x = testing.shaped_arange((3, 4), cupy, self.dtype)
        self.gy = testing.shaped_arange((3, 4), cupy, self.dtype)
        self.states = cudnn.DropoutStates(cudnn.get_handle(), self.seed)

    def test_dropout_forward(self):
        _, y = self.states.forward(cudnn.get_handle(), self.x, self.ratio)
        if self.ratio == 0:
            self.assertTrue(cupy.all(self.x == y))
        else:
            self.assertTrue(cupy.all(self.x != y))

    def test_dropout_backward(self):
        rspace, y = self.states.forward(cudnn.get_handle(), self.x, self.ratio)
        gx = self.states.backward(
            cudnn.get_handle(), self.gy, self.ratio, rspace)

        forward_mask = y / self.x
        backward_mask = gx / self.gy

        # backward_mask must be the same as forward_mask
        self.assertTrue(cupy.all(forward_mask == backward_mask))

    def test_dropout_seed(self):
        handle = cudnn.get_handle()

        # initialize Dropoutstates with the same seed
        states2 = cudnn.DropoutStates(handle, self.seed)

        rspace, y = self.states.forward(handle, self.x, self.ratio)
        rspace2, y2 = states2.forward(handle, self.x, self.ratio)
        # forward results must be the same
        self.assertTrue(cupy.all(y == y2))

        gx = self.states.backward(handle, self.gy, self.ratio, rspace)
        gx2 = states2.backward(handle, self.gy, self.ratio, rspace2)
        # backward results must be the same
        self.assertTrue(cupy.all(gx == gx2))


@testing.parameterize(*(testing.product({
    'tensor_core': ['always', 'auto', 'never'],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'dilate': [1, 2],
    'group': [1, 2],
    'ndim': [2],
    'max_workspace_size': [0, 2 ** 22],
    'auto_tune': [True, False],
    'bias': [True, False],
})))
class TestConvolutionForward(unittest.TestCase):

    def setUp(self):
        ndim = self.ndim
        dtype = self.dtype
        batches = 2
        in_channels_a_group = 3
        out_channels_a_group = 2
        in_channels = in_channels_a_group * self.group
        out_channels = out_channels_a_group * self.group
        ksize = 3
        stride = 2
        pad = ksize // stride * self.dilate
        self.strides = (stride,) * ndim
        self.pads = (pad,) * ndim
        self.dilations = (self.dilate,) * ndim
        self.x = cupy.zeros(
            (batches, in_channels) + (ksize,) * ndim, dtype)
        self.W = cupy.zeros(
            (out_channels, in_channels_a_group) + (ksize,) * ndim, dtype)
        self.b = None
        if self.bias:
            self.b = cupy.zeros((out_channels,), dtype)

        self.y = cupy.ones((batches, out_channels) + (2,) * ndim, dtype)

        version = libcudnn.getVersion()
        self.err = None
        if ((self.dilate > 1 and version < 6000) or
                (self.group > 1 and version < 7000)):
            self.err = ValueError
        elif ndim > 2 and self.dilate > 1:
            self.err = libcudnn.CuDNNError

    def test_call(self):
        args = (self.x, self.W, self.b, self.y,
                self.pads, self.strides, self.dilations,
                self.group, self.max_workspace_size, self.auto_tune,
                self.tensor_core)
        if self.err is None:
            cudnn.convolution_forward(*args)
            self.assertTrue((self.y == 0).all())
        else:
            with self.assertRaises(self.err):
                cudnn.convolution_forward(*args)


@testing.parameterize(*(testing.product({
    'tensor_core': ['always', 'auto', 'never'],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'dilate': [1, 2],
    'group': [1, 2],
    'ndim': [2, 3],
    'max_workspace_size': [0, 2 ** 22],
    'auto_tune': [True, False],
    'deterministic': [True, False],
})))
class TestConvolutionBackwardFilter(unittest.TestCase):

    def setUp(self):
        ndim = self.ndim
        dtype = self.dtype
        batches = 2
        in_channels_a_group = 3
        out_channels_a_group = 2
        in_channels = in_channels_a_group * self.group
        out_channels = out_channels_a_group * self.group
        ksize = 3
        stride = 2
        pad = ksize // stride * self.dilate
        self.strides = (stride,) * ndim
        self.pads = (pad,) * ndim
        self.dilations = (self.dilate,) * ndim
        self.x = cupy.zeros(
            (batches, in_channels) + (ksize,) * ndim, dtype)
        self.gy = cupy.zeros((batches, out_channels) + (2,) * ndim, dtype)

        self.gW = cupy.ones(
            (out_channels, in_channels_a_group) + (ksize,) * ndim, dtype)

        version = libcudnn.getVersion()
        deterministic = self.deterministic
        self.err = None
        if ((self.dilate > 1 and version < 6000) or
                (self.group > 1 and version < 7000)):
            self.err = ValueError
        elif ((self.dilate > 1 and deterministic and version < 7000) or
                (ndim > 2 and deterministic and version < 6000) or
                (ndim > 2 and deterministic and self.dtype == numpy.float64)):
            self.err = libcudnn.CuDNNError

    def test_call(self):
        args = (self.x, self.gy, self.gW,
                self.pads, self.strides, self.dilations,
                self.group, self.max_workspace_size, self.deterministic,
                self.auto_tune, self.tensor_core)
        if self.err is None:
            cudnn.convolution_backward_filter(*args)
            self.assertTrue((self.gW == 0).all())
        else:
            with self.assertRaises(self.err):
                cudnn.convolution_backward_filter(*args)


@testing.parameterize(*(testing.product({
    'tensor_core': ['always', 'auto', 'never'],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'dilate': [1, 2],
    'group': [1, 2],
    'ndim': [2, 3],
    'max_workspace_size': [0, 2 ** 22],
    'auto_tune': [True, False],
    'deterministic': [True, False],
    'bias': [True, False],
})))
class TestConvolutionBackwardData(unittest.TestCase):

    def setUp(self):
        ndim = self.ndim
        dtype = self.dtype
        batches = 2
        in_channels_a_group = 3
        out_channels_a_group = 2
        in_channels = in_channels_a_group * self.group
        out_channels = out_channels_a_group * self.group
        ksize = 3
        stride = 2
        pad = ksize // stride * self.dilate
        self.strides = (stride,) * ndim
        self.pads = (pad,) * ndim
        self.dilations = (self.dilate,) * ndim
        self.W = cupy.zeros(
            (out_channels, in_channels_a_group) + (ksize,) * ndim, dtype)
        self.gy = cupy.zeros((batches, out_channels) + (2,) * ndim, dtype)
        self.b = None
        if self.bias:
            self.b = cupy.zeros((in_channels,), dtype)

        self.gx = cupy.ones(
            (batches, in_channels) + (ksize,) * ndim, dtype)

        version = libcudnn.getVersion()
        deterministic = self.deterministic
        self.err = None
        if ((self.dilate > 1 and version < 6000) or
                (self.group > 1 and version < 7000)):
            self.err = ValueError
        elif ((self.dilate > 1 and deterministic) or
                  (ndim > 2 and deterministic and 5000 <= version < 6000) or
                  (ndim > 2 and deterministic and self.dtype == numpy.float64)):
            self.err = libcudnn.CuDNNError

    def test_call(self):
        args = (self.W, self.gy, self.b, self.gx,
                self.pads, self.strides, self.dilations,
                self.group, self.max_workspace_size, self.deterministic,
                self.auto_tune, self.tensor_core)
        if self.err is None:
            cudnn.convolution_backward_data(*args)
            self.assertTrue((self.gx == 0).all())
        else:
            with self.assertRaises(self.err):
                cudnn.convolution_backward_data(*args)

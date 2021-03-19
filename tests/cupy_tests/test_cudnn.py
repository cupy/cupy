import sys
import unittest

import numpy

import cupy
import cupy.cuda.cudnn as libcudnn
from cupy import testing


cudnn_enabled = libcudnn.available


if cudnn_enabled:
    modes = [
        libcudnn.CUDNN_ACTIVATION_SIGMOID,
        libcudnn.CUDNN_ACTIVATION_RELU,
        libcudnn.CUDNN_ACTIVATION_TANH,
    ]
    coef_modes = [
        libcudnn.CUDNN_ACTIVATION_CLIPPED_RELU,
    ]
    layouts = [
        libcudnn.CUDNN_TENSOR_NCHW,
        libcudnn.CUDNN_TENSOR_NHWC,
    ]
    cudnn_version = libcudnn.getVersion()
    if cudnn_version >= 6000:
        coef_modes.append(libcudnn.CUDNN_ACTIVATION_ELU)

    from cupy import cudnn
else:
    cudnn_version = -1
    modes = []
    coef_modes = []
    layouts = []


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
@unittest.skipUnless(cudnn_enabled, 'cuDNN is not available')
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
@unittest.skipUnless(cudnn_enabled, 'cuDNN is not available')
class TestCudnnDropout(unittest.TestCase):

    def setUp(self):
        self.x = testing.shaped_arange((3, 4), cupy, self.dtype)
        self.gy = testing.shaped_arange((3, 4), cupy, self.dtype)
        self.states = cudnn.DropoutStates(None, self.seed)

    def test_dropout_forward(self):
        _, y = self.states.forward(None, self.x, self.ratio)
        if self.ratio == 0:
            assert cupy.all(self.x == y)
        else:
            assert cupy.all(self.x != y)

    def test_dropout_backward(self):
        rspace, y = self.states.forward(None, self.x, self.ratio)
        gx = self.states.backward(
            None, self.gy, self.ratio, rspace)

        forward_mask = y / self.x
        backward_mask = gx / self.gy

        # backward_mask must be the same as forward_mask
        assert cupy.all(forward_mask == backward_mask)

    def test_dropout_seed(self):
        # initialize Dropoutstates with the same seed
        states2 = cudnn.DropoutStates(None, self.seed)

        rspace, y = self.states.forward(None, self.x, self.ratio)
        rspace2, y2 = states2.forward(None, self.x, self.ratio)
        # forward results must be the same
        assert cupy.all(y == y2)

        gx = self.states.backward(None, self.gy, self.ratio, rspace)
        gx2 = states2.backward(None, self.gy, self.ratio, rspace2)
        # backward results must be the same
        assert cupy.all(gx == gx2)


@testing.parameterize(*(testing.product({
    'tensor_core': ['always', 'auto', 'never'],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'dilate': [1, 2],
    'groups': [1, 2],
    'ndim': [2],
    'max_workspace_size': [0, 2 ** 22],
    'auto_tune': [True, False],
    'bias': [True, False],
    'layout': layouts,
})))
@unittest.skipUnless(cudnn_enabled, 'cuDNN is not available')
class TestConvolutionForward(unittest.TestCase):

    def setUp(self):
        ndim = self.ndim
        dtype = self.dtype
        batches = 2
        if self.layout == libcudnn.CUDNN_TENSOR_NHWC:
            # channel size must be multiple of 4
            in_channels_a_group = 4
            out_channels_a_group = 4
        else:
            in_channels_a_group = 3
            out_channels_a_group = 2
        in_channels = in_channels_a_group * self.groups
        out_channels = out_channels_a_group * self.groups
        # TODO(anaruse): increase test cases.
        ksize = 3
        stride = 2
        pad = ksize // stride * self.dilate
        self.strides = (stride,) * ndim
        self.pads = (pad,) * ndim
        self.dilations = (self.dilate,) * ndim
        if self.layout == libcudnn.CUDNN_TENSOR_NHWC:
            self.x = cupy.zeros(
                (batches,) + (ksize,) * ndim + (in_channels,), dtype)
            self.W = cupy.zeros(
                (out_channels,) + (ksize,) * ndim + (in_channels_a_group,),
                dtype)
            self.y = cupy.ones(
                (batches,) + (2,) * ndim + (out_channels,), dtype)
        else:
            self.x = cupy.zeros(
                (batches, in_channels) + (ksize,) * ndim, dtype)
            self.W = cupy.zeros(
                (out_channels, in_channels_a_group) + (ksize,) * ndim, dtype)
            self.y = cupy.ones((batches, out_channels) + (2,) * ndim, dtype)
        self.b = None
        if self.bias:
            self.b = cupy.zeros((out_channels,), dtype)

        version = libcudnn.getVersion()
        self.err = None
        if ((self.dilate > 1 and version < 6000) or
                (self.groups > 1 and version < 7000)):
            self.err = ValueError
        elif ndim > 2 and self.dilate > 1:
            self.err = libcudnn.CuDNNError
        self._workspace_size = cudnn.get_max_workspace_size()
        cudnn.set_max_workspace_size(self.max_workspace_size)

    def tearDown(self):
        cudnn.set_max_workspace_size(self._workspace_size)

    def call(self):
        cudnn.convolution_forward(
            self.x, self.W, self.b, self.y,
            self.pads, self.strides, self.dilations, self.groups,
            auto_tune=self.auto_tune, tensor_core=self.tensor_core,
            d_layout=self.layout, w_layout=self.layout)

    def test_call(self):
        if self.layout == libcudnn.CUDNN_TENSOR_NHWC:
            version = libcudnn.getVersion()
            if self.groups > 1:
                return unittest.SkipTest()
            if self.dilate > 1 and version < 7300:
                return unittest.SkipTest()
            if self.dtype is numpy.float64 and version < 7100:
                return unittest.SkipTest()
        if self.err is None:
            self.call()
            assert (self.y == 0).all()
        else:
            with self.assertRaises(self.err):
                self.call()


@testing.parameterize(*(testing.product({
    'tensor_core': ['always', 'auto', 'never'],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'dilate': [1, 2],
    'groups': [1, 2],
    'ndim': [2, 3],
    'max_workspace_size': [0, 2 ** 22],
    'auto_tune': [True, False],
    'deterministic': [True, False],
})))
@unittest.skipUnless(cudnn_enabled, 'cuDNN is not available')
class TestConvolutionBackwardFilter(unittest.TestCase):

    def setUp(self):
        ndim = self.ndim
        dtype = self.dtype
        batches = 2
        in_channels_a_group = 3
        out_channels_a_group = 2
        in_channels = in_channels_a_group * self.groups
        out_channels = out_channels_a_group * self.groups
        # TODO(anaruse): increase test cases.
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
                (self.groups > 1 and version < 7000)):
            self.err = ValueError
        elif deterministic and (
                (self.dilate > 1 and version < 7000) or
                (ndim > 2 and version < 6000) or
                (ndim > 2 and self.dtype == numpy.float64 and version < 8100)):
            self.err = libcudnn.CuDNNError
        elif (8000 <= version < 8100 and
              self.max_workspace_size == 0 and
              int(cupy.cuda.device.get_compute_capability()) < 70 and
              self.groups > 1 and ndim > 2 and
              self.dtype == numpy.float16):
            self.err = RuntimeError
        self._workspace_size = cudnn.get_max_workspace_size()
        cudnn.set_max_workspace_size(self.max_workspace_size)

    def tearDown(self):
        cudnn.set_max_workspace_size(self._workspace_size)

    def call(self):
        cudnn.convolution_backward_filter(
            self.x, self.gy, self.gW,
            self.pads, self.strides, self.dilations, self.groups,
            deterministic=self.deterministic,
            auto_tune=self.auto_tune,
            tensor_core=self.tensor_core)

    def test_call(self):
        if self.deterministic and self.max_workspace_size == 0:
            # This test case is very unstable
            return
        if self.err is None:
            self.call()
            assert (self.gW == 0).all()
        else:
            with self.assertRaises(self.err):
                self.call()


@testing.parameterize(*(testing.product({
    'tensor_core': ['always', 'auto', 'never'],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'dilate': [1, 2],
    'groups': [1, 2],
    'ndim': [2, 3],
    'max_workspace_size': [0, 2 ** 22],
    'auto_tune': [True, False],
    'deterministic': [True, False],
    'bias': [True, False],
})))
@unittest.skipUnless(cudnn_enabled, 'cuDNN is not available')
class TestConvolutionBackwardData(unittest.TestCase):

    def setUp(self):
        ndim = self.ndim
        dtype = self.dtype
        batches = 2
        in_channels_a_group = 3
        out_channels_a_group = 2
        in_channels = in_channels_a_group * self.groups
        out_channels = out_channels_a_group * self.groups
        # TODO(anaruse): increase test cases.
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
                (self.groups > 1 and version < 7000)):
            self.err = ValueError
        elif (sys.platform.startswith('win32') and version == 7605
                and deterministic and self.dtype == numpy.float16
                and self.ndim == 3 and self.dilate == 2 and self.groups == 2):
            # see https://github.com/cupy/cupy/pull/4893
            self.err = RuntimeError
        elif deterministic and (
                (self.dilate > 1 and
                 (ndim != 2 and version < 8100 or version < 7300)) or
                (ndim > 2 and version < 6000) or
                (ndim > 2 and self.dtype == numpy.float64 and version < 8100)):
            self.err = libcudnn.CuDNNError
        elif (8000 <= version < 8100 and
              int(cupy.cuda.device.get_compute_capability()) < 70 and
              self.dilate > 1 and self.groups > 1 and ndim > 2 and
              self.dtype == numpy.float16):
            self.err = RuntimeError
        self._workspace_size = cudnn.get_max_workspace_size()
        cudnn.set_max_workspace_size(self.max_workspace_size)

    def tearDown(self):
        cudnn.set_max_workspace_size(self._workspace_size)

    def call(self):
        cudnn.convolution_backward_data(
            self.W, self.gy, self.b, self.gx,
            self.pads, self.strides, self.dilations, self.groups,
            deterministic=self.deterministic,
            auto_tune=self.auto_tune,
            tensor_core=self.tensor_core)

    def test_call(self):
        if self.deterministic and self.max_workspace_size == 0:
            # This test case is very unstable
            return
        if self.err is None:
            self.call()
            assert (self.gx == 0).all()
        else:
            with self.assertRaises(self.err):
                self.call()


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
    'ksize': [1, 3, 5],
    'stride': [2, 4],
    'auto_tune': [True, False],
}))
@unittest.skipIf(not cudnn_enabled or cudnn_version < 7500 or
                 cudnn_version >= 8000, 'cuDNN 7.5.0 or later is required')
class TestConvolutionNoAvailableAlgorithm(unittest.TestCase):
    '''Checks if an expected error is raised.

    This checks if an expected error is raised when no available algorithm
    is found by cuDNN for a configuration. This (no available algorithm found)
    can occur when convolution_backward_data or convolution_backward_filter is
    performed with NHWC layout.

    Please notice that conditions that cause the error may change depending on
    cuDNN version. The conditions below are set based on cuDNN 7.5.0 and 7.6.0.
    '''

    def setUp(self):
        self.layout = libcudnn.CUDNN_TENSOR_NHWC
        n = 16
        x_c, y_c = 64, 64
        x_h, x_w = 32, 32
        y_h, y_w = x_h // self.stride, x_w // self.stride
        self.pad = (self.ksize - 1) // 2
        if self.layout == libcudnn.CUDNN_TENSOR_NHWC:
            x_shape = (n, x_h, x_w, x_c)
            y_shape = (n, y_h, y_w, y_c)
            W_shape = (y_c, self.ksize, self.ksize, x_c)
        else:
            x_shape = (n, x_c, x_h, x_w)
            y_shape = (n, y_c, y_h, y_w)
            W_shape = (y_c, x_c, self.ksize, self.ksize)
        self.x = cupy.ones(x_shape, dtype=self.dtype)
        self.W = cupy.ones(W_shape, dtype=self.dtype)
        self.y = cupy.empty(y_shape, dtype=self.dtype)
        self.gx = cupy.empty(x_shape, dtype=self.dtype)
        self.gW = cupy.empty(W_shape, dtype=self.dtype)
        self.gy = cupy.ones(y_shape, dtype=self.dtype)
        self._workspace_size = cudnn.get_max_workspace_size()
        cudnn.set_max_workspace_size(0)

    def tearDown(self):
        cudnn.set_max_workspace_size(self._workspace_size)

    def test_backward_filter(self):
        if not (self.layout == libcudnn.CUDNN_TENSOR_NHWC and
                self.dtype == numpy.float64):
            return unittest.SkipTest()
        with self.assertRaises(RuntimeError):
            cudnn.convolution_backward_filter(
                self.x, self.gy, self.gW,
                pad=(self.pad, self.pad), stride=(self.stride, self.stride),
                dilation=(1, 1), groups=1, deterministic=False,
                auto_tune=self.auto_tune, tensor_core='always',
                d_layout=self.layout, w_layout=self.layout)

    def test_backward_data(self):
        if self.layout != libcudnn.CUDNN_TENSOR_NHWC:
            return unittest.SkipTest()
        with self.assertRaises(RuntimeError):
            cudnn.convolution_backward_data(
                self.W, self.gy, None, self.gx,
                pad=(self.pad, self.pad), stride=(self.stride, self.stride),
                dilation=(1, 1), groups=1, deterministic=0,
                auto_tune=self.auto_tune, tensor_core='always',
                d_layout=self.layout, w_layout=self.layout)

    def _get_error_type(self):
        if self.auto_tune:
            return RuntimeError
        else:
            return libcudnn.CuDNNError

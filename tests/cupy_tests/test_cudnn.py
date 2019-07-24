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
    layouts = [
        libcudnn.CUDNN_TENSOR_NCHW,
        libcudnn.CUDNN_TENSOR_NHWC,
    ]
    cudnn_version = libcudnn.getVersion()
    if cudnn_version >= 6000:
        coef_modes.append(libcudnn.CUDNN_ACTIVATION_ELU)

    from cupy import cudnn
except ImportError:
    cudnn_enabled = False
    cudnn_version = -1
    modes = []
    coef_modes = []
    layouts = []

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
            self.assertTrue(cupy.all(self.x == y))
        else:
            self.assertTrue(cupy.all(self.x != y))

    def test_dropout_backward(self):
        rspace, y = self.states.forward(None, self.x, self.ratio)
        gx = self.states.backward(
            None, self.gy, self.ratio, rspace)

        forward_mask = y / self.x
        backward_mask = gx / self.gy

        # backward_mask must be the same as forward_mask
        self.assertTrue(cupy.all(forward_mask == backward_mask))

    def test_dropout_seed(self):
        # initialize Dropoutstates with the same seed
        states2 = cudnn.DropoutStates(None, self.seed)

        rspace, y = self.states.forward(None, self.x, self.ratio)
        rspace2, y2 = states2.forward(None, self.x, self.ratio)
        # forward results must be the same
        self.assertTrue(cupy.all(y == y2))

        gx = self.states.backward(None, self.gy, self.ratio, rspace)
        gx2 = states2.backward(None, self.gy, self.ratio, rspace2)
        # backward results must be the same
        self.assertTrue(cupy.all(gx == gx2))


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
            self.assertTrue((self.y == 0).all())
        else:
            with self.assertRaises(self.err):
                self.call()


@testing.parameterize(*testing.product({
    'tensor_core': ['always', 'auto', 'never'],
    'max_workspace_size': [0, 2 ** 22],
    'auto_tune': [True, False],
}))
@unittest.skipUnless(cudnn_enabled, 'cuDNN is not available')
class TestConvolutionForwardSkipTensorCore(unittest.TestCase):

    def setUp(self):
        ndim = 2
        self.batches = 2
        self.in_channels_a_group = 3
        self.out_channels_a_group = 2
        self.groups = 2
        dilate = 1
        self.in_channels = self.in_channels_a_group * self.groups
        self.out_channels = self.out_channels_a_group * self.groups
        kh, kw = (3, 3)
        stride = 2
        self.pads = int(kh / 2) * dilate, int(kw / 2) * dilate
        self.strides = (stride,) * ndim
        self.dilations = (dilate,) * ndim
        self.layout = libcudnn.CUDNN_TENSOR_NCHW

        x = [[[[0.776,    -0.4504,   -0.2369, ],
               [0.689,     0.8115,    0.049, ],
               [-0.5625,    0.006454,  0.9307, ],
               [0.00689,  -0.2722,    0.6987, ]],

              [[0.4014,    0.5864,   -0.1614, ],
               [-0.563,    -0.3965,    0.92, ],
               [-0.8145,    0.963,     0.4856, ],
               [-0.06204,  -0.695,    -0.1067, ]],

              [[0.991,    -0.6064,   -0.818, ],
               [-0.987,    -0.5425,   -0.9062, ],
               [-0.7124,    0.9126,   -0.0744, ],
               [-0.1617,    0.693,    -0.561, ]],

              [[0.007305, -0.8125,    0.4832, ],
               [0.7744,    0.3582,   -0.215, ],
               [-0.525,    -0.5874,    0.1411, ],
               [0.8213,   -0.1137,   -0.747, ]],

              [[-0.3376,    0.3293,   -0.2698, ],
               [0.939,    -0.954,    -0.569, ],
               [0.1354,   -0.11444,  -0.4707, ],
               [0.7593,   -0.7104,    0.803, ]],

              [[-0.663,    -0.4812,   -0.4238, ],
               [-0.7163,    0.5034,   -0.1537, ],
               [-0.3987,    0.778,     0.5083, ],
               [-0.709,     0.2113,    0.1915, ]]],


             [[[-0.795,    -0.2402,    0.91, ],
               [-0.1226,    0.1809,    0.3376, ],
               [0.3254,   -0.8447,   -0.8022, ],
               [-0.2147,    0.3584,    0.463, ]],

              [[0.23,      0.169,     0.1134, ],
               [-0.975,    -0.244,     0.573, ],
               [0.5503,   -0.863,    -0.0419, ],
               [0.8984,    0.941,    -0.1965, ]],

              [[0.9263,    0.02089,  -0.5874, ],
               [0.11957,  -0.9814,   -0.01357, ],
               [-0.9487,   -0.2556,    0.65, ],
               [0.6113,    0.8735,   -0.11664, ]],

              [[0.9175,    0.652,     0.975, ],
               [0.2301,   -0.07086,  -0.01735, ],
               [-0.884,    -0.583,     0.607, ],
               [0.515,     0.05148,   0.2993, ]],

              [[0.4553,    0.4463,   -0.377, ],
               [0.4026,    0.1733,   -0.7812, ],
               [0.544,    -0.5244,   -0.4343, ],
               [-0.4248,    0.178,    -0.5576, ]],

              [[-0.4858,   -0.9824,   -0.6255, ],
               [0.4001,   -0.5107,    0.977, ],
               [-0.2837,   -0.5957,   -0.781, ],
               [-0.674,    -0.4985,    0.941, ]]]
             ]

        W = [[[[0.02321,  -0.06177,  -0.2913, ],
               [-0.11633,  -0.4014,   -0.2534, ],
               [0.00454,   0.07556,  -0.0777, ]],

              [[0.1081,   -0.2637,   -0.2081, ],
               [0.263,     0.3916,   -0.1979, ],
               [-0.129,    -0.325,    -0.03036, ]],

              [[0.0693,   -0.163,     0.06915, ],
               [0.0689,    0.3005,    0.07153, ],
               [0.05164,  -0.2052,    0.1224, ]]],


             [[[-0.06976,   0.0836,   -0.12463, ],
               [0.2145,   -0.2952,   -0.1434, ],
               [-0.06226,   0.1884,    0.0932, ]],

              [[0.213,     0.07306,  -0.005753, ],
               [0.2742,    0.173,     0.034, ],
               [-0.0927,   -0.2458,   -0.1143, ]],

              [[-0.0489,    0.181,    -0.11945, ],
               [-0.09436,  -0.1709,   -0.2335, ],
               [0.08453,  -0.336,    -0.311, ]]],


             [[[-0.372,    -0.0971,    0.2194, ],
               [0.241,    -0.1783,   -0.03806, ],
               [-0.327,    -0.11584,   0.4104, ]],

              [[0.09705,   0.02713,  -0.1573, ],
               [0.1149,   -0.01614,  -0.2957, ],
               [0.2744,   -0.1859,    0.08325, ]],

              [[-0.07227,  -0.1393,   -0.07214, ],
               [0.03867,  -0.05402,   0.2009, ],
               [0.04028,  -0.2137,    0.2622, ]]],


             [[[0.3481,    0.11273,   0.1486, ],
               [-0.141,     0.04504,  -0.1824, ],
               [0.2659,    0.03023,  -0.1613, ]],

              [[0.292,    -0.3323,    0.1097, ],
               [-0.2434,    0.11884,   0.1748, ],
               [-0.0466,    0.007015, -0.2869, ]],

              [[0.1998,   -0.09784,  -0.08527, ],
               [-0.227,    -0.3413,    0.001161, ],
               [-0.1744,   -0.0505,   -0.2593, ]]]]

        y = [[[[0.4185,  -0.1315, ],
               [-0.1779,   0.2769, ]],

              [[0.7866,   0.331, ],
               [-0.3213,   0.2803, ]],

              [[-0.0348,  -0.4307, ],
               [0.4326,  -0.658, ]],

              [[0.5435,   0.3271, ],
               [0.2009,  -0.2231, ]]],


             [[[0.8726,  -0.6, ],
               [-0.04147,  0.1428, ]],

              [[0.6777,  -0.3003, ],
               [-0.4636,  -0.0851, ]],

              [[-0.8345,   0.02492, ],
               [0.2172,  -0.4785, ]],

              [[0.303,    0.242, ],
               [0.1572,   0.723, ]]]]

        self.x = cupy.array(x, dtype='float16')
        self.W = cupy.array(W, dtype='float16')
        self.b = None
        self.expected = cupy.array(y, dtype='float16')

    def test_forward(self):
        y = cupy.zeros_like(self.expected)
        cudnn.convolution_forward(
            self.x, self.W, self.b, y,
            self.pads, self.strides, self.dilations, self.groups,
            auto_tune=self.auto_tune, tensor_core=self.tensor_core,
            d_layout=self.layout, w_layout=self.layout)
        testing.assert_allclose(y, self.expected)


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
        elif ((self.dilate > 1 and deterministic and version < 7000) or
                (ndim > 2 and deterministic and version < 6000) or
                (ndim > 2 and deterministic and self.dtype == numpy.float64)):
            self.err = libcudnn.CuDNNError
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
        if self.err is None:
            self.call()
            self.assertTrue((self.gW == 0).all())
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
        elif deterministic and (
                (self.dilate > 1 and (ndim != 2 or version < 7300)) or
                (ndim > 2 and version < 6000) or
                (ndim > 2 and self.dtype == numpy.float64)):
            self.err = libcudnn.CuDNNError
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
        if self.err is None:
            self.call()
            self.assertTrue((self.gx == 0).all())
        else:
            with self.assertRaises(self.err):
                self.call()


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
    'ksize': [1, 3, 5],
    'stride': [2, 4],
    'auto_tune': [True, False],
}))
@unittest.skipIf(not cudnn_enabled or cudnn_version < 7500,
                 'cuDNN 7.5.0 or later is required')
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
        err = None
        if (self.layout == libcudnn.CUDNN_TENSOR_NHWC and
                self.dtype == numpy.float64):
            err = self._get_error_type()
        if err is None:
            return unittest.SkipTest()
        with self.assertRaises(err):
            cudnn.convolution_backward_filter(
                self.x, self.gy, self.gW,
                pad=(self.pad, self.pad), stride=(self.stride, self.stride),
                dilation=(1, 1), groups=1, deterministic=0,
                auto_tune=self.auto_tune, tensor_core='always',
                d_layout=self.layout, w_layout=self.layout)

    def test_backward_data(self):
        err = None
        if self.layout == libcudnn.CUDNN_TENSOR_NHWC:
            err = self._get_error_type()
        if err is None:
            return unittest.SkipTest()
        with self.assertRaises(err):
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

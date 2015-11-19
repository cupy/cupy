import math

import numpy
from six import moves

from chainer import cuda
from chainer import function
from chainer.utils import conv
from chainer.utils import type_check
import ctypes

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cuda.cudnn.cudnn
    _fwd_pref = libcudnn.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return (x, x)


def get_deconv_outsize(h, kh, sy, ph):
    return sy * (h - 1) + kh - 2 * ph


class Deconvolution2D(function.Function):

    """Two dimensional deconvolution function.

    The details of this function are described below the arguments description.

    Args:
        in_channels (int): Number of channels of input arrays.
        out_channels (int): Number of channels of output arrays.
        ksize (int or (int, int)): Size of filters (a.k.a. kernels).
            ``ksize=k`` and ``ksize=(k, k)`` are equivalent.
        stride (int or (int, int)): Stride of filter applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent.
        pad (int or (int, int)): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        wscale (float): Scaling factor of the initial weight.
        bias (float): Initial bias value.
        nobias (bool): If True, then this function does not use the bias term.
        use_cudnn (bool): If True, then this function uses CuDNN if available.
        initialW (4-D array): Initial weight value. If ``None``, then this
            function uses to initialize ``wscale``.
        initial_bias (1-D array): Initial bias value. If ``None``, then this
            function uses to initialize ``bias``.

    This function holds at most two parameter arrays: ``W`` and ``b``, which
    indicate the filter weight and the bias vector, respectively.

    The filter weight has four dimensions :math:`(c_I, c_O, k_H, k_W)`
    which indicate the number of the number of input channels, output channels,
    height and width of the kernels, respectively.
    The filter weight is initialized with i.i.d. Gaussian random samples, each
    of which has zero mean and deviation :math:`\sqrt{1/(c_I k_H k_W)}` by
    default. The deviation is scaled by ``wscale`` if specified.

    The bias vector is of size :math:`c_O`.
    Each element of it is initialized by ``bias`` argument.
    If ``nobias`` argument is set to True, then this function does not hold
    the bias parameter.

    Let :math:`X` be the input tensor of dimensions :math:`(n, c_I, h, w)`,
    :math:`(s_Y, s_X)` be the stride of filter application, and
    :math:`(p_H, p_W)` the spatial padding size. Then, the output size
    :math:`(h_O, w_O)` is determined by the following equations:

    .. math::

       h_O &= s_Y (h - 1) + k_H - 2p_H,\\\\
       w_O &= s_X (w - 1) + k_W - 2p_W.

    """

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 wscale=1, bias=0, nobias=False, use_cudnn=True,
                 initialW=None, initial_bias=None):
        ksize = _pair(ksize)
        stride = _pair(stride)
        pad = _pair(pad)

        self.kh, self.kw = ksize
        self.sy, self.sx = stride
        self.ph, self.pw = pad

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.W = None
        self.gW = None
        self.b = None
        self.gb = None

        if initialW is not None:
            assert initialW.shape == \
                (in_channels, out_channels, self.kh, self.kw)
            self.W = initialW
        else:
            self.W = numpy.random.normal(
                0, wscale * math.sqrt(1. / (self.kh * self.kw * in_channels)),
                (in_channels, out_channels, self.kh, self.kw)
            ).astype(numpy.float32)
        xp = cuda.get_array_module(self.W)
        self.gW = xp.full_like(self.W, numpy.nan)

        if initial_bias is not None:
            assert initial_bias.shape == (out_channels,)
            self.b = initial_bias
        elif not nobias:
            self.b = numpy.repeat(numpy.float32(bias), out_channels)

        if self.b is not None:
            self.gb = xp.full_like(self.b, numpy.nan)

        self.use_cudnn = use_cudnn
        # chance to choose implicit-precomp-gemm algorithm
        self.max_workspace_size = out_channels * self.kh * self.kw * 4

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim == 4,
            x_type.shape[1] == self.in_channels
        )

    @property
    def parameter_names(self):
        if self.b is None:
            return 'W',
        return 'W', 'b'

    @property
    def gradient_names(self):
        if self.gb is None:
            return 'gW',
        return 'gW', 'gb'

    def zero_grads(self):
        self.gW.fill(0)
        if self.gb is not None:
            self.gb.fill(0)

    def forward_cpu(self, x):
        n, c, h, w = x[0].shape
        gcol = numpy.tensordot(self.W, x[0], (0, 1))
        # k, m, n, b, h, w
        gcol = numpy.rollaxis(gcol, 3)
        # b, k, m, n, h, w
        h_ = get_deconv_outsize(h, self.kh, self.sy, self.ph)
        w_ = get_deconv_outsize(w, self.kw, self.sx, self.pw)
        y = conv.col2im_cpu(
            gcol, self.sy, self.sx, self.ph, self.pw, h_, w_)
        # b, k, h, w
        if self.b is not None:
            y += self.b.reshape(1, self.b.size, 1, 1)
        return y,

    def forward_gpu(self, x):
        n, in_c, in_h, in_w = x[0].shape
        c = self.W.shape[1]  # out_c
        h = get_deconv_outsize(in_h, self.kh, self.sy, self.ph)
        w = get_deconv_outsize(in_w, self.kw, self.sx, self.pw)
        if cuda.cudnn_enabled and self.use_cudnn:
            handle = cudnn.get_handle()
            x_desc = cudnn.create_tensor_descriptor(x[0])
            y = cuda.empty((n, c, h, w), dtype=numpy.float32)
            y_desc = cudnn.create_tensor_descriptor(y)

            self.filter_desc = cudnn.create_filter_descriptor(self.W)
            self.conv_desc = cudnn.create_convolution_descriptor(
                (self.ph, self.pw), (self.sy, self.sx))
            if self.b is not None:
                self.bias_desc = cudnn.create_tensor_descriptor(
                    self.b[None, :, None, None])

            one = ctypes.c_float(1)
            zero = ctypes.c_float(0)

            libcudnn.convolutionBackwardData(
                handle, one, self.filter_desc.value, self.W.data.ptr,
                x_desc.value, x[0].data.ptr, self.conv_desc.value,
                zero, y_desc.value, y.data.ptr)
            if self.b is not None:
                libcudnn.addTensor(
                    handle, libcudnn.CUDNN_ADD_SAME_C,
                    one, self.bias_desc.value, self.b.data.ptr,
                    one, y_desc.value, y.data.ptr)
        else:
            W_mat = self.W.reshape(in_c, c * self.kh * self.kw)
            x_mats = x[0].reshape(n, in_c, in_h * in_w)
            gcol = cuda.empty((n, c, self.kh, self.kw, in_h,
                               in_w), dtype=numpy.float32)
            gcol_mats = gcol.reshape(n, c * self.kh * self.kw, in_h * in_w)
            for i in moves.range(n):
                cuda.cupy.dot(W_mat.T, x_mats[i], gcol_mats[i])
            y = conv.col2im_gpu(
                gcol, self.sy, self.sx, self.ph, self.pw, h, w)
            if self.b is not None:
                y += self.b.reshape(1, self.b.size, 1, 1)
        return y,

    def backward_cpu(self, x, gy):
        if self.gb is not None:
            self.gb += gy[0].sum(axis=(0, 2, 3))
        col = conv.im2col_cpu(
            gy[0], self.kh, self.kw, self.sy, self.sx, self.ph, self.pw)
        self.gW += numpy.tensordot(x[0], col, ([0, 2, 3], [0, 4, 5]))
        gx = numpy.tensordot(col, self.W, ([1, 2, 3], [1, 2, 3]))
        gx = numpy.rollaxis(gx, 3, 1)
        return gx,

    def backward_gpu(self, x, gy):
        n, in_c, in_h, in_w = x[0].shape
        c, h, w = gy[0].shape[1:]
        gx = cuda.empty((n, in_c, in_h, in_w), dtype=numpy.float32)
        if cuda.cudnn_enabled and self.use_cudnn:
            handle = cudnn.get_handle()
            gy_desc = cudnn.create_tensor_descriptor(gy[0])
            gx_desc = cudnn.create_tensor_descriptor(gx)

            algo = libcudnn.getConvolutionForwardAlgorithm(
                handle, gy_desc.value, self.filter_desc.value,
                self.conv_desc.value, gx_desc.value, _fwd_pref,
                self.max_workspace_size)
            workspace_size = libcudnn.getConvolutionForwardWorkspaceSize(
                handle, gy_desc.value, self.filter_desc.value,
                self.conv_desc.value, gx_desc.value, algo)
            workspace = cuda.empty(
                (max(workspace_size // 4, 1),), dtype=numpy.float32)

            one = ctypes.c_float(1)
            zero = ctypes.c_float(0)

            libcudnn.convolutionForward(
                handle, one, gy_desc.value, gy[0].data.ptr,
                self.filter_desc.value, self.W.data.ptr,
                self.conv_desc.value, algo, workspace.data.ptr, workspace_size,
                zero, gx_desc.value, gx.data.ptr)
            # bias backward
            if self.b is not None:
                libcudnn.convolutionBackwardBias(
                    handle, one, gy_desc.value, gy[0].data.ptr,
                    zero, self.bias_desc.value, self.gb.data.ptr)
            # filter backward
            libcudnn.convolutionBackwardFilter(
                handle, one, gy_desc.value, gy[0].data.ptr,
                gx_desc.value, x[0].data.ptr, self.conv_desc.value,
                one, self.filter_desc.value, self.gW.data.ptr)
        else:
            # Implementation using im2col
            col = conv.im2col_gpu(
                gy[0], self.kh, self.kw, self.sy, self.sx, self.ph, self.pw)

            W_mat = self.W.reshape(in_c, c * self.kh * self.kw)
            col_mats = col.reshape(
                n, c * self.kh * self.kw, in_h * in_w)
            gx_mats = gx.reshape(n, in_c, in_h * in_w)
            for i in moves.range(n):
                gx_mats[i] = W_mat.dot(col_mats[i])

            # bias backward
            if self.gb is not None:
                self.gb += gy[0].sum(axis=(0, 2, 3))

            # filter backward
            gW_mat = self.gW.reshape(in_c, c * self.kh * self.kw)
            x_mats = x[0].reshape(n, in_c, in_h * in_w)
            for i in moves.range(n):
                gW_mat += x_mats[i].dot(col_mats[i].T)
        return gx,

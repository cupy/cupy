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


class Deconvolution2DFunction(function.Function):

    def __init__(self, stride=1, pad=0, use_cudnn=True):
        self.sy, self.sx = _pair(stride)
        self.ph, self.pw = _pair(pad)
        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)
        x_type, w_type = in_types[:2]

        type_check.expect(
            x_type.dtype == numpy.float32,
            w_type.dtype == numpy.float32,
            x_type.ndim == 4,
            w_type.ndim == 4,
            x_type.shape[1] == w_type.shape[0]
        )

        if n_in.eval() == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == numpy.float32,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[1]
            )

    def forward_cpu(self, inputs):
        x, W = inputs[:2]
        kh, kw = W.shape[2:]
        _, _, h, w = x.shape
        gcol = numpy.tensordot(W, x, (0, 1))
        # - k, m, n: shape of out_channel
        # - b: number of inputs
        # - h, w: height and width of kernels
        # k, m, n, b, h, w -> b, k, m, n, h, w
        gcol = numpy.rollaxis(gcol, 3)
        h_ = conv.get_deconv_outsize(h, kh, self.sy, self.ph)
        w_ = conv.get_deconv_outsize(w, kw, self.sx, self.pw)
        y = conv.col2im_cpu(
            gcol, self.sy, self.sx, self.ph, self.pw, h_, w_)
        # b, k, h, w
        if len(inputs) == 3:
            b = inputs[2]
            y += b.reshape(1, b.size, 1, 1)
        return y,

    def forward_gpu(self, inputs):
        x, W = inputs[:2]
        kh, kw = W.shape[2:]
        n, in_c, in_h, in_w = x.shape
        c = W.shape[1]  # out_c
        h = conv.get_deconv_outsize(in_h, kh, self.sy, self.ph)
        w = conv.get_deconv_outsize(in_w, kw, self.sx, self.pw)
        if len(inputs) == 3:
            b = inputs[2]
        if cuda.cudnn_enabled and self.use_cudnn:
            handle = cudnn.get_handle()
            x_desc = cudnn.create_tensor_descriptor(x)
            y = cuda.empty((n, c, h, w), dtype=numpy.float32)
            y_desc = cudnn.create_tensor_descriptor(y)

            self.filter_desc = cudnn.create_filter_descriptor(W)
            self.conv_desc = cudnn.create_convolution_descriptor(
                (self.ph, self.pw), (self.sy, self.sx))
            if len(inputs) == 3:
                self.bias_desc = cudnn.create_tensor_descriptor(
                    b[None, :, None, None])

            one = ctypes.c_float(1)
            zero = ctypes.c_float(0)

            libcudnn.convolutionBackwardData(
                handle, one, self.filter_desc.value, W.data.ptr,
                x_desc.value, x.data.ptr, self.conv_desc.value,
                zero, y_desc.value, y.data.ptr)
            if len(inputs) == 3:
                libcudnn.addTensor(
                    handle, libcudnn.CUDNN_ADD_SAME_C,
                    one, self.bias_desc.value, b.data.ptr,
                    one, y_desc.value, y.data.ptr)
        else:
            W_mat = W.reshape(in_c, c * kh * kw)
            x_mats = x.reshape(n, in_c, in_h * in_w)
            gcol = cuda.empty((n, c, kh, kw, in_h,
                               in_w), dtype=numpy.float32)
            gcol_mats = gcol.reshape(n, c * kh * kw, in_h * in_w)
            for i in moves.range(n):
                cuda.cupy.dot(W_mat.T, x_mats[i], gcol_mats[i])
            y = conv.col2im_gpu(
                gcol, self.sy, self.sx, self.ph, self.pw, h, w)
            if len(inputs) == 3:
                y += b.reshape(1, b.size, 1, 1)
        return y,

    def backward_cpu(self, inputs, grad_outputs):
        x, W = inputs[:2]
        gy = grad_outputs[0]
        kh, kw = W.shape[2:]
        col = conv.im2col_cpu(
            gy, kh, kw, self.sy, self.sx, self.ph, self.pw)
        gW = numpy.tensordot(x, col, ([0, 2, 3], [0, 4, 5]))
        gx = numpy.tensordot(col, W, ([1, 2, 3], [1, 2, 3]))
        gx = numpy.rollaxis(gx, 3, 1)

        if len(inputs) == 3:
            gb = gy.sum(axis=(0, 2, 3))
            return gx, gW, gb
        else:
            return gx, gW

    def backward_gpu(self, inputs, grad_outputs):
        x, W = inputs[:2]
        gy = grad_outputs[0]
        n, in_c, in_h, in_w = x.shape
        _, out_channels, kh, kw = W.shape
        c, h, w = gy.shape[1:]
        gx = cuda.empty((n, in_c, in_h, in_w), dtype=numpy.float32)
        gW = cuda.cupy.empty_like(W)
        if cuda.cudnn_enabled and self.use_cudnn:
            handle = cudnn.get_handle()
            gy_desc = cudnn.create_tensor_descriptor(gy)
            gx_desc = cudnn.create_tensor_descriptor(gx)

            # chance to choose implicit-precomp-gemm algorithm
            self.max_workspace_size = out_channels * kh * kw * 4
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
                handle, one, gy_desc.value, gy.data.ptr,
                self.filter_desc.value, W.data.ptr,
                self.conv_desc.value, algo, workspace.data.ptr, workspace_size,
                zero, gx_desc.value, gx.data.ptr)
            # bias backward
            if len(inputs) == 3:
                b = inputs[2]
                gb = cuda.cupy.empty_like(b)
                libcudnn.convolutionBackwardBias(
                    handle, one, gy_desc.value, gy.data.ptr,
                    zero, self.bias_desc.value, gb.data.ptr)
            # filter backward
            libcudnn.convolutionBackwardFilter(
                handle, one, gy_desc.value, gy.data.ptr,
                gx_desc.value, x.data.ptr, self.conv_desc.value,
                one, self.filter_desc.value, gW.data.ptr)
        else:
            # Implementation using im2col
            col = conv.im2col_gpu(
                gy, kh, kw, self.sy, self.sx, self.ph, self.pw)

            W_mat = W.reshape(in_c, c * kh * kw)
            col_mats = col.reshape(
                n, c * kh * kw, in_h * in_w)
            gx_mats = gx.reshape(n, in_c, in_h * in_w)
            for i in moves.range(n):
                gx_mats[i] = W_mat.dot(col_mats[i])

            # bias backward
            if len(inputs) == 3:
                gb = gy.sum(axis=(0, 2, 3))

            # filter backward
            gW_mat = gW.reshape(in_c, c * kh * kw)
            x_mats = x.reshape(n, in_c, in_h * in_w)
            for i in moves.range(n):
                gW_mat += x_mats[i].dot(col_mats[i].T)
        if len(inputs) == 3:
            return gx, gW, gb
        else:
            return gx, gW


def deconvolution_2d(x, W, b=None, stride=1, pad=0, use_cudnn=True):
    """Two dimensional deconvolution function.

    This is an implementation of two-dimensional deconvolution.
    It takes three variables: input image ``x``,
    the filter weight ``W``, and the bias vector ``b``.

    Args:
        x (~chainer.Variable): Input variable of shape :math:`(n, c_I, h, w)`
        W (~chainer.Variable): Weight variable of shape
        :math:`(c_I, c_O, k_H, k_W)`.
        b (~chainer.Variable): Bias variable of length :math:`c_O` (optional).
        stride (int or (int, int)): Stride of filter applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent.
        pad (int or (int, int)): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        use_cudnn (bool): If True, then this function uses CuDNN if available.


    The filter weight has four dimensions :math:`(c_I, c_O, k_H, k_W)`
    which indicate the number of the number of input channels, output channels,
    height and width of the kernels, respectively.

    The bias vector is of size :math:`c_O`.

    Let :math:`X` be the input tensor of dimensions :math:`(n, c_I, h, w)`,
    :math:`(s_Y, s_X)` the stride of filter application, and
    :math:`(p_H, p_W)` the spatial padding size. Then, the output size
    :math:`(h_O, w_O)` is determined by the following equations:

    .. math::

       h_O &= s_Y (h - 1) + k_H - 2p_H,\\\\
       w_O &= s_X (w - 1) + k_W - 2p_W.

    """
    func = Deconvolution2DFunction(stride, pad, use_cudnn)
    if b is None:
        return func(x, W)
    else:
        return func(x, W, b)

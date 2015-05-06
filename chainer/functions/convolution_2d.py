import math, os
import numpy
from scikits.cuda import cublas
import scikits.cuda.linalg as culinalg
import scikits.cuda.misc as cumisc
from chainer import cuda, cudnn, Function
from chainer.utils import conv

if cudnn.available:
    from chainer.cudnn import libcudnn
    _fwd_pref = libcudnn.cudnnConvolutionFwdPreference[
        'CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT']

def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return (x, x)

class Convolution2D(Function):
    """Two-dimensional convolution function."""

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 wscale=1, bias=0, nobias=False, use_cudnn=True):
        ksize  = _pair(ksize)
        stride = _pair(stride)
        pad    = _pair(pad)

        self.kh, self.kw = ksize
        self.sy, self.sx = stride
        self.ph, self.pw = pad

        self.W = numpy.random.normal(
            0, wscale * math.sqrt(1. / (self.kh * self.kw * in_channels)),
            (out_channels, in_channels, self.kh, self.kw)).astype(numpy.float32)
        self.gW = numpy.empty_like(self.W)

        if nobias:
            self.b  = None
            self.gb = None
        else:
            self.b  = numpy.repeat(numpy.float32(bias), out_channels)
            self.gb = numpy.empty_like(self.b)

        self.use_cudnn = use_cudnn
        if cudnn.enabled and use_cudnn:
            self.filter_desc = cudnn.get_filter4d_desc(self.W)
            self.conv_desc = cudnn.get_conv2d_desc(pad, stride)
            if self.b is not None:
                self.bias_desc = cudnn.get_conv_bias_desc(self.b)

            # chance to choose implicit-precomp-gemm algorithm
            self.max_workspace_size = in_channels * self.kh * self.kw * 4

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

    def forward_cpu(self, x):
        self.col = conv.im2col_cpu(
            x[0], self.kh, self.kw, self.sy, self.sx, self.ph, self.pw)
        y = numpy.tensordot(self.col, self.W, ([1, 2, 3], [1, 2, 3]))
        if self.b is not None:
            y += self.b
        return numpy.rollaxis(y, 3, 1),

    def forward_gpu(self, x):
        n, c, h, w = x[0].shape
        out_h = conv.get_conv_outsize(h, self.kh, self.sy, self.ph)
        out_w = conv.get_conv_outsize(w, self.kw, self.sx, self.pw)
        out_c = self.W.shape[0]
        y = cuda.empty((n, out_c, out_h, out_w), dtype=numpy.float32)

        if cudnn.enabled and self.use_cudnn:
            handle = cudnn.get_default_handle()
            x_desc = cudnn.get_tensor_desc(x[0], h, w)
            y_desc = cudnn.get_tensor_desc(y, out_h, out_w)

            algo = libcudnn.cudnnGetConvolutionForwardAlgorithm(
                handle, x_desc.value, self.filter_desc.value, self.conv_desc.value,
                y_desc.value, _fwd_pref, self.max_workspace_size)
            workspace_size = libcudnn.cudnnGetConvolutionForwardWorkspaceSize(
                handle, x_desc.value, self.filter_desc.value, self.conv_desc.value,
                y_desc.value, algo).value
            workspace = cuda.empty(
                (max(workspace_size / 4, 1),), dtype=numpy.float32)

            libcudnn.cudnnConvolutionForward(
                handle, 1, x_desc.value, cudnn.get_ptr(x[0]),
                self.filter_desc.value, cudnn.get_ptr(self.W),
                self.conv_desc.value, algo, cudnn.get_ptr(workspace), workspace_size,
                0, y_desc.value, cudnn.get_ptr(y))

            # TODO(beam2d): Support unshared bias
            if self.b is not None:
                libcudnn.cudnnAddTensor(
                    handle, libcudnn.cudnnAddMode['CUDNN_ADD_SAME_C'],
                    1, self.bias_desc.value, cudnn.get_ptr(self.b),
                    1, y_desc.value, cudnn.get_ptr(y))
        else:
            # Implementation using im2col
            self.col = conv.im2col_gpu(
                x[0], self.kh, self.kw, self.sy, self.sx, self.ph, self.pw)

            # TODO(beam2d): Use streams
            handle   = cuda.get_cublas_handle()
            W_mat    = self.W.reshape(out_c, c * self.kh * self.kw)
            col_mats = self.col.reshape(n, c * self.kh * self.kw, out_h * out_w)
            y_mats   = y.reshape(n, out_c, out_h * out_w)
            for i in xrange(n):
                culinalg.dot(W_mat, col_mats[i], handle=handle, out=y_mats[i])

            # TODO(beam2d): Support unshared bias
            if self.b is not None:
                cuda.elementwise(
                    'float* y, const float* b, int c, int hw',
                    'y[i] += b[i / hw % c]',
                    'conv_bias_fwd')(y, self.b, out_c, out_h * out_w)

        return y,

    def backward_cpu(self, x, gy):
        if self.gb is not None:
            self.gb += gy[0].sum(axis=(0, 2, 3))
        self.gW += numpy.tensordot(gy[0], self.col, ([0, 2, 3], [0, 4, 5]))
        gcol = numpy.tensordot(self.W, gy[0], (0, 1))
        gcol = numpy.rollaxis(gcol, 3)

        h, w = x[0].shape[2:]
        return conv.col2im_cpu(gcol, self.sy, self.sx, self.ph, self.pw, h, w),

    def backward_gpu(self, x, gy):
        out_c, out_h, out_w = gy[0].shape[1:]
        n, c, h, w = x[0].shape

        if cudnn.enabled and self.use_cudnn:
            handle = cudnn.get_default_handle()
            x_desc  = cudnn.get_tensor_desc(x[0], h, w)
            gy_desc = cudnn.get_tensor_desc(gy[0], out_h, out_w)
            if self.b is not None:
                libcudnn.cudnnConvolutionBackwardBias(
                    handle, 1, gy_desc.value, cudnn.get_ptr(gy[0]),
                    1, self.bias_desc.value, cudnn.get_ptr(self.gb))

            libcudnn.cudnnConvolutionBackwardFilter(
                handle, 1, x_desc.value, cudnn.get_ptr(x[0]),
                gy_desc.value, cudnn.get_ptr(gy[0]), self.conv_desc.value,
                1, self.filter_desc.value, cudnn.get_ptr(self.gW))

            gx = cuda.empty_like(x[0])
            libcudnn.cudnnConvolutionBackwardData(
                handle, 1, self.filter_desc.value, cudnn.get_ptr(self.W),
                gy_desc.value, cudnn.get_ptr(gy[0]), self.conv_desc.value,
                0, x_desc.value, cudnn.get_ptr(gx))
        else:
            handle = cuda.get_cublas_handle()
            if self.gb is not None:
                # TODO(beam2d): Unify kernels
                with cuda.using_cumisc(handle):
                    tmp = cumisc.sum(gy[0].reshape(n * out_c, out_h * out_w), axis=1)
                    tmp = cumisc.sum(tmp.reshape(n, out_c), axis=0)
                    self.gb += tmp

            # TODO(beam2d): Use streams
            gW_mat   = self.gW.reshape(out_c, c * self.kh * self.kw)
            col_mats = self.col.reshape(n, c * self.kh * self.kw, out_h * out_w)
            gy_mats  = gy[0].reshape(n, out_c, out_h * out_w)
            for i in xrange(n):
                culinalg.add_dot(
                    gy_mats[i], col_mats[i], gW_mat, transb='T', handle=handle)

            W_mat     = self.W.reshape(out_c, c * self.kh * self.kw)
            gcol      = cuda.empty_like(self.col)
            gcol_mats = gcol.reshape(n, c * self.kh * self.kw, out_h * out_w)
            for i in xrange(n):
                culinalg.dot(
                    W_mat, gy_mats[i], transa='T', handle=handle, out=gcol_mats[i])

            gx = conv.col2im_gpu(gcol, self.sy, self.sx, self.ph, self.pw, h, w)
            
        return gx,

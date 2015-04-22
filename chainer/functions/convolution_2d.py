import math
import libcudnn
import numpy
from pycuda import gpuarray
from chainer import Function, cudnn

def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return (x, x)

_fwd_pref = libcudnn.cudnnConvolutionFwdPreference[
    'CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT']

class Convolution2D(Function):
    """Two-dimensional convolution function."""

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 wscale=1, bias=0, nobias=False):
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
            self.bias_desc = cudnn.get_conv_bias_desc(self.b)

        self.filter_desc = cudnn.get_filter4d_desc(self.W)
        self.conv_desc = cudnn.get_conv2d_desc(pad, stride)

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

    # TODO(beam2d): Implement CPU version.

    def forward_gpu(self, x):
        handle = cudnn.get_default_handle()
        x_desc = cudnn.get_tensor_desc(x[0], x[0].shape[2], x[0].shape[3])

        out_shape = libcudnn.cudnnGetConvolution2dForwardOutputDim(
            self.conv_desc.value, x_desc.value, self.filter_desc.value)
        y = gpuarray.empty(out_shape, dtype=numpy.float32)
        y_desc = cudnn.get_tensor_desc(y, y.shape[2], y.shape[3])

        algo = libcudnn.cudnnGetConvolutionForwardAlgorithm(
            handle, x_desc.value, self.filter_desc.value, self.conv_desc.value,
            y_desc.value, _fwd_pref, self.max_workspace_size)
        workspace_size = libcudnn.cudnnGetConvolutionForwardWorkspaceSize(
            handle, x_desc.value, self.filter_desc.value, self.conv_desc.value,
            y_desc.value, algo).value
        workspace = gpuarray.empty(
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

        return y,

    def backward_gpu(self, x, gy):
        handle = cudnn.get_default_handle()
        x_desc  = cudnn.get_tensor_desc( x[0],  x[0].shape[2],  x[0].shape[3])
        gy_desc = cudnn.get_tensor_desc(gy[0], gy[0].shape[2], gy[0].shape[3])

        if self.b is not None:
            libcudnn.cudnnConvolutionBackwardBias(
                handle, 1, gy_desc.value, cudnn.get_ptr(gy[0]),
                1, self.bias_desc.value, cudnn.get_ptr(self.gb))

        libcudnn.cudnnConvolutionBackwardFilter(
            handle, 1, x_desc.value, cudnn.get_ptr(x[0]),
            gy_desc.value, cudnn.get_ptr(gy[0]), self.conv_desc.value,
            1, self.filter_desc.value, cudnn.get_ptr(self.gW))

        gx = gpuarray.empty_like(x[0])
        libcudnn.cudnnConvolutionBackwardData(
            handle, 1, self.filter_desc.value, cudnn.get_ptr(self.W),
            gy_desc.value, cudnn.get_ptr(gy[0]), self.conv_desc.value,
            0, x_desc.value, cudnn.get_ptr(gx))

        return gx,

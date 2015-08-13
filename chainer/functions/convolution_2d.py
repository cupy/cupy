import math

import numpy
from six import moves

from chainer import cuda
from chainer import cudnn
from chainer import function
from chainer.utils import conv
from chainer.utils import type_check

if cudnn.available:
    from chainer.cudnn import libcudnn
    _fwd_pref = libcudnn.cudnnConvolutionFwdPreference[
        'CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT']


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return (x, x)


class Convolution2D(function.Function):

    """Two-dimensional convolution function.

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

    The filter weight has four dimensions :math:`(c_O, c_I, k_H, k_W)`
    which indicate the number of output channels, the number of input channels,
    height and width of the kernels, respectively.
    The filter weight is initialized with i.i.d. Gaussian random samples, each
    of which has zero mean and deviation :math:`\sqrt{1/(c_I k_H k_W)}` by
    default. The deviation is scaled by ``wscale`` if specified.

    The bias vector is of size :math:`c_O`.
    Each element of it is initialized by ``bias`` argument.
    If ``nobias`` argument is set to True, then this function does not hold
    the bias parameter.

    The two-dimensional convolution function is defined as follows.
    Let :math:`X` be the input tensor of dimensions :math:`(n, c_I, h, w)`,
    where :math:`n` is the batch size, and :math:`(h, w)` is spatial size of
    the input image.
    Then the ``Convolution2D`` function computes correlations between filters
    and patches of size :math:`(k_H, k_W)` in :math:`X`.
    Note that correlation here is equivalent to the inner product between
    expanded vectors.
    Patches are extracted at positions shifted by multiples of ``stride`` from
    the first position ``-pad`` for each spatial axis.
    The right-most (or bottom-most) patches do not run over the padded spatial
    size.

    Let :math:`(s_Y, s_X)` be the stride of filter application, and
    :math:`(p_H, p_W)` the spatial padding size. Then, the output size
    :math:`(h_O, w_O)` is determined by the following equations:

    .. math::

       h_O &= (h + 2p_H - k_H) / s_Y + 1,\\\\
       w_O &= (w + 2p_W - k_W) / s_X + 1.

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
                (out_channels, in_channels, self.kh, self.kw)
            self.W = initialW
        else:
            self.W = numpy.random.normal(
                0, wscale * math.sqrt(1. / (self.kh * self.kw * in_channels)),
                (out_channels, in_channels, self.kh, self.kw)
            ).astype(numpy.float32)
        if isinstance(self.W, cuda.GPUArray):
            self.gW = cuda.empty_like(self.W)
        else:
            self.gW = numpy.empty_like(self.W)

        if initial_bias is not None:
            assert initial_bias.shape == (out_channels,)
            self.b = initial_bias
        elif not nobias:
            self.b = numpy.repeat(numpy.float32(bias), out_channels)

        if self.b is not None:
            if isinstance(self.b, cuda.GPUArray):
                self.gb = cuda.empty_like(self.b)
            else:
                self.gb = numpy.empty_like(self.b)

        self.use_cudnn = use_cudnn
        if cudnn.enabled and use_cudnn:
            # chance to choose implicit-precomp-gemm algorithm
            self.max_workspace_size = in_channels * self.kh * self.kw * 4

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

            self.filter_desc = cudnn.get_filter4d_desc(self.W)
            self.conv_desc = cudnn.get_conv2d_desc(
                (self.ph, self.pw), (self.sy, self.sx))
            if self.b is not None:
                self.bias_desc = cudnn.get_conv_bias_desc(self.b)

            algo = libcudnn.cudnnGetConvolutionForwardAlgorithm(
                handle, x_desc.value, self.filter_desc.value,
                self.conv_desc.value, y_desc.value, _fwd_pref,
                self.max_workspace_size)
            workspace_size = libcudnn.cudnnGetConvolutionForwardWorkspaceSize(
                handle, x_desc.value, self.filter_desc.value,
                self.conv_desc.value, y_desc.value, algo).value
            workspace = cuda.empty(
                (max(workspace_size // 4, 1),), dtype=numpy.float32)

            libcudnn.cudnnConvolutionForward(
                handle, 1, x_desc.value, cudnn.get_ptr(x[0]),
                self.filter_desc.value, cudnn.get_ptr(self.W),
                self.conv_desc.value, algo, cudnn.get_ptr(
                    workspace), workspace_size,
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
            handle = cuda.get_cublas_handle()
            W_mat = self.W.reshape(out_c, c * self.kh * self.kw)
            col_mats = self.col.reshape(
                n, c * self.kh * self.kw, out_h * out_w)
            y_mats = y.reshape(n, out_c, out_h * out_w)
            for i in moves.range(n):
                cuda.culinalg.dot(W_mat, col_mats[i], handle=handle,
                                  out=y_mats[i])

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
            x_desc = cudnn.get_tensor_desc(x[0], h, w)
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
                    tmp = cuda.cumisc.sum(
                        gy[0].reshape(n * out_c, out_h * out_w), axis=1)
                    tmp = cuda.cumisc.sum(tmp.reshape(n, out_c), axis=0)
                    self.gb += tmp

            # TODO(beam2d): Use streams
            gW_mat = self.gW.reshape(out_c, c * self.kh * self.kw)
            col_mats = self.col.reshape(
                n, c * self.kh * self.kw, out_h * out_w)
            gy_mats = gy[0].reshape(n, out_c, out_h * out_w)
            for i in moves.range(n):
                cuda.culinalg.add_dot(
                    gy_mats[i], col_mats[i], gW_mat, transb='T', handle=handle)

            W_mat = self.W.reshape(out_c, c * self.kh * self.kw)
            gcol = cuda.empty_like(self.col)
            gcol_mats = gcol.reshape(n, c * self.kh * self.kw, out_h * out_w)
            for i in moves.range(n):
                cuda.culinalg.dot(W_mat, gy_mats[i], transa='T', handle=handle,
                                  out=gcol_mats[i])

            gx = conv.col2im_gpu(
                gcol, self.sy, self.sx, self.ph, self.pw, h, w)

        return gx,

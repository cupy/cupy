import numpy
import six

from chainer import cuda
from chainer import function
from chainer.functions.connection import convolution_2d
from chainer.utils import conv
from chainer.utils import conv_nd
from chainer.utils import type_check


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cuda.cudnn.cudnn
    _cudnn_version = libcudnn.getVersion()
    _fwd_pref = libcudnn.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
    if _cudnn_version >= 4000:
        _bwd_filter_pref = \
            libcudnn.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT
        _bwd_data_pref = \
            libcudnn.CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT


_check_cudnn_acceptable_type = convolution_2d._check_cudnn_acceptable_type


class DeconvolutionND(function.Function):

    def __init__(self, ndim, stride=1, pad=0, outsize=None, use_cudnn=True):
        self.ndim = ndim
        self.stride = conv_nd.as_tuple(stride, ndim)
        self.pad = conv_nd.as_tuple(pad, ndim)
        self.use_cudnn = use_cudnn
        if outsize is not None:
            assert len(outsize) == ndim
        self.outs = outsize

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)
        x_type, w_type = in_types[:2]

        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim == self.ndim + 2,
            w_type.ndim == self.ndim + 2,
            x_type.shape[1] == w_type.shape[0]
        )

        if self.outs is not None:
            for i, (out, s, p) in enumerate(zip(
                    self.outs, self.stride, self.pad)):
                type_check.expect(
                    x_type.shape[i + 2] ==
                    conv.get_conv_outsize(out, w_type.shape[i + 2], s, p)
                )

        if n_in.eval() == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[1]
            )

    def _use_cudnn(self, x, W):
        return (cuda.cudnn_enabled and
                self.use_cudnn and
                self.ndim > 1 and
                _check_cudnn_acceptable_type(x.dtype, W.dtype))

    def _forward_xp(self, x, W, b, xp):
        ndim = self.ndim
        ksize = W.shape[2:]     # W: C_I, C_O, k_1, k_2, ..., k_N
        dims = x.shape[2:]      # x: n, C_I, d_1, d_2, ..., d_N
        stride = self.stride
        pad = self.pad

        # gcol: C_O, k_1, ..., k_N, n, d_1, ..., d_N
        gcol = xp.tensordot(W, x, (0, 1)).astype(x.dtype, copy=False)
        # Roll n, which is batch size, before the first.
        gcol = xp.rollaxis(gcol, ndim + 1)

        if self.outs is None:
            self.outs = tuple(
                conv.get_deconv_outsize(d, k, s, p)
                for d, k, s, p in zip(dims, ksize, stride, pad))
            assert all(out > 0 for out in self.outs), \
                'Output sizes should be positive.'
        # y: n, C_O, d_1, d_2, ..., d_N
        if xp is numpy:
            y = conv_nd.col2im_nd_cpu(gcol, stride, pad, self.outs)
        else:
            y = conv_nd.col2im_nd_gpu(gcol, stride, pad, self.outs)
        if b is not None:
            b_shape = (1, -1) + (1,) * ndim
            y += b.reshape(b_shape)

        return y,

    def _forward_cudnn(self, x, W, b):
        c = W.shape[1]          # W: C_I, C_O, k_1, k_2, ..., k_N
        ksize = W.shape[2:]
        n, in_c = x.shape[:2]   # x: n, C_I, d_1, d_2, ..., d_N
        dims = x.shape[2:]
        ndim = self.ndim
        colon = slice(None)

        # Make empty array for output.
        if self.outs is None:
            self.outs = tuple(
                conv.get_deconv_outsize(d, k, s, p)
                for d, k, s, p in zip(dims, ksize, self.stride, self.pad))
            assert all(out > 0 for out in self.outs), \
                'Output sizes should be positive.'
        y_shape = (n, c) + self.outs  # (n, c_O, out_1, out_2, ..., out_N)
        y = cuda.cupy.empty(y_shape, dtype=x.dtype)

        # Convert to C-contiguous arrays.
        x = cuda.cupy.ascontiguousarray(x)
        W = cuda.cupy.ascontiguousarray(W)
        if b is not None:
            b = cuda.cupy.ascontiguousarray(b)

        # Get cuDNN handler and descriptors.
        handle = cudnn.get_handle()
        x_desc = cudnn.create_tensor_descriptor(x)
        y_desc = cudnn.create_tensor_descriptor(y)
        self.filter_desc = cudnn.create_filter_descriptor(W)
        self.conv_desc = cudnn.create_convolution_descriptor(
            self.pad, self.stride, x.dtype)
        if b is not None:
            b_index = (None, colon) + (None,) * ndim
            self.bias_desc = cudnn.create_tensor_descriptor(b[b_index])

        # cuDNN forward computation.
        oz_dtype = 'd' if x.dtype == 'd' else 'f'
        one = numpy.array(1, dtype=oz_dtype).ctypes
        zero = numpy.array(0, dtype=oz_dtype).ctypes
        if _cudnn_version >= 4000:
            workspace_size = cuda.get_max_workspace_size()
            workspace = cuda.cupy.empty((workspace_size,), dtype='b')
            algo = libcudnn.getConvolutionBackwardDataAlgorithm(
                handle, self.filter_desc.value, x_desc.value,
                self.conv_desc.value, y_desc.value, _bwd_data_pref,
                workspace_size)
            libcudnn.convolutionBackwardData_v3(
                handle, one.data, self.filter_desc.value, W.data.ptr,
                x_desc.value, x.data.ptr, self.conv_desc.value,
                algo, workspace.data.ptr, workspace_size,
                zero.data, y_desc.value, y.data.ptr)
        else:
            libcudnn.convolutionBackwardData_v2(
                handle, one.data, self.filter_desc.value, W.data.ptr,
                x_desc.value, x.data.ptr, self.conv_desc.value,
                zero.data, y_desc.value, y.data.ptr)

        # Add bias if given.
        # TODO(takagi) Support unshared bias
        if b is not None:
            if _cudnn_version >= 3000 or ndim == 2:
                cudnn.add_tensor(
                    handle, one.data, self.bias_desc.value, b.data.ptr,
                    one.data, y_desc.value, y.data.ptr)
            else:
                # cuDNN v2 does not seem to support bias addition in spatial
                # dimensions other than two.
                b_index = (None, colon) + (None,) * ndim
                y += b[b_index]

        return y,

    def forward(self, inputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None

        xp = cuda.get_array_module(*inputs)
        if xp is numpy:
            return self._forward_xp(x, W, b, numpy)
        elif self._use_cudnn(x, W):
            return self._forward_cudnn(x, W, b)
        else:
            return self._forward_xp(x, W, b, cuda.cupy)

    def _backward_xp(self, x, W, b, gy, xp):
        ndim = self.ndim
        ksize = W.shape[2:]
        stride = self.stride
        pad = self.pad
        if xp is numpy:
            col = conv_nd.im2col_nd_cpu(gy, ksize, stride, pad)
        else:
            col = conv_nd.im2col_nd_gpu(gy, ksize, stride, pad)

        # x  : n, C_I, d_1, d_2, ..., d_N
        # col: n, C_I, k_1, k_2, ..., k_N, d_1, d_2, ..., d_N
        x_axes = (0,) + tuple(six.moves.range(2, ndim + 2))
        col_axes = (0,) + tuple(six.moves.range(ndim + 2, ndim * 2 + 2))
        gW = xp.tensordot(x, col, (x_axes, col_axes)).astype(
            W.dtype, copy=False)

        # col: n, C_I, k_1, k_2, ..., k_N, d_1, d_2, ..., d_N
        # W  : C_I, C_O, k_1, k_2, ..., k_N
        axes = (1,) + tuple(six.moves.range(2, ndim + 2))
        gx = xp.tensordot(col, W, (axes, axes)).astype(x.dtype, copy=False)
        gx = xp.rollaxis(gx, ndim + 1, 1)

        if b is None:
            return gx, gW
        else:
            sum_axis = (0,) + tuple(six.moves.range(2, ndim + 2))
            gb = gy.sum(axis=sum_axis)
            return gx, gW, gb

    def _backward_cudnn(self, x, W, b, gy):
        # Convert to C-contiguous arrays.
        x = cuda.cupy.ascontiguousarray(x)
        W = cuda.cupy.ascontiguousarray(W)
        gy = cuda.cupy.ascontiguousarray(gy)

        # Make empty arrays for results.
        gx = cuda.cupy.empty_like(x)
        gW = cuda.cupy.empty_like(W)

        # Get cuDNN handler and descriptors.
        handle = cudnn.get_handle()
        gy_desc = cudnn.create_tensor_descriptor(gy)
        gx_desc = cudnn.create_tensor_descriptor(gx)

        # Chance to choose implicit-precom-gemm algorithm.
        workspace_size = cuda.get_max_workspace_size()
        algo = libcudnn.getConvolutionForwardAlgorithm(
            handle, gy_desc.value, self.filter_desc.value,
            self.conv_desc.value, gx_desc.value, _fwd_pref,
            workspace_size)
        workspace = cuda.cupy.empty((workspace_size,), dtype='b')

        # Compute input gradient.
        oz_dtype = 'd' if x.dtype == 'd' else 'f'
        one = numpy.array(1, dtype=oz_dtype).ctypes
        zero = numpy.array(0, dtype=oz_dtype).ctypes
        libcudnn.convolutionForward(
            handle, one.data, gy_desc.value, gy.data.ptr,
            self.filter_desc.value, W.data.ptr,
            self.conv_desc.value, algo, workspace.data.ptr, workspace_size,
            zero.data, gx_desc.value, gx.data.ptr)

        # Compute bias gradient.
        if b is not None:
            if _cudnn_version >= 3000 or self.ndim == 2:
                gb = cuda.cupy.empty_like(b)
                libcudnn.convolutionBackwardBias(
                    handle, one.data, gy_desc.value, gy.data.ptr,
                    zero.data, self.bias_desc.value, gb.data.ptr)
            else:
                # cuDNN v2 does not seem to support bias backward in spatial
                # dimensions other than two.

                # (n, _, out_1, out_2, ..., out_N)
                axis = (0,) + tuple(six.moves.range(2, self.ndim + 2))
                gb = gy.sum(axis=axis)

        # Compute filter gradient.
        if _cudnn_version >= 4000:
            algo = libcudnn.getConvolutionBackwardFilterAlgorithm(
                handle, gy_desc.value, gx_desc.value,
                self.conv_desc.value, self.filter_desc.value,
                _bwd_filter_pref, workspace_size)

            libcudnn.convolutionBackwardFilter_v3(
                handle, one.data, gy_desc.value, gy.data.ptr,
                gx_desc.value, x.data.ptr, self.conv_desc.value,
                algo, workspace.data.ptr, workspace_size,
                zero.data, self.filter_desc.value, gW.data.ptr)
        else:
            libcudnn.convolutionBackwardFilter_v2(
                handle, one.data, gy_desc.value, gy.data.ptr,
                gx_desc.value, x.data.ptr, self.conv_desc.value,
                zero.data, self.filter_desc.value, gW.data.ptr)

        if b is None:
            return gx, gW
        else:
            return gx, gW, gb

    def backward(self, inputs, grad_outputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        gy = grad_outputs[0]

        xp = cuda.get_array_module(*inputs)
        if xp is numpy:
            return self._backward_xp(x, W, b, gy, numpy)
        elif self._use_cudnn(x, W):
            return self._backward_cudnn(x, W, b, gy)
        else:
            return self._backward_xp(x, W, b, gy, cuda.cupy)


def deconvolution_nd(x, W, b=None, stride=1, pad=0, outsize=None,
                     use_cudnn=True):
    """N-dimensional deconvolution function.

    This is an implementation of N-dimensional deconvolution which generalizes
    two-dimensional one. It takes three variables: input ``x``, the filter
    weight ``W``, and the bias vector ``b``.

    Args:
        x (chainer.Variable or :class:`numpy.ndarray` or cupy.ndarray):
            Input data of shape :math:`(n, c_I, d_1, d_2, ..., d_N)`.
        W (chainer.Variable or :class:`numpy.ndarray` or cupy.ndarray):
            Weight data of shape :math:`(c_I, c_O, k_1, k_2, ..., k_N)`.
        b (chainer.Variable or :class:`numpy.ndarray` or cupy.ndarray):
            Bias vector of length :math:`c_O` (optional).
        stride (int or tuple of ints): Stride of filter applications
            :math:`(s_1, s_2, ..., s_N)`. ``stride=s`` is equivalent to
            ``(s, s, ..., s)``.
        pad (int or tuple of ints): Spatial padding size for input arrays
            :math:`(p_1, p_2, ..., p_N)`. ``pad=p`` is equivalent to
            ``(p, p, ..., p)``.
        outsize (tuple of ints): Expected output size of deconvolutional
            operation. It should be a tuple of ints
            :math:`(out_1, out_2, ..., out_N)`. Default value is ``None`` and
            the outsize is estimated by input size, stride and pad.
        use_cudnn (bool): If ``True``, then this function uses cuDNN if
            available. Note that cuDNN supports more than one-dimensional
            deconvolution operations only.

    Returns:
        ~chainer.Variable: Output variable.

    The filter weight has the following dimensions
    :math:`(c_I, c_O, k_1, k_2, ..., k_N)` which indicate the number of input
    channels, that of output channels and the filter's spatial sizes,
    respectively.

    The one-dimensional bias vector is of size :math:`c_O`.

    Let :math:`X` be the input tensor of dimensions
    :math:`(n, c_I, d_1, d_2, ..., d_N)`, :math:`(s_1, s_2, ..., s_N)` the
    stride of filter applications, and :math:`(p_1, p_2, ..., p_N)` the spacial
    padding size. Then the output size :math:`(out_1, out_2, ..., out_N)` is
    determined by the following equations:

    .. math::

        out_1 &= s_1 (d_1 - 1) + k_1 - 2 p_1,\\\\
        out_2 &= s_2 (d_2 - 1) + k_2 - 2 p_2,\\\\
        ...,\\\\
        out_N &= s_N (d_N - 1) + k_N - 2 p_N.

    .. seealso:: :class:`links.DeconvolutionND`, :func:`deconvolution_2d`
    """
    ndim = len(x.shape[2:])
    func = DeconvolutionND(ndim, stride, pad, outsize, use_cudnn)
    if b is None:
        return func(x, W)
    else:
        return func(x, W, b)

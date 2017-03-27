import numpy

from six import moves

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


class ConvolutionND(function.Function):

    def __init__(self, ndim, stride=1, pad=0, use_cudnn=True, cover_all=False):
        self.ndim = ndim
        self.stride = conv_nd.as_tuple(stride, ndim)
        self.pad = conv_nd.as_tuple(pad, ndim)
        self.use_cudnn = use_cudnn
        self.cover_all = cover_all

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)

        x_type = in_types[0]
        w_type = in_types[1]
        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim == self.ndim + 2,
            w_type.ndim == self.ndim + 2,
            x_type.shape[1] == w_type.shape[1],
        )

        if n_in.eval() == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    def _use_cudnn(self, x, W):
        return (not self.cover_all and
                cuda.cudnn_enabled and
                self.use_cudnn and
                self.ndim > 1 and
                _check_cudnn_acceptable_type(x.dtype, W.dtype))

    def _forward_xp(self, x, W, b, xp):
        ndim = self.ndim
        ksize = W.shape[2:]
        stride = self.stride
        pad = self.pad

        # Make patch array.
        if xp is numpy:
            self.col = conv_nd.im2col_nd_cpu(
                x, ksize, stride, pad, cover_all=self.cover_all)
        else:
            self.col = conv_nd.im2col_nd_gpu(
                x, ksize, stride, pad, cover_all=self.cover_all)

        # Compute correlation.
        axes = tuple(moves.range(1, ndim + 2))  # (1, 2, ..., N+1)
        y = xp.tensordot(self.col, W, (axes, axes)).astype(x.dtype, copy=False)

        # Apply bias if given.
        if b is not None:
            y += b

        # Roll c_O before the second in (n, y_1, y_2, ..., y_N, c_O).
        return xp.rollaxis(y, ndim + 1, 1),

    def _forward_cudnn(self, x, W, b):
        out_c = W.shape[0]      # (c_O, _, k_1, k_2, ..., k_N)
        ksize = W.shape[2:]
        n, c = x.shape[:2]      # (n, c_I, d_1, d_2, ..., d_N)
        dims = x.shape[2:]
        stride = self.stride
        pad = self.pad
        ndim = self.ndim
        colon = slice(None)

        # Make empty array for result.
        outs = tuple(
            conv.get_conv_outsize(d, k, s, p, cover_all=self.cover_all)
            for (d, k, s, p) in zip(dims, ksize, stride, pad))
        assert all(out > 0 for out in outs), 'Output sizes should be positive.'
        y_shape = (n, out_c) + outs  # (n, c_O, out_1, out_2, ..., out_N)
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
            pad, stride, x.dtype)
        if b is not None:
            b_index = (None, colon) + (None,) * ndim
            self.bias_desc = cudnn.create_tensor_descriptor(b[b_index])

        # Find cuDNN algorithm to be used.
        workspace_size = cuda.get_max_workspace_size()
        workspace = cuda.cupy.empty((workspace_size,), dtype='b')
        algo = libcudnn.getConvolutionForwardAlgorithm(
            handle, x_desc.value, self.filter_desc.value,
            self.conv_desc.value, y_desc.value, _fwd_pref,
            workspace_size)

        # cuDNN forward computation.
        oz_dtype = 'd' if x.dtype == 'd' else 'f'
        one = numpy.array(1, dtype=oz_dtype).ctypes
        zero = numpy.array(0, dtype=oz_dtype).ctypes
        libcudnn.convolutionForward(
            handle, one.data, x_desc.value, x.data.ptr,
            self.filter_desc.value, W.data.ptr, self.conv_desc.value,
            algo, workspace.data.ptr, workspace_size, zero.data,
            y_desc.value, y.data.ptr)

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

        if not type_check.same_types(*inputs):
            if b is not None:
                raise ValueError('numpy and cupy must not be used together\n'
                                 'type(W): {0}, type(x): {1}, type(b): {2}'
                                 .format(type(W), type(x), type(b)))
            else:
                raise ValueError('numpy and cupy must not be used together\n'
                                 'type(W): {0}, type(x): {1}'
                                 .format(type(W), type(x)))

        xp = cuda.get_array_module(*inputs)
        if xp is numpy:
            return self._forward_xp(x, W, b, numpy)
        elif not self._use_cudnn(x, W):
            return self._forward_xp(x, W, b, cuda.cupy)
        else:
            return self._forward_cudnn(x, W, b)

    def _backward_xp(self, x, W, b, gy, xp):
        dims = x.shape[2:]     # (n, c_I, d_1, d_2, ..., d_N)
        stride = self.stride
        pad = self.pad
        ndim = self.ndim

        # Compute filter weight gradient.
        # (n, _, out_1, out_2, ..., out_N)
        out_axes = (0,) + tuple(moves.range(2, ndim + 2))
        # (n, _, _, ..., _, out_1, out_2, ..., out_N)
        col_axes = (0,) + tuple(moves.range(ndim + 2, ndim * 2 + 2))
        gW = xp.tensordot(gy, self.col, (out_axes, col_axes)).astype(
            W.dtype, copy=False)

        # Compute patch array gradient.
        gcol = xp.tensordot(W, gy, (0, 1)).astype(x.dtype, copy=False)
        gcol = xp.rollaxis(gcol, ndim + 1)

        # Compute input gradient.
        if xp is numpy:
            gx = conv_nd.col2im_nd_cpu(gcol, stride, pad, dims)
        else:
            gx = conv_nd.col2im_nd_gpu(gcol, stride, pad, dims)

        # Compute bias gradient if given and return gradients.
        if b is None:
            return gx, gW
        else:
            # (n, _, out_1, out_2, ..., out_N)
            axis = (0,) + tuple(moves.range(2, ndim + 2))
            gb = gy.sum(axis=axis)
            return gx, gW, gb

    def _backward_cudnn(self, x, W, b, gy):
        # Convert to C-contiguous arrays.
        x = cuda.cupy.ascontiguousarray(x)
        W = cuda.cupy.ascontiguousarray(W)
        gy = cuda.cupy.ascontiguousarray(gy)

        # Make empty arrays for result.
        gx = cuda.cupy.empty_like(x)
        gW = cuda.cupy.empty_like(W)

        # Get cuDNN handler and descriptors.
        handle = cudnn.get_handle()
        x_desc = cudnn.create_tensor_descriptor(x)
        gy_desc = cudnn.create_tensor_descriptor(gy)

        # Compute gradients.
        oz_dtype = 'd' if x.dtype == 'd' else 'f'
        one = numpy.array(1, dtype=oz_dtype).ctypes
        zero = numpy.array(0, dtype=oz_dtype).ctypes
        if _cudnn_version >= 4000:
            workspace_size = cuda.get_max_workspace_size()
            workspace = cuda.cupy.empty((workspace_size,), dtype='b')

            # Compute filter weight gradient.
            algo = libcudnn.getConvolutionBackwardFilterAlgorithm(
                handle, x_desc.value, gy_desc.value,
                self.conv_desc.value, self.filter_desc.value,
                _bwd_filter_pref, workspace_size)
            libcudnn.convolutionBackwardFilter_v3(
                handle, one.data, x_desc.value, x.data.ptr,
                gy_desc.value, gy.data.ptr, self.conv_desc.value,
                algo, workspace.data.ptr, workspace_size,
                zero.data, self.filter_desc.value, gW.data.ptr)

            # Compute input gradient.
            algo = libcudnn.getConvolutionBackwardDataAlgorithm(
                handle, self.filter_desc.value, gy_desc.value,
                self.conv_desc.value, x_desc.value, _bwd_data_pref,
                workspace_size)
            libcudnn.convolutionBackwardData_v3(
                handle, one.data, self.filter_desc.value, W.data.ptr,
                gy_desc.value, gy.data.ptr, self.conv_desc.value,
                algo, workspace.data.ptr, workspace_size,
                zero.data, x_desc.value, gx.data.ptr)
        else:
            # Compute input and filter weight gradients.
            libcudnn.convolutionBackwardFilter_v2(
                handle, one.data, x_desc.value, x.data.ptr,
                gy_desc.value, gy.data.ptr, self.conv_desc.value,
                zero.data, self.filter_desc.value, gW.data.ptr)
            libcudnn.convolutionBackwardData_v2(
                handle, one.data, self.filter_desc.value, W.data.ptr,
                gy_desc.value, gy.data.ptr, self.conv_desc.value,
                zero.data, x_desc.value, gx.data.ptr)

        # Compute bias gradient if given and return gradients.
        if b is None:
            return gx, gW
        elif _cudnn_version >= 3000 or self.ndim == 2:
            gb = cuda.cupy.empty_like(b)
            libcudnn.convolutionBackwardBias(
                handle, one.data, gy_desc.value, gy.data.ptr,
                zero.data, self.bias_desc.value, gb.data.ptr)
            return gx, gW, gb
        else:
            # cuDNN v2 does not seem to support bias backward in spatial
            # dimensions other than two.

            # (n, _, out_1, out_2, ..., out_N)
            axis = (0,) + tuple(moves.range(2, self.ndim + 2))
            gb = gy.sum(axis=axis)
            return gx, gW, gb

    def backward(self, inputs, grad_outputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None

        if not type_check.same_types(*inputs):
            if b is not None:
                raise ValueError('numpy and cupy must not be used together\n'
                                 'type(W): {0}, type(x): {1}, type(b): {2}'
                                 .format(type(W), type(x), type(b)))
            else:
                raise ValueError('numpy and cupy must not be used together\n'
                                 'type(W): {0}, type(x): {1}'
                                 .format(type(W), type(x)))

        gy = grad_outputs[0]    # (n, c_O, out_1, out_2, ..., out_N)

        xp = cuda.get_array_module(*inputs)
        if xp is numpy:
            return self._backward_xp(x, W, b, gy, numpy)
        elif not self._use_cudnn(x, W):
            return self._backward_xp(x, W, b, gy, cuda.cupy)
        else:
            return self._backward_cudnn(x, W, b, gy)


def convolution_nd(x, W, b=None, stride=1, pad=0, use_cudnn=True,
                   cover_all=False):
    """N-dimensional convolution function.

    This is an implementation of N-dimensional convolution which is generalized
    two-dimensional convolution in ConvNets. It takes three variables: the
    input ``x``, the filter weight ``W`` and the bias vector ``b``.

    Notation: here is a notation for dimensionalities.

    - :math:`N` is the number of spatial dimensions.
    - :math:`n` is the batch size.
    - :math:`c_I` and :math:`c_O` are the number of the input and output
      channels, respectively.
    - :math:`d_1, d_2, ..., d_N` are the size of each axis of the input's
      spatial dimensions, respectively.
    - :math:`k_1, k_2, ..., k_N` are the size of each axis of the filters,
      respectively.

    Args:
        x (~chainer.Variable): Input variable of shape
            :math:`(n, c_I, d_1, d_2, ..., d_N)`.
        W (~chainer.Variable): Weight variable of shape
            :math:`(c_O, c_I, k_1, k_2, ..., k_N)`.
        b (~chainer.Variable): One-dimensional bias variable with length
            :math:`c_O` (optional).
        stride (int or tuple of ints): Stride of filter applications
            :math:`(s_1, s_2, ..., s_N)`. ``stride=s`` is equivalent to
            ``(s, s, ..., s)``.
        pad (int or tuple of ints): Spatial padding width for input arrays
            :math:`(p_1, p_2, ..., p_N)`. ``pad=p`` is equivalent to
            ``(p, p, ..., p)``.
        use_cudnn (bool): If ``True``, then this function uses cuDNN if
            available. See below for the excact conditions.
        cover_all (bool): If ``True``, all spatial locations are convoluted
            into some output pixels. It may make the output size larger.
            `cover_all` needs to be ``False`` if you want to use cuDNN.

    Returns:
        ~chainer.Variable: Output variable.

    This function uses cuDNN implementation for its forward and backward
    computation if ALL of the following conditions are satisfied:

    - ``cuda.cudnn_enabled`` is ``True``
    - ``use_cudnn`` is ``True``
    - The number of spatial dimensions is more than one.
    - ``cover_all`` is ``False``
    - The input's ``dtype`` is equal to the filter weight's.
    - The ``dtype`` is FP32, FP64 or FP16(cuDNN version is equal to or greater
      than v3)


    .. seealso:: :class:`~chainer.links.ConvolutionND`, :func:`convolution_2d`
    """
    ndim = len(x.shape[2:])
    func = ConvolutionND(ndim, stride, pad, use_cudnn, cover_all)
    if b is None:
        return func(x, W)
    else:
        return func(x, W, b)

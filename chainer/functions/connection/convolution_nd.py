import numpy
import operator

from functools import reduce
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

    def forward_cpu(self, inputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        ndim = self.ndim
        ksize = W.shape[2:]
        stride = self.stride
        pad = self.pad

        # Make patch array.
        self.col = conv_nd.im2col_nd_cpu(
            x, ksize, stride, pad, cover_all=self.cover_all)

        # Compute correlation.
        axes = tuple(moves.range(1, ndim + 2))  # (1, 2, ..., N+1)
        y = numpy.tensordot(self.col, W, (axes, axes)).astype(x.dtype)

        # Apply bias if given.
        if b is not None:
            y += b

        # Roll c_O before the second in (n, y_1, y_2, ..., y_N, c_O).
        return numpy.rollaxis(y, ndim + 1, 1),

    def forward_gpu(self, inputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        out_c = W.shape[0]      # (c_O, _, k_1, k_2, ..., k_N)
        ksize = W.shape[2:]
        n, c = x.shape[:2]      # (n, c_I, d_1, d_2, ..., d_N)
        dims = x.shape[2:]
        stride = self.stride
        pad = self.pad
        ndim = self.ndim
        colon = slice(None)

        # Compute output image's dimensions.
        outs = tuple(
            conv.get_conv_outsize(d, k, s, p, cover_all=self.cover_all)
            for (d, k, s, p) in zip(dims, ksize, stride, pad))

        # Make empty array for result.
        y_shape = (n, out_c) + outs  # (n, c_O, out_1, out_2, ..., out_N)
        y = cuda.cupy.empty(y_shape, dtype=x.dtype)
        # Implementation using cuDNN.
        if (not self.cover_all and cuda.cudnn_enabled and self.use_cudnn and
            ndim > 1 and convolution_2d._check_cudnn_acceptable_type(
                x.dtype, W.dtype)):
            x = cuda.cupy.ascontiguousarray(x)
            W = cuda.cupy.ascontiguousarray(W)
            if b is not None:
                b = cuda.cupy.ascontiguousarray(b)

            handle = cudnn.get_handle()
            x_desc = cudnn.create_tensor_descriptor(x)
            y_desc = cudnn.create_tensor_descriptor(y)

            self.filter_desc = cudnn.create_filter_descriptor(W)
            self.conv_desc = cudnn.create_convolution_descriptor(
                pad, stride)
            if b is not None:
                b_index = (None, colon) + (None,) * ndim
                self.bias_desc = cudnn.create_tensor_descriptor(b[b_index])

            workspace_size = cuda.get_max_workspace_size()
            workspace = cuda.cupy.empty((workspace_size,), dtype='b')
            algo = libcudnn.getConvolutionForwardAlgorithm(
                handle, x_desc.value, self.filter_desc.value,
                self.conv_desc.value, y_desc.value, _fwd_pref,
                workspace_size)

            oz_dtype = 'd' if x.dtype == 'd' else 'f'
            one = numpy.array(1, dtype=oz_dtype).ctypes
            zero = numpy.array(0, dtype=oz_dtype).ctypes
            libcudnn.convolutionForward(
                handle, one.data, x_desc.value, x.data.ptr,
                self.filter_desc.value, W.data.ptr, self.conv_desc.value,
                algo, workspace.data.ptr, workspace_size, zero.data,
                y_desc.value, y.data.ptr)

            # TODO(takagi) Support unshared bias
            if b is not None:
                cudnn.add_tensor(
                    handle, one.data, self.bias_desc.value, b.data.ptr,
                    one.data, y_desc.value, y.data.ptr)
        # Implementation using im2col.
        else:
            # Make patch array.
            self.col = conv_nd.im2col_nd_gpu(
                x, ksize, stride, pad, cover_all=self.cover_all)

            # Compute correlation.
            W_mat = W.reshape(out_c, -1)
            col_mats = self.col.reshape(n, -1, reduce(operator.mul, outs))
            y_mats = y.reshape(n, out_c, -1)
            # TODO(takagi): Use streams or batch gemm
            for i in moves.range(n):
                y_mats[i] = W_mat.dot(col_mats[i])

            # Apply bias if given.
            # TODO(takagi): Support unshared bias
            if b is not None:
                index = (colon,) + (None,) * ndim  # (:, None, ..., None)
                y += b[index]

        return y,

    def backward_cpu(self, inputs, grad_outputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        gy = grad_outputs[0]   # (n, c_O, out_1, out_2, ..., out_N)
        dims = x.shape[2:]     # (n, c_I, d_1, d_2, ..., d_N)
        stride = self.stride
        pad = self.pad
        ndim = self.ndim

        # Compute filter weight gradient.
        # (n, _, out_1, out_2, ..., out_N)
        out_axes = (0,) + tuple(moves.range(2, ndim + 2))
        # (n, _, _, ..., _, out_1, out_2, ..., out_N)
        col_axes = (0,) + tuple(moves.range(ndim + 2, ndim * 2 + 2))
        gW = numpy.tensordot(
            gy, self.col, (out_axes, col_axes)).astype(W.dtype)

        # Compute patch array gradient.
        gcol = numpy.tensordot(W, gy, (0, 1)).astype(x.dtype)
        gcol = numpy.rollaxis(gcol, ndim + 1)

        # Compute input gradient.
        gx = conv_nd.col2im_nd_cpu(gcol, stride, pad, dims)

        # Compute bias gradient if given and return gradients.
        if b is None:
            return gx, gW
        else:
            # (n, _, out_1, out_2, ..., out_N)
            axis = (0,) + tuple(moves.range(2, ndim + 2))
            gb = gy.sum(axis=axis)
            return gx, gW, gb

    def backward_gpu(self, inputs, grad_putputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        gy = grad_putputs[0]    # (n, c_O, out_1, out_2, ..., out_N)
        out_c = gy.shape[1]
        outs = gy.shape[2:]
        n, c = x.shape[:2]      # (n, c_I, d_1, d_2, ..., d_N)
        dims = x.shape[2:]
        ksize = W.shape[2:]     # (_, _, k_1, k_2, ..., k_N)
        stride = self.stride
        pad = self.pad
        ndim = self.ndim

        # Compute filter weight gradient.
        gW = cuda.cupy.empty_like(W)
        # Implementation using cuDNN.
        if (not self.cover_all and cuda.cudnn_enabled and self.use_cudnn and
            ndim > 1 and convolution_2d._check_cudnn_acceptable_type(
                x.dtype, W.dtype)):
            x = cuda.cupy.ascontiguousarray(x)
            W = cuda.cupy.ascontiguousarray(W)
            gy = cuda.cupy.ascontiguousarray(gy)

            handle = cudnn.get_handle()
            x_desc = cudnn.create_tensor_descriptor(x)
            gy_desc = cudnn.create_tensor_descriptor(gy)
            oz_dtype = 'd' if x.dtype == 'd' else 'f'
            one = numpy.array(1, dtype=oz_dtype).ctypes
            zero = numpy.array(0, dtype=oz_dtype).ctypes
            gx = cuda.cupy.empty_like(x)

            if _cudnn_version >= 4000:
                workspace_size = cuda.get_max_workspace_size()
                workspace = cuda.cupy.empty((workspace_size,), dtype='b')

                algo = libcudnn.getConvolutionBackwardFilterAlgorithm(
                    handle, x_desc.value, gy_desc.value,
                    self.conv_desc.value, self.filter_desc.value,
                    _bwd_filter_pref, workspace_size)
                libcudnn.convolutionBackwardFilter_v3(
                    handle, one.data, x_desc.value, x.data.ptr,
                    gy_desc.value, gy.data.ptr, self.conv_desc.value,
                    algo, workspace.data.ptr, workspace_size,
                    zero.data, self.filter_desc.value, gW.data.ptr)

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
                libcudnn.convolutionBackwardFilter_v2(
                    handle, one.data, x_desc.value, x.data.ptr,
                    gy_desc.value, gy.data.ptr, self.conv_desc.value,
                    zero.data, self.filter_desc.value, gW.data.ptr)
                libcudnn.convolutionBackwardData_v2(
                    handle, one.data, self.filter_desc.value, W.data.ptr,
                    gy_desc.value, gy.data.ptr, self.conv_desc.value,
                    zero.data, x_desc.value, gx.data.ptr)

            if b is not None:
                gb = cuda.cupy.empty_like(b)
                libcudnn.convolutionBackwardBias(
                    handle, one.data, gy_desc.value, gy.data.ptr,
                    zero.data, self.bias_desc.value, gb.data.ptr)
        # Implementation using col2im.
        else:
            gW_mat = gW.reshape(out_c, reduce(operator.mul, ksize, c))
            col_mats = self.col.reshape(
                n, reduce(operator.mul, ksize, c), reduce(operator.mul, outs))
            gy_mats = gy.reshape(n, out_c, reduce(operator.mul, outs))
            # TODO(takagi): Use streams or batch gemm
            gW_mat[...] = 0
            for i in moves.range(n):
                gW_mat += cuda.cupy.dot(gy_mats[i], col_mats[i].T)

            # Compute patch array gradient.
            W_mat = W.reshape(out_c, -1)
            gcol = cuda.cupy.empty_like(self.col)
            gcol_mats = gcol.reshape(
                n, reduce(operator.mul, ksize, c), reduce(operator.mul, outs))
            for i in moves.range(n):
                gcol_mats[i] = cuda.cupy.dot(W_mat.T, gy_mats[i])

            # Compute input gradient.
            gx = conv_nd.col2im_nd_gpu(gcol, stride, pad, dims)

            # Compute bias gradient if given.
            if b is not None:
                # (n, _, out_1, out_2, ..., out_N)
                axis = (0,) + tuple(moves.range(2, ndim + 2))
                gb = gy.sum(axis=axis)

        if b is None:
            return gx, gW
        else:
            return gx, gW, gb


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
            available. cuDNN supports more than one-dimensional convolution.
        cover_all (bool): If ``True``, all spatial locations are convoluted
            into some output pixels. It may make the output size larger.

    Returns:
        ~chainer.Variable: Output variable.

    .. seealso:: :class:`ConvolutionND`, :func:`convolution_2d`
    """
    ndim = len(x.data.shape[2:])
    func = ConvolutionND(ndim, stride, pad, use_cudnn, cover_all)
    if b is None:
        return func(x, W)
    else:
        return func(x, W, b)

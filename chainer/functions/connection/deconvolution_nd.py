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
        if b is not None:
            b = cuda.cupy.ascontiguousarray(b)

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

        if not type_check.same_types(*inputs):
            if b is not None:
                raise ValueError('numpy and cupy must not be used together\n'
                                 'type(W): {0}, type(x): {1}, type(b): {2}'
                                 .format(type(W), type(x), type(b)))
            else:
                raise ValueError('numpy and cupy must not be used together\n'
                                 'type(W): {0}, type(x): {1}'
                                 .format(type(W), type(x)))

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
    two-dimensional one. In most of deep learning frameworks and papers, this
    function is called **transposed convolution**. But because of historical
    reasons (e.g. paper by Ziller `Deconvolutional Networks`_) and backward
    compatibility, this function is called **deconvolution** in Chainer.

    .. _Deconvolutional Networks: \
http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf

    It takes three variables: the input ``x``, the filter weight ``W``, and the
    bias vector ``b``.

    Notation: here is a notation for dimensionalities.

    - :math:`N` is the number of spatial dimensions.
    - :math:`n` is the batch size.
    - :math:`c_I` and :math:`c_O` are the number of the input and output
      channels, respectively.
    - :math:`d_1, d_2, ..., d_N` are the size of each axis of the input's
      spatial dimensions, respectively.
    - :math:`k_1, k_2, ..., k_N` are the size of each axis of the filters,
      respectively.
    - :math:`p_1, p_2, ..., p_N` are the size of each axis of the spatial
      padding size, respectively.
    - :math:`s_1, s_2, ..., s_N` are the stride of each axis of filter
      application, respectively.

    If ``outsize`` option is ``None``, the output size
    :math:`(l_1, l_2, ..., l_N)` is determined by the following equations with
    the items in the above list:

    .. math::

       l_n = s_n (d_n - 1)  + k_n - 2 p_n \ \ (n = 1, ..., N)

    If ``outsize`` option is given, the output size is determined by
    ``outsize``. In this case, the ``outsize`` :math:`(l_1, l_2, ..., l_N)`
    must satisfy the following equations:

    .. math::

       d_n = \\lfloor (l_n + 2p_n - k_n) / s_n \\rfloor + 1 \ \ (n = 1, ..., N)

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable of shape :math:`(n, c_I, d_1, d_2, ..., d_N)`.
        W (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Weight variable of shape :math:`(c_I, c_O, k_1, k_2, ..., k_N)`.
        b (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            One-dimensional bias variable with length :math:`c_O` (optional).
        stride (:class:`int` or :class:`tuple` of :class:`int` s):
            Stride of filter applications :math:`(s_1, s_2, ..., s_N)`.
            ``stride=s`` is equivalent to ``(s, s, ..., s)``.
        pad (:class:`int` or :class:`tuple` of :class:`int` s):
            Spatial padding width for input arrays
            :math:`(p_1, p_2, ..., p_N)`. ``pad=p`` is equivalent to
            ``(p, p, ..., p)``.
        outsize (:class:`tuple` of :class:`int` s):
            Expected output size of deconvolutional operation. It should be a
            tuple of ints :math:`(l_1, l_2, ..., l_N)`. Default value is
            ``None`` and the outsize is estimated by input size, stride and
            pad.
        use_cudnn (bool): If ``True``, then this function uses cuDNN if
            available. Note that cuDNN supports more than one-dimensional
            deconvolution operations only.

    Returns:
        ~chainer.Variable:
            Output variable of shape :math:`(n, c_O, l_1, l_2, ..., l_N)`.

    .. seealso:: :class:`links.DeconvolutionND`, :func:`deconvolution_2d`

    .. admonition:: Example

        **Example1**: the case when ``outsize`` is not given.

        >>> n = 10
        >>> c_i, c_o = 3, 1
        >>> d1, d2, d3 = 5, 10, 15
        >>> k1, k2, k3 = 10, 10, 10
        >>> p1, p2, p3 = 5, 5, 5
        >>> x = np.random.uniform(0, 1, (n, c_i, d1, d2, d3)).astype('f')
        >>> x.shape
        (10, 3, 5, 10, 15)
        >>> W = np.random.uniform(0, 1, (c_i, c_o, k1, k2, k3)).astype('f')
        >>> W.shape
        (3, 1, 10, 10, 10)
        >>> b = np.random.uniform(0, 1, (c_o)).astype('f')
        >>> b.shape
        (1,)
        >>> s1, s2, s3 = 2, 4, 6
        >>> y = F.deconvolution_nd(x, W, b, stride=(s1, s2, s3), \
pad=(p1, p2, p3))
        >>> y.shape
        (10, 1, 8, 36, 84)
        >>> l1 = s1 * (d1 - 1) + k1 - 2 * p1
        >>> l2 = s2 * (d2 - 1) + k2 - 2 * p2
        >>> l3 = s3 * (d3 - 1) + k3 - 2 * p3
        >>> y.shape == (n, c_o, l1, l2, l3)
        True

        **Example2**: the case when ``outsize`` is given.

        >>> n = 10
        >>> c_i, c_o = 3, 1
        >>> d1, d2, d3 = 5, 10, 15
        >>> k1, k2, k3 = 10, 10, 10
        >>> p1, p2, p3 = 5, 5, 5
        >>> x = np.random.uniform(0, 1, (n, c_i, d1, d2, d3)).astype('f')
        >>> x.shape
        (10, 3, 5, 10, 15)
        >>> W = np.random.uniform(0, 1, (c_i, c_o, k1, k2, k3)).astype('f')
        >>> W.shape
        (3, 1, 10, 10, 10)
        >>> b = np.random.uniform(0, 1, (c_o)).astype('f')
        >>> b.shape
        (1,)
        >>> s1, s2, s3 = 2, 4, 6
        >>> l1, l2, l3 = 9, 38, 87
        >>> d1 == int((l1 + 2 * p1 - k1) / s1) + 1
        True
        >>> d2 == int((l2 + 2 * p2 - k2) / s2) + 1
        True
        >>> d3 == int((l3 + 2 * p3 - k3) / s3) + 1
        True
        >>> y = F.deconvolution_nd(x, W, b, stride=(s1, s2, s3), \
pad=(p1, p2, p3), outsize=(l1, l2, l3))
        >>> y.shape
        (10, 1, 9, 38, 87)
        >>> y.shape == (n, c_o, l1, l2, l3)
        True

    """
    ndim = len(x.shape[2:])
    func = DeconvolutionND(ndim, stride, pad, outsize, use_cudnn)
    if b is None:
        return func(x, W)
    else:
        return func(x, W, b)

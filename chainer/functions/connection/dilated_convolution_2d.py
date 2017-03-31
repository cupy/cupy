import numpy
from six import moves

from chainer import cuda
from chainer import function
from chainer.utils import conv
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


def _check_cudnn_acceptable_type(x_dtype, W_dtype):
    return x_dtype == W_dtype and (
        _cudnn_version >= 3000 or x_dtype != numpy.float16)


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


class DilatedConvolution2DFunction(function.Function):

    def __init__(self, stride=1, pad=0, dilate=1,
                 use_cudnn=True, cover_all=False):
        self.sy, self.sx = _pair(stride)
        self.ph, self.pw = _pair(pad)
        self.dy, self.dx = _pair(dilate)
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
            x_type.ndim == 4,
            w_type.ndim == 4,
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

        if not type_check.same_types(*inputs):
            if b is not None:
                raise ValueError('numpy and cupy must not be used together\n'
                                 'type(W): {0}, type(x): {1}, type(b): {2}'
                                 .format(type(W), type(x), type(b)))
            else:
                raise ValueError('numpy and cupy must not be used together\n'
                                 'type(W): {0}, type(x): {1}'
                                 .format(type(W), type(x)))

        kh, kw = W.shape[2:]
        self.col = conv.im2col_cpu(
            x, kh, kw, self.sy, self.sx, self.ph, self.pw,
            cover_all=self.cover_all, dy=self.dy, dx=self.dx)
        y = numpy.tensordot(
            self.col, W, ((1, 2, 3), (1, 2, 3))).astype(x.dtype, copy=False)
        if b is not None:
            y += b
        return numpy.rollaxis(y, 3, 1),

    def forward_gpu(self, inputs):
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

        out_c, _, kh, kw = W.shape
        n, c, h, w = x.shape
        dkh, dkw = kh + (kh - 1) * (self.dy - 1), kw + (kw - 1) * (self.dx - 1)

        out_h = conv.get_conv_outsize(h, kh, self.sy, self.ph,
                                      cover_all=self.cover_all, d=self.dy)
        out_w = conv.get_conv_outsize(w, kw, self.sx, self.pw,
                                      cover_all=self.cover_all, d=self.dx)

        y = cuda.cupy.zeros((n, out_c, out_h, out_w), dtype=x.dtype)
        if (not self.cover_all and cuda.cudnn_enabled and self.use_cudnn and
                _check_cudnn_acceptable_type(x.dtype, W.dtype)):

            pad_x = cuda.cupy.zeros((n, c, h + 2 * self.ph, w + 2 * self.pw),
                                    dtype=x.dtype)
            pad_x[:, :, self.ph:self.ph + h, self.pw:self.pw + w] = x

            out_h_s1 = h + 2 * self.ph - dkh + 1
            out_w_s1 = w + 2 * self.pw - dkw + 1

            for j in moves.range(kh):
                for i in moves.range(kw):
                    xji = cuda.cupy.ascontiguousarray(
                        pad_x[:, :,
                              j * self.dy:j * self.dy + out_h_s1,
                              i * self.dx:i * self.dx + out_w_s1])
                    Wji = cuda.cupy.ascontiguousarray(
                        W[:, :, j:j + 1, i:i + 1])

                    if i == 0 and j == 0:
                        handle = cudnn.get_handle()
                        x_desc = cudnn.create_tensor_descriptor(xji)
                        y_desc = cudnn.create_tensor_descriptor(y)
                        self.filter_desc = cudnn.create_filter_descriptor(Wji)
                        self.conv_desc = cudnn.create_convolution_descriptor(
                            (0, 0), (self.sy, self.sx), xji.dtype)

                        workspace_size = cuda.get_max_workspace_size()
                        workspace = cuda.cupy.empty(
                            (workspace_size,), dtype='b')
                        algo = libcudnn.getConvolutionForwardAlgorithm(
                            handle, x_desc.value, self.filter_desc.value,
                            self.conv_desc.value, y_desc.value, _fwd_pref,
                            workspace_size)

                        oz_dtype = 'd' if x.dtype == 'd' else 'f'
                        one = numpy.array(1, dtype=oz_dtype).ctypes

                    libcudnn.convolutionForward(
                        handle, one.data, x_desc.value, xji.data.ptr,
                        self.filter_desc.value, Wji.data.ptr,
                        self.conv_desc.value, algo, workspace.data.ptr,
                        workspace_size, one.data, y_desc.value, y.data.ptr)

            if b is not None:
                b = cuda.cupy.ascontiguousarray(b)
                self.bias_desc = cudnn.create_tensor_descriptor(
                    b[None, :, None, None])
                cudnn.add_tensor(
                    handle, one.data, self.bias_desc.value, b.data.ptr,
                    one.data, y_desc.value, y.data.ptr)
        else:
            # Implementation using im2col
            self.col = conv.im2col_gpu(
                x, kh, kw, self.sy, self.sx, self.ph, self.pw,
                cover_all=self.cover_all, dy=self.dy, dx=self.dx)
            y = cuda.cupy.tensordot(
                self.col, W, ((1, 2, 3), (1, 2, 3))).astype(x.dtype,
                                                            copy=False)
            # TODO(beam2d): Support unshared bias
            if b is not None:
                y += b
            y = cuda.cupy.rollaxis(y, 3, 1)

        return y,

    def backward_cpu(self, inputs, grad_outputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        gy = grad_outputs[0]
        h, w = x.shape[2:]

        gW = numpy.tensordot(
            gy, self.col, ((0, 2, 3), (0, 4, 5))).astype(W.dtype, copy=False)
        gcol = numpy.tensordot(W, gy, (0, 1)).astype(x.dtype, copy=False)
        gcol = numpy.rollaxis(gcol, 3)
        gx = conv.col2im_cpu(gcol, self.sy, self.sx,
                             self.ph, self.pw, h, w, dy=self.dy, dx=self.dx)

        if b is None:
            return gx, gW
        else:
            gb = gy.sum(axis=(0, 2, 3))
            return gx, gW, gb

    def backward_gpu(self, inputs, grad_outputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        gy = grad_outputs[0]
        _, out_c, out_h, out_w = gy.shape
        n, c, h, w = x.shape
        kh, kw = W.shape[2:]
        dkh, dkw = kh + (kh - 1) * (self.dy - 1), kw + (kw - 1) * (self.dx - 1)

        gW = cuda.cupy.empty_like(W)
        if (not self.cover_all and cuda.cudnn_enabled and self.use_cudnn and
                _check_cudnn_acceptable_type(x.dtype, W.dtype)):

            pad_x = cuda.cupy.zeros(
                (n, c, h + 2 * self.ph, w + 2 * self.pw), dtype=x.dtype)
            pad_x[:, :, self.ph:self.ph + h, self.pw:self.pw + w] = x

            out_h_s1 = h + 2 * self.ph - dkh + 1
            out_w_s1 = w + 2 * self.pw - dkw + 1

            out_sh = out_h + (out_h - 1) * (self.sy - 1)
            out_sw = out_w + (out_w - 1) * (self.sx - 1)

            gy_ph = (h + dkh - out_sh - 1) / 2
            gy_pw = (w + dkw - out_sw - 1) / 2

            pad_gy = cuda.cupy.zeros(
                (n, out_c, h + dkh - 1, w + dkw - 1), dtype=x.dtype)
            pad_gy[:, :,
                   gy_ph:gy_ph + out_sh:self.sy,
                   gy_pw:gy_pw + out_sw:self.sx] = gy

            for j in moves.range(kh):
                for i in moves.range(kw):
                    xji = cuda.cupy.ascontiguousarray(
                        pad_x[:, :,
                              j * self.dy:j * self.dy + out_h_s1,
                              i * self.dx:i * self.dx + out_w_s1])
                    gyji = cuda.cupy.ascontiguousarray(
                        pad_gy[:, :,
                               j * self.dy:j * self.dy + h,
                               i * self.dx:i * self.dx + w])
                    Wji = cuda.cupy.ascontiguousarray(
                        W[:, :, -1::-1, -1::-1][:, :, j:j + 1, i:i + 1])

                    if i == 0 and j == 0:
                        x = cuda.cupy.ascontiguousarray(x)
                        gy = cuda.cupy.ascontiguousarray(gy)

                        handle = cudnn.get_handle()
                        x_desc = cudnn.create_tensor_descriptor(x)
                        xji_desc = cudnn.create_tensor_descriptor(xji)
                        gy_desc = cudnn.create_tensor_descriptor(gy)
                        gyji_desc = cudnn.create_tensor_descriptor(gyji)
                        conv_desc_data = cudnn.create_convolution_descriptor(
                            (0, 0), (1, 1), xji.dtype)

                        oz_dtype = 'd' if x.dtype == 'd' else 'f'
                        one = numpy.array(1, dtype=oz_dtype).ctypes
                        zero = numpy.array(0, dtype=oz_dtype).ctypes
                        gx = cuda.cupy.zeros_like(x)
                        gWji = cuda.cupy.empty((out_c, c, 1, 1), dtype=W.dtype)

                        if _cudnn_version >= 4000:
                            workspace_size = cuda.get_max_workspace_size()
                            workspace = cuda.cupy.empty(
                                (workspace_size,), dtype='b')

                            algo_filter = (
                                libcudnn.getConvolutionBackwardFilterAlgorithm(
                                    handle, xji_desc.value, gy_desc.value,
                                    self.conv_desc.value,
                                    self.filter_desc.value,
                                    _bwd_filter_pref, workspace_size))
                            algo_data = (
                                libcudnn.getConvolutionBackwardDataAlgorithm(
                                    handle, self.filter_desc.value,
                                    gyji_desc.value, conv_desc_data.value,
                                    x_desc.value, _bwd_data_pref,
                                    workspace_size))

                    if _cudnn_version >= 4000:
                        libcudnn.convolutionBackwardFilter_v3(
                            handle, one.data, xji_desc.value, xji.data.ptr,
                            gy_desc.value, gy.data.ptr, self.conv_desc.value,
                            algo_filter, workspace.data.ptr, workspace_size,
                            zero.data, self.filter_desc.value, gWji.data.ptr)
                        libcudnn.convolutionBackwardData_v3(
                            handle, one.data, self.filter_desc.value,
                            Wji.data.ptr, gyji_desc.value,
                            gyji.data.ptr, conv_desc_data.value,
                            algo_data, workspace.data.ptr, workspace_size,
                            one.data, x_desc.value, gx.data.ptr)
                    else:
                        libcudnn.convolutionBackwardFilter_v2(
                            handle, one.data, xji_desc.value, xji.data.ptr,
                            gy_desc.value, gy.data.ptr, self.conv_desc.value,
                            zero.data, self.filter_desc.value, gWji.data.ptr)
                        libcudnn.convolutionBackwardData_v2(
                            handle, one.data, self.filter_desc.value,
                            Wji.data.ptr, gyji_desc.value,
                            gyji.data.ptr, conv_desc_data.value,
                            one.data, x_desc.value, gx.data.ptr)
                    gW[:, :, j:j + 1, i:i + 1] = gWji

            if b is not None:
                gb = cuda.cupy.empty_like(b)
                libcudnn.convolutionBackwardBias(
                    handle, one.data, gy_desc.value, gy.data.ptr,
                    zero.data, self.bias_desc.value, gb.data.ptr)
        else:
            gW = cuda.cupy.tensordot(
                gy, self.col, ((0, 2, 3), (0, 4, 5))).astype(W.dtype,
                                                             copy=False)
            gcol = cuda.cupy.tensordot(W, gy, (0, 1)).astype(x.dtype,
                                                             copy=False)
            gcol = cuda.cupy.rollaxis(gcol, 3)
            gx = conv.col2im_gpu(gcol, self.sy, self.sx, self.ph, self.pw,
                                 h, w, dy=self.dy, dx=self.dx)

            if b is not None:
                gb = gy.sum(axis=(0, 2, 3))

        if b is None:
            return gx, gW
        else:
            return gx, gW, gb


def dilated_convolution_2d(x, W, b=None, stride=1, pad=0, dilate=1,
                           use_cudnn=True, cover_all=False):
    """Two-dimensional dilated convolution function.

    This is an implementation of two-dimensional dilated convolution
    in ConvNets.
    It takes three variables: the input image ``x``, the filter weight ``W``,
    and the bias vector ``b``.

    Notation: here is a notation for dimensionalities.

    - :math:`n` is the batch size.
    - :math:`c_I` and :math:`c_O` are the number of the input and output,
      respectively.
    - :math:`h` and :math:`w` are the height and width of the input image,
      respectively.
    - :math:`k_H` and :math:`k_W` are the height and width of the filters,
      respectively.

    Args:
        x (~chainer.Variable): Input variable of shape :math:`(n, c_I, h, w)`.
        W (~chainer.Variable): Weight variable of shape
            :math:`(c_O, c_I, k_H, k_W)`.
        b (~chainer.Variable): Bias variable of length :math:`c_O` (optional).
        stride (int or pair of ints): Stride of filter applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent.
        pad (int or pair of ints): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        dilate (int or pair of ints): Dilation factor of filter applications.
            ``dilate=d`` and ``dilate=(d, d)`` are equivalent.
        use_cudnn (bool): If ``True``, then this function uses cuDNN if
            available.
        cover_all (bool): If ``True``, all spatial locations are convoluted
            into some output pixels. It may make the output size larger.

    Returns:
        ~chainer.Variable: Output variable.

    The two-dimensional dilated convolution function is defined as follows.
    Then the ``DilatedConvolution2D`` function computes correlations
    between filters and patches of size :math:`(k_H, k_W)` in ``x``.
    Patches here are extracted at intervals of the dilation factor.
    Note that correlation here is equivalent to the inner product between
    expanded vectors.
    Patches are extracted at intervals of the dilation factor and at positions
    shifted by multiples of ``stride`` from the first position ``-pad`` for
    each spatial axis. The right-most (or bottom-most) patches do not run over
    the padded spatial size.

    Let :math:`(s_Y, s_X)` be the stride of filter application,
    :math:`(p_H, p_W)` the spatial padding size, and :math:`(d_Y, d_X)`
    the dilation factor of filter application. Then, the output size
    :math:`(h_O, w_O)` is determined by the following equations:

    .. math::

       h_O &= (h + 2p_H - k_H - (k_H - 1) * (d_Y - 1)) / s_Y + 1,\\\\
       w_O &= (w + 2p_W - k_W - (k_W - 1) * (d_X - 1)) / s_X + 1.

    If the bias vector is given, then it is added to all spatial locations of
    the output of convolution.

    .. seealso:: :class:`DilatedConvolution2D`

    """
    func = DilatedConvolution2DFunction(
        stride, pad, dilate, use_cudnn, cover_all)
    if b is None:
        return func(x, W)
    else:
        return func(x, W, b)

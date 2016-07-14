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

    def __init__(self, stride=1, pad=0, use_cudnn=True, cover_all=False, dilate=1):
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
        kh, kw = W.shape[2:]
        self.col = conv.im2col_cpu(
            x, kh, kw, self.sy, self.sx, self.ph, self.pw,
            cover_all=self.cover_all, dy=self.dy, dx=self.dx)
        y = numpy.tensordot(
            self.col, W, ((1, 2, 3), (1, 2, 3))).astype(x.dtype)
        if b is not None:
            y += b
        return numpy.rollaxis(y, 3, 1),

    def forward_gpu(self, inputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None

        out_c, _, kh, kw = W.shape
        n, c, h, w = x.shape
        dkh, dkw = kh + (kh - 1) * (self.dy - 1), kw + (kw - 1) * (self.dx - 1)

        out_h = conv.get_conv_outsize(h, dkh, self.sy, self.ph,
                                      cover_all=self.cover_all)
        out_w = conv.get_conv_outsize(w, dkw, self.sx, self.pw,
                                      cover_all=self.cover_all)

        y = cuda.cupy.zeros((n, out_c, out_h, out_w), dtype=x.dtype)
        if (not self.cover_all and cuda.cudnn_enabled and self.use_cudnn and
                _check_cudnn_acceptable_type(x.dtype, W.dtype)):

            pad_x = cuda.cupy.zeros((n, c, h + 2 * self.ph, w + 2 * self.pw),
                                    dtype=x.dtype)
            pad_x[:, :, self.ph:self.ph + h, self.pw:self.pw + w] = x

            for j in moves.range(kh):
                for i in moves.range(kw):
                    xji = cuda.cupy.ascontiguousarray(
                        pad_x[:, :,
                        j * self.dy:j * self.dy + h + 2 * self.ph - dkh + 1,
                        i * self.dx:i * self.dx + w + 2 * self.pw - dkw + 1])
                    Wji = cuda.cupy.ascontiguousarray(W[:, :, j:j + 1, i:i + 1])

                    if i == 0 and j == 0:
                        handle = cudnn.get_handle()
                        x_desc = cudnn.create_tensor_descriptor(xji)
                        y_desc = cudnn.create_tensor_descriptor(y)
                        self.filter_desc = cudnn.create_filter_descriptor(Wji)
                        self.conv_desc = cudnn.create_convolution_descriptor(
                            (0, 0), (self.sy, self.sx))

                        workspace_size = cuda.get_max_workspace_size()
                        workspace = cuda.cupy.empty((workspace_size,), dtype='b')
                        algo = libcudnn.getConvolutionForwardAlgorithm(
                            handle, x_desc.value, self.filter_desc.value,
                            self.conv_desc.value, y_desc.value, _fwd_pref,
                            workspace_size)

                        oz_dtype = 'd' if x.dtype == 'd' else 'f'
                        one = numpy.array(1, dtype=oz_dtype).ctypes

                    libcudnn.convolutionForward(
                        handle, one.data, x_desc.value, xji.data.ptr,
                        self.filter_desc.value, Wji.data.ptr, self.conv_desc.value,
                        algo, workspace.data.ptr, workspace_size, one.data,
                        y_desc.value, y.data.ptr)

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
            W_mat = W.reshape(out_c, -1)
            col_mats = self.col.reshape(n, -1, out_h * out_w)
            y_mats = y.reshape(n, out_c, -1)
            # TODO(beam2d): Use streams or batch gemm
            for i in moves.range(n):
                y_mats[i] = W_mat.dot(col_mats[i])
            # TODO(beam2d): Support unshared bias
            if b is not None:
                y += b[:, None, None]

        return y,

    def backward_cpu(self, inputs, grad_outputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        gy = grad_outputs[0]
        h, w = x.shape[2:]

        gW = numpy.tensordot(
            gy, self.col, ((0, 2, 3), (0, 4, 5))).astype(W.dtype)
        gcol = numpy.tensordot(W, gy, (0, 1)).astype(x.dtype)
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

        gW = cuda.cupy.empty_like(W)
        gW_mat = gW.reshape(out_c, c * kh * kw)
        col_mats = self.col.reshape(n, c * kh * kw, out_h * out_w)
        gy_mats = gy.reshape(n, out_c, out_h * out_w)
        # TODO(beam2d): Use streams or batch gemm
        gW_mat[...] = 0
        for i in moves.range(n):
            gW_mat += cuda.cupy.dot(gy_mats[i], col_mats[i].T)

        W_mat = W.reshape(out_c, -1)
        gcol = cuda.cupy.empty_like(self.col)
        gcol_mats = gcol.reshape(n, c * kh * kw, out_h * out_w)

        for i in moves.range(n):
            gcol_mats[i] = cuda.cupy.dot(W_mat.T, gy_mats[i])

        # dilate col2im_gpu
        # TODO(yasunorikudo): Write cuda.elementwise
        img = cuda.cupy.zeros(
            (n, c, h + 2 * self.ph + self.sy - 1, w + 2 * self.pw + self.sx - 1),
            dtype=gcol.dtype)
        for j in moves.range(kh):
            q = j * self.dy
            q_lim = q + self.sy * out_h
            for i in moves.range(kw):
                p = i * self.dx
                p_lim = p + self.sx * out_w
                img[:, :, q:q_lim:self.sy, p:p_lim:self.sx] += gcol[:, :, j, i, :, :]
        gx = img[:, :, self.ph:h + self.ph, self.pw:w + self.pw]

        if b is not None:
            gb = gy.sum(axis=(0, 2, 3))

        if b is None:
            return gx, gW
        else:
            return gx, gW, gb

def dilated_convolution_2d(x, W, b=None, stride=1, pad=0, use_cudnn=True,
                   cover_all=False, dilate=1):
    """Two-dimensional dilated convolution function.
    """
    func = DilatedConvolution2DFunction(stride, pad, use_cudnn, cover_all, dilate)
    if b is None:
        return func(x, W)
    else:
        return func(x, W, b)

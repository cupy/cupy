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

    def __init__(self, dilate=1, stride=1, pad=0, use_cudnn=True, cover_all=False):
        self.dy, self.dx = _pair(dilate)
        self.sy, self.sx = _pair(stride)
        self.ph, self.pw = _pair(pad)
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
        dkh, dkw = kh + (kh - 1) * (self.dy - 1), kw + (kw - 1) * (self.dx - 1)

        self.col = conv.im2col_cpu(
            x, dkh, dkw, self.sy, self.sx, self.ph, self.pw,
            cover_all=self.cover_all)[:, :, 0:dkh:self.dy, 0:dkw:self.dx, :, :]
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

        y = cuda.cupy.empty((n, out_c, out_h, out_w), dtype=x.dtype)

        # Implementation using im2col
        self.col = conv.im2col_gpu(
            x, dkh, dkw, self.sy, self.sx, self.ph, self.pw,
            cover_all=self.cover_all)[:, :, 0:dkh:self.dy, 0:dkw:self.dx, :, :]
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
        kh, kw = W.shape[2:]

        # TODO(yasunorikudo): Dilate W efficiently
        if not self.dy == 1:
            for i in range(self.dy - 1):
                seq_h = numpy.arange(1, kh) if i == 0 else numpy.dstack((seq_h, numpy.arange(1, kh)))
            seq_h = seq_h.reshape((self.dy - 1) * (kh - 1))
            W = numpy.insert(W, seq_h, numpy.zeros(kw), axis=2)
        if not self.dx == 1:
            for i in range(self.dx - 1):
                seq_w = numpy.arange(1, kw) if i == 0 else numpy.dstack((seq_w, numpy.arange(1, kw)))
            seq_w = seq_w.reshape((self.dx - 1) * (kw - 1))
            W = numpy.insert(W, seq_w, 0, axis=3)

        gW = numpy.tensordot(
            gy, self.col, ((0, 2, 3), (0, 4, 5))).astype(W.dtype)
        gcol = numpy.tensordot(W, gy, (0, 1)).astype(x.dtype)
        gcol = numpy.rollaxis(gcol, 3)
        gx = conv.col2im_cpu(gcol, self.sy, self.sx, self.ph, self.pw, h, w)

        if b is None:
            return gx, gW
        else:
            gb = gy.sum(axis=(0, 2, 3))
            return gx, gW, gb

    def backward_gpu(self, inputs, grad_outputs):

def dilated_convolution_2d(x, W, b=None, dilate=1, stride=1, pad=0, use_cudnn=True,
                   cover_all=False):
    """Two-dimensional dilated convolution function.
    """
    func = DilatedConvolution2DFunction(dilate, stride, pad, use_cudnn, cover_all)
    if b is None:
        return func(x, W)
    else:
        return func(x, W, b)

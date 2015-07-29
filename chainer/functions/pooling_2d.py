import collections
import ctypes

import numpy

from chainer import cuda
from chainer.cuda import cudnn
from chainer import function
from chainer.utils import conv
from chainer.utils import type_check

if cudnn.available:
    libcudnn = cudnn.cudnn


def _pair(x):
    if isinstance(x, collections.Iterable):
        return x
    return (x, x)


class Pooling2D(function.Function):

    """Base class of pooling function over a set of 2d planes."""

    def __init__(self, ksize, stride=None, pad=0, cover_all=True,
                 use_cudnn=True):
        if stride is None:
            stride = ksize

        self.kh, self.kw = _pair(ksize)
        self.sy, self.sx = _pair(stride)
        self.ph, self.pw = _pair(pad)

        self.cover_all = cover_all
        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype == numpy.float32,
            in_types[0].ndim == 4
        )

    def forward_gpu(self, x):
        # Implementation using cudnn
        n, c, h, w = x[0].shape
        y_h = conv.get_conv_outsize(
            h, self.kh, self.sy, self.ph, self.cover_all)
        y_w = conv.get_conv_outsize(
            w, self.kw, self.sx, self.pw, self.cover_all)
        y = cuda.empty((n, c, y_h, y_w), dtype=numpy.float32)

        handle = cudnn.get_handle()
        pool_desc = self.create_pool_desc()
        x_desc = cudnn.create_tensor_descriptor(x[0])
        y_desc = cudnn.create_tensor_descriptor(y)

        libcudnn.poolingForward(
            handle, pool_desc.value, ctypes.c_float(1), x_desc.value,
            x[0].data.ptr, ctypes.c_float(0), y_desc.value, y.data.ptr)
        self.y = y

        return y,

    def backward_gpu(self, x, gy):
        # Implementation using cudnn
        handle = cudnn.get_handle()
        pool_desc = self.create_pool_desc()

        x_desc = cudnn.create_tensor_descriptor(x[0])
        y_desc = cudnn.create_tensor_descriptor(gy[0])

        gx = cuda.empty_like(x[0])
        libcudnn.poolingBackward(
            handle, pool_desc.value, ctypes.c_float(1), y_desc.value,
            self.y.data.ptr, y_desc.value, gy[0].data.ptr, x_desc.value,
            x[0].data.ptr, ctypes.c_float(0), x_desc.value, gx.data.ptr)
        return gx,

    def create_pool_desc(self):
        raise NotImplementedError()


class MaxPooling2D(Pooling2D):

    """Max pooling over a set of 2d planes."""

    def forward_cpu(self, x):
        col = conv.im2col_cpu(
            x[0], self.kh, self.kw, self.sy, self.sx, self.ph, self.pw,
            pval=-float('inf'), cover_all=self.cover_all)
        n, c, kh, kw, out_h, out_w = col.shape
        col = col.reshape(n, c, kh * kw, out_h, out_w)

        # We select maximum twice, since the implementation using numpy.choose
        # hits its bug when kh * kw >= 32.
        self.indexes = col.argmax(axis=2)
        y = col.max(axis=2)
        return y,

    def forward_gpu(self, x):
        if cuda.cudnn_enabled and self.use_cudnn:
            return super(MaxPooling2D, self).forward_gpu(x)

        n, c, h, w = x[0].shape
        y_h = conv.get_conv_outsize(
            h, self.kh, self.sy, self.ph, self.cover_all)
        y_w = conv.get_conv_outsize(
            w, self.kw, self.sx, self.pw, self.cover_all)
        y = cuda.empty((n, c, y_h, y_w), dtype=numpy.float32)
        self.indexes = cuda.empty((n, c, y_h, y_w), dtype=numpy.int32)

        args = tuple(map(numpy.int32, (h, w, y_h, y_w, self.kh, self.kw,
                                       self.sy, self.sx, self.ph, self.pw)))
        cuda.elementwise(
            ['out', 'indexes', 'in', 'h', 'w', 'out_h', 'out_w',
             'kh', 'kw', 'sy', 'sx', 'ph', 'pw'],
            '''
               int c0    = i / (out_h * out_w);
               int out_y = i / out_w % out_h;
               int out_x = i % out_w;
               int in_y_0 = max(0, out_y * sy - ph);
               int in_y_1 = min(h, out_y * sy + kh - ph);
               int in_x_0 = max(0, out_x * sx - pw);
               int in_x_1 = min(w, out_x * sx + kw - pw);

               float maxval = in[in_x_0 + w * (in_y_0 + h * c0)];
               int argmax_y = in_y_0;
               int argmax_x = in_x_0;
               for (int y = in_y_0; y < in_y_1; ++y) {
                 int offset_y = w * (y + h * c0);
                 for (int x = in_x_0; x < in_x_1; ++x) {
                   float v = in[x + offset_y];
                   if (maxval < v) {
                     maxval   = v;
                     argmax_y = y;
                     argmax_x = x;
                   }
                 }
               }
               out[i] = maxval;

               int argmax_ky = argmax_y + ph - out_y * sy;
               int argmax_kx = argmax_x + pw - out_x * sx;
               indexes[i] = argmax_kx + kw * argmax_ky;
            ''', 'max_pool_fwd')(y, self.indexes, x[0], *args)
        return y,

    def backward_cpu(self, x, gy):
        n, c, out_h, out_w = gy[0].shape
        h, w = x[0].shape[2:]
        gcol = numpy.zeros(
            (n, c, self.kh, self.kw, out_h, out_w), dtype=numpy.float32)

        # TODO(beam2d): Make it fast
        gcol_r = numpy.rollaxis(gcol.reshape(n, c, -1, out_h, out_w), 2)
        for i in numpy.ndindex(n, c, out_h, out_w):
            gcol_r[self.indexes[i]][i] = gy[0][i]

        gx = conv.col2im_cpu(gcol, self.sy, self.sx, self.ph, self.pw, h, w)
        return gx,

    def backward_gpu(self, x, gy):
        if cuda.cudnn_enabled and self.use_cudnn:
            return super(MaxPooling2D, self).backward_gpu(x, gy)

        n, c, h, w = x[0].shape
        y_h, y_w = gy[0].shape[2:]
        gx = cuda.empty_like(x[0])

        args = tuple(map(numpy.int32, (h, w, y_h, y_w, self.kh, self.kw,
                                       self.sy, self.sx, self.ph, self.pw)))
        cuda.elementwise(
            ['gx', 'indexes', 'gy', 'h', 'w', 'out_h', 'out_w',
             'kh', 'kw', 'sy', 'sx', 'ph', 'pw'],
            '''
               int c0 = i / (h * w);
               int y  = i / w % h + ph;
               int x  = i % w + pw;
               int out_y_0 = max(0,     (y - kh + sy) / sy);
               int out_y_1 = min(out_h, (y      + sy) / sy);
               int out_x_0 = max(0,     (x - kw + sx) / sx);
               int out_x_1 = min(out_w, (x      + sx) / sx);

               float val = 0;
               for (int out_y = out_y_0; out_y < out_y_1; ++out_y) {
                 int ky = y - out_y * sy;
                 for (int out_x = out_x_0; out_x < out_x_1; ++out_x) {
                   int kx = x - out_x * sx;
                   int offset = out_x + out_w * (out_y + out_h * c0);
                   if (indexes[offset] == kx + kw * ky) {
                     val += gy[offset];
                   }
                 }
               }
               gx[i] = val;
            ''',
            'max_pool_bwd')(gx, self.indexes, gy[0], *args)
        return gx,

    def create_pool_desc(self):
        return cudnn.create_pooling_descriptor(
            (self.kh, self.kw), (self.sy, self.sx), (self.ph, self.pw),
            libcudnn.CUDNN_POOLING_MAX)


def max_pooling_2d(x, ksize, stride=None, pad=0, cover_all=True,
                   use_cudnn=True):
    """Spatial max pooling function.

    This function acts similarly to :class:`~functions.Convolution2D`, but
    it computes the maximum of input spatial patch for each channel
    without any parameter instead of computing the inner products.

    Args:
        x (~chainer.Variable): Input variable.
        ksize (int or (int, int)): Size of pooling window. ``ksize=k`` and
            ``ksize=(k, k)`` are equivalent.
        stride (int or (int, int) or None): Stride of pooling applications.
            ``ksize=k`` and ``ksize=(k, k)`` are equivalent. If None is
            specified, then it uses same stride as the pooling window size.
        pad (int or (int, int)): Spatial padding width for the input array.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        cover_all (bool): If True, all spatial locations are pooled into some
            output pixels. It may make the output size larger.
        use_cudnn (bool): If True and CuDNN is enabled, then this function
            uses CuDNN as the core implementation.

    Returns:
        ~chainer.Variable: Ouptut variable.

    """
    return MaxPooling2D(ksize, stride, pad, cover_all, use_cudnn)(x)


class AveragePooling2D(Pooling2D):

    """Average pooling over a set of 2d planes."""
    # TODO(beam2d): Support cover_all mode.

    def forward_cpu(self, x):
        col = conv.im2col_cpu(x[0], self.kh, self.kw, self.sy, self.sx,
                              self.ph, self.pw)
        y = col.mean(axis=(2, 3))
        return y,

    def forward_gpu(self, x):
        if cuda.cudnn_enabled and self.use_cudnn:
            return super(AveragePooling2D, self).forward_gpu(x)

        n, c, h, w = x[0].shape
        y_h = conv.get_conv_outsize(h, self.kh, self.sy, self.ph)
        y_w = conv.get_conv_outsize(w, self.kw, self.sx, self.pw)
        y = cuda.empty((n, c, y_h, y_w), dtype=numpy.float32)
        coeff = 1. / (self.kh * self.kw)
        args = list(map(numpy.int32, (h, w, y_h, y_w, self.kh, self.kw,
                                      self.sy, self.sx, self.ph, self.pw)))
        args = [y, x[0]] + args + [x[0].dtype.type(coeff)]
        cuda.elementwise(
            ['out', 'in', 'h', 'w', 'out_h', 'out_w',
             'kh', 'kw', 'sy', 'sx', 'ph', 'pw', 'coeff'],
            '''
               int c0    = i / (out_h * out_w);
               int out_y = i / out_w % out_h;
               int out_x = i % out_w;
               int in_y_0 = max(0, out_y * sy - ph);
               int in_y_1 = min(h, out_y * sy + kh - ph);
               int in_x_0 = max(0, out_x * sx - pw);
               int in_x_1 = min(w, out_x * sx + kw - pw);

               float val = 0;
               for (int y = in_y_0; y < in_y_1; ++y) {
                 int offset_y = w * (y + h * c0);
                 for (int x = in_x_0; x < in_x_1; ++x) {
                   val += in[x + offset_y];
                 }
               }
               out[i] = val * coeff;
            ''', 'avg_pool_fwd')(*args)
        return y,

    def backward_cpu(self, x, gy):
        h, w = x[0].shape[2:]
        gcol = numpy.tile(gy[0][:, :, numpy.newaxis, numpy.newaxis],
                          (1, 1, self.kh, self.kw, 1, 1))
        gx = conv.col2im_cpu(gcol, self.sy, self.sx, self.ph, self.pw, h, w)
        gx /= self.kh * self.kw
        return gx,

    def backward_gpu(self, x, gy):
        if cuda.cudnn_enabled and self.use_cudnn:
            return super(AveragePooling2D, self).backward_gpu(x, gy)

        n, c, h, w = x[0].shape
        y_h, y_w = gy[0].shape[2:]
        gx = cuda.empty_like(x[0])
        coeff = 1. / (self.kh * self.kw)
        args = list(map(numpy.int32, (h, w, y_h, y_w, self.kh, self.kw,
                                      self.sy, self.sx, self.ph, self.pw)))
        args = [gx, gy[0]] + args + [x[0].dtype.type(coeff)]
        cuda.elementwise(
            ['gx', 'gy', 'h', 'w', 'out_h', 'out_w',
             'kh', 'kw', 'sy', 'sx', 'ph', 'pw', 'coeff'],
            '''
               int c0 = i / (h * w);
               int y  = i / w % h + ph;
               int x  = i % w + pw;
               int out_y_0 = max(0,     (y - kh + sy) / sy);
               int out_y_1 = min(out_h, (y      + sy) / sy);
               int out_x_0 = max(0,     (x - kw + sx) / sx);
               int out_x_1 = min(out_w, (x      + sx) / sx);
               int hc0  = out_h * c0;

               float val = 0;
               for (int out_y = out_y_0; out_y < out_y_1; ++out_y) {
                 for (int out_x = out_x_0; out_x < out_x_1; ++out_x) {
                   val += gy[out_x + out_w * (out_y + hc0)];
                 }
               }
               gx[i] = val * coeff;
            ''', 'avg_pool_bwd')(*args)
        return gx,

    def create_pool_desc(self):
        return cudnn.create_pooling_descriptor(
            (self.kh, self.kw), (self.sy, self.sx), (self.ph, self.pw),
            libcudnn.CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING)


def average_pooling_2d(x, ksize, stride=None, pad=0, use_cudnn=True):
    """Spatial average pooling function.

    This function acts similarly to :class:`~functions.Convolution2D`, but
    it computes the average of input spatial patch for each channel
    without any parameter instead of computing the inner products.

    Args:
        x (~chainer.Variable): Input variable.
        ksize (int or (int, int)): Size of pooling window. ``ksize=k`` and
            ``ksize=(k, k)`` are equivalent.
        stride (int or (int, int) or None): Stride of pooling applications.
            ``ksize=k`` and ``ksize=(k, k)`` are equivalent. If None is
            specified, then it uses same stride as the pooling window size.
        pad (int or (int, int)): Spatial padding width for the input array.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        use_cudnn (bool): If True and CuDNN is enabled, then this function
            uses CuDNN as the core implementation.

    Returns:
        ~chainer.Variable: Output variable.

    .. note::

       This function currently does not support ``cover_all`` mode as
       :func:`max_pooling_2d`. Average pooling runs in non-cover-all mode.

    """
    return AveragePooling2D(ksize, stride, pad, False, use_cudnn)(x)

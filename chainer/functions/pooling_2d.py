import collections
import ctypes

import numpy
import six

from chainer import cuda
from chainer import function
from chainer.functions import concat
from chainer.utils import conv
from chainer.utils import type_check

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
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

        # Pooling of cuDNNv2 does not seem to support non-contiguous gradients
        gy = cuda.cupy.ascontiguousarray(gy[0])

        x_desc = cudnn.create_tensor_descriptor(x[0])
        y_desc = cudnn.create_tensor_descriptor(gy)

        gx = cuda.empty_like(x[0])
        libcudnn.poolingBackward(
            handle, pool_desc.value, ctypes.c_float(1), y_desc.value,
            self.y.data.ptr, y_desc.value, gy.data.ptr, x_desc.value,
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
        y = cuda.empty((n, c, y_h, y_w), dtype=x[0].dtype)
        self.indexes = cuda.empty((n, c, y_h, y_w), dtype=numpy.int32)

        cuda.elementwise(
            'raw T in, int32 h, int32 w, int32 out_h, int32 out_w,'
            'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw',
            'T out, S indexes',
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
               out = maxval;

               int argmax_ky = argmax_y + ph - out_y * sy;
               int argmax_kx = argmax_x + pw - out_x * sx;
               indexes = argmax_kx + kw * argmax_ky;
            ''', 'max_pool_fwd')(x[0],
                                 h, w, y_h, y_w, self.kh, self.kw,
                                 self.sy, self.sx, self.ph, self.pw,
                                 y, self.indexes)
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

        cuda.elementwise(
            'raw T gy, raw S indexes, int32 h, int32 w,'
            'int32 out_h, int32 out_w, int32 kh, int32 kw,'
            'int32 sy, int32 sx, int32 ph, int32 pw',
            'T gx',
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
               gx = val;
            ''',
            'max_pool_bwd')(gy[0], self.indexes,
                            h, w, y_h, y_w, self.kh, self.kw,
                            self.sy, self.sx, self.ph, self.pw,
                            gx)
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
            ``stride=s`` and ``stride=(s, s)`` are equivalent. If None is
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
        cuda.elementwise(
            'raw T in, int32 h, int32 w,'
            'int32 out_h, int32 out_w, int32 kh, int32 kw,'
            'int32 sy, int32 sx, int32 ph, int32 pw, T coeff',
            'T out',
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
               out = val * coeff;
            ''', 'avg_pool_fwd')(x[0], h, w, y_h, y_w, self.kh, self.kw,
                                 self.sy, self.sx, self.ph, self.pw, coeff,
                                 y)
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
        cuda.elementwise(
            'raw T gy, int32 h, int32 w,'
            'int32 out_h, int32 out_w, int32 kh, int32 kw,'
            'int32 sy, int32 sx, int32 ph, int32 pw, T coeff',
            'T gx',
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
               gx = val * coeff;
            ''', 'avg_pool_bwd')(gy[0], h, w, y_h, y_w, self.kh, self.kw,
                                 self.sy, self.sx, self.ph, self.pw, coeff,
                                 gx)
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
            ``stride=s`` and ``stride=(s, s)`` are equivalent. If None is
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


class SpatialPyramidPooling2D(function.Function):

    """Spatial pyramid pooling over a set of 2d planes."""

    def __init__(self, x_shape, pyramid_height, pooling_class, use_cudnn=True):
        bottom_c, bottom_h, bottom_w = x_shape
        self.pyramid_height = pyramid_height

        # create pooling functions for different pyramid levels
        out_dim = 0
        self.split_inds = []
        self.poolers = []
        for pyramid_level in six.moves.range(pyramid_height):
            num_bins = int(2 ** pyramid_level)

            ksize_h = int(numpy.ceil(bottom_h / (float(num_bins))))
            remainder_h = ksize_h * num_bins - bottom_h
            pad_h = remainder_h // 2

            ksize_w = int(numpy.ceil(bottom_w / (float(num_bins))))
            remainder_w = ksize_w * num_bins - bottom_w
            pad_w = remainder_w // 2

            ksize = (ksize_h, ksize_w)
            pad = (pad_h, pad_w)

            if pooling_class == MaxPooling2D:
                pooler = pooling_class(ksize=ksize, stride=None, pad=pad,
                                       cover_all=True, use_cudnn=use_cudnn)
                self.poolers.append(pooler)
            else:
                raise NotImplementedError()

            out_dim += bottom_c * (num_bins ** 2)
            if pyramid_level < pyramid_height - 1:
                self.split_inds.append(out_dim)

    def forward(self, x):
        self.ys = []
        for pooler in self.poolers:
            y = pooler.forward(x)[0]
            n, c, h, w = pooler.out_shape = y.shape
            self.ys.append(y.reshape((n, c * h * w, 1, 1)))

        return concat.Concat(axis=1).forward(self.ys)

    def backward(self, x, gy):
        xp = cuda.get_array_module(*x)
        gx = xp.zeros_like(x[0])
        gys = xp.split(gy[0], self.split_inds, axis=1)
        for pooler, gy in zip(self.poolers, gys):
            gy = gy.reshape(pooler.out_shape)
            gx += pooler.backward(x, (gy,))[0]

        return gx,


def spatial_pyramid_pooling_2d(x, pyramid_height, pooling_class,
                               use_cudnn=True):
    """Spatial pyramid pooling function.

    It outputs a fixed-length vector regardless of input feature map size.

    It performs pooling operation to the input 4D-array ``x`` with different
    kernel sizes and padding sizes, and then flattens all dimensions except
    first dimension of all pooling results, and finally concatenates them along
    2nd dimension.

    At :math:`i`-th pyramid level, the kernel size
    :math:`(k_h^{(i)}, k_w^{(i)})` and padding size
    :math:`(p_h^{(i)}, p_w^{(i)})` of pooling operation are calculated as
    below:

    .. math::
        k_h^{(i)} &= \\lceil b_h / 2^i \\rceil, \\\\
        k_w^{(i)} &= \\lceil b_w / 2^i \\rceil, \\\\
        p_h^{(i)} &= (2^i k_h^{(i)} - b_h) / 2, \\\\
        p_w^{(i)} &= (2^i k_w^{(i)} - b_w) / 2,

    where :math:`\\lceil \\cdot \\rceil` denotes the ceiling function, and
    :math:`b_h, b_w` are height and width of input variable ``x``,
    respectively. Note that index of pyramid level :math:`i` is zero-based.

    See detail in paper: `Spatial Pyramid Pooling in Deep Convolutional \
    Networks for Visual Recognition \
    <http://arxiv.org/abs/1406.4729>`_.

    Args:
        x (~chainer.Variable): Input variable. The shape of ``x`` should be
            (batchsize, # of channels, height, width).
        pyramid_height (int): the number of pyramid levels
        pooling_class (MaxPooling2D or AveragePooling2D):
            Only MaxPooling2D class can be available for now.
        use_cudnn (bool): If True and CuDNN is enabled, then this function
            uses CuDNN as the core implementation.

    Returns:
        ~chainer.Variable: Ouptut variable. The shape of the output variable
            will be (batchsize, :math:`c \\sum_{h=0}^{H-1} 2^{2h}`, 1, 1),
            where :math:`c` is the number of channels of input variable ``x``
            and :math:`H` is the number of pyramid levels.

    .. note::

        This function uses some pooling classes as components to perform
        spatial pyramid pooling. Now it supports only
        :class:`~functions.MaxPooling2D` as elemental pooling operator so far.

    """

    return SpatialPyramidPooling2D(x.data.shape[1:], pyramid_height,
                                   pooling_class, use_cudnn=use_cudnn)(x)

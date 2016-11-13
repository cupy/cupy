import numpy

from chainer import cuda
from chainer.functions.pooling import pooling_2d
from chainer.utils import conv

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn


class AveragePooling2D(pooling_2d.Pooling2D):

    """Average pooling over a set of 2d planes."""
    # TODO(beam2d): Support cover_all mode.

    def forward_cpu(self, x):
        col = conv.im2col_cpu(x[0], self.kh, self.kw, self.sy, self.sx,
                              self.ph, self.pw)
        y = col.mean(axis=(2, 3))
        return y,

    def forward_gpu(self, x):
        if (cuda.cudnn_enabled and self.use_cudnn and
                pooling_2d._check_cudnn_acceptable_type(x[0].dtype)):
            return super(AveragePooling2D, self).forward_gpu(x)

        n, c, h, w = x[0].shape
        y_h = conv.get_conv_outsize(h, self.kh, self.sy, self.ph)
        y_w = conv.get_conv_outsize(w, self.kw, self.sx, self.pw)
        y = cuda.cupy.empty((n, c, y_h, y_w), dtype=x[0].dtype)
        coeff = 1. / (self.kh * self.kw)
        kern = cuda.elementwise(
            'raw T in, int32 h, int32 w,'
            'int32 out_h, int32 out_w, int32 kh, int32 kw,'
            'int32 sy, int32 sx, int32 ph, int32 pw, T coeff',
            'T out', '''
            int c0    = i / (out_h * out_w);
            int out_y = i / out_w % out_h;
            int out_x = i % out_w;
            int in_y_0 = max(0, out_y * sy - ph);
            int in_y_1 = min(h, out_y * sy + kh - ph);
            int in_x_0 = max(0, out_x * sx - pw);
            int in_x_1 = min(w, out_x * sx + kw - pw);

            T val = 0;
            for (int y = in_y_0; y < in_y_1; ++y) {
              int offset_y = w * (y + h * c0);
              for (int x = in_x_0; x < in_x_1; ++x) {
                val = val + in[x + offset_y];
              }
            }
            out = val * coeff;
            ''', 'avg_pool_fwd')
        kern(x[0].reduced_view(), h, w, y_h, y_w, self.kh, self.kw,
             self.sy, self.sx, self.ph, self.pw, coeff, y)
        return y,

    def backward_cpu(self, x, gy):
        h, w = x[0].shape[2:]
        gcol = numpy.tile(gy[0][:, :, None, None],
                          (1, 1, self.kh, self.kw, 1, 1))
        gx = conv.col2im_cpu(gcol, self.sy, self.sx, self.ph, self.pw, h, w)
        gx /= self.kh * self.kw
        return gx,

    def backward_gpu(self, x, gy):
        if (cuda.cudnn_enabled and self.use_cudnn and
                pooling_2d._check_cudnn_acceptable_type(x[0].dtype)):
            return super(AveragePooling2D, self).backward_gpu(x, gy)

        n, c, h, w = x[0].shape
        y_h, y_w = gy[0].shape[2:]
        gx = cuda.cupy.empty_like(x[0])
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

               T val = 0;
               for (int out_y = out_y_0; out_y < out_y_1; ++out_y) {
                 for (int out_x = out_x_0; out_x < out_x_1; ++out_x) {
                   val = val + gy[out_x + out_w * (out_y + hc0)];
                 }
               }
               gx = val * coeff;
            ''', 'avg_pool_bwd')(gy[0].reduced_view(),
                                 h, w, y_h, y_w, self.kh, self.kw,
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
        ksize (int or pair of ints): Size of pooling window. ``ksize=k`` and
            ``ksize=(k, k)`` are equivalent.
        stride (int or pair of ints or None): Stride of pooling applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent. If ``None`` is
            specified, then it uses same stride as the pooling window size.
        pad (int or pair of ints): Spatial padding width for the input array.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        use_cudnn (bool): If ``True`` and cuDNN is enabled, then this function
            uses cuDNN as the core implementation.

    Returns:
        ~chainer.Variable: Output variable.

    .. note::

       This function currently does not support ``cover_all`` mode as
       :func:`max_pooling_2d`. Average pooling runs in non-cover-all mode.

    """
    return AveragePooling2D(ksize, stride, pad, False, use_cudnn)(x)

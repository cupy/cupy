from chainer.cuda import cupy
from chainer.functions.pooling import pooling_2d
from chainer.utils import conv
from chainer.utils import type_check

import numpy
import six


class Upsampling2D(pooling_2d.Pooling2D):

    """Upsampling over a set of 2d planes w/ indices used for max pooling."""

    def __init__(self, indexes, ksize, stride=None, pad=0, outsize=None,
                 cover_all=True):
        super(Upsampling2D, self).__init__(ksize, stride, pad, cover_all)
        self.indexes = indexes
        self.outh, self.outw = (None, None) if outsize is None else outsize

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(n_in == 1)
        x_type = in_types[0]

        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim == 4,
            x_type.shape == self.indexes.shape,
        )

        if self.outh is not None:
            expected_h = conv.get_conv_outsize(
                self.outh, self.kh, self.sy, self.ph, cover_all=self.cover_all)
            type_check.expect(x_type.shape[2] == expected_h)
        if self.outw is not None:
            expected_w = conv.get_conv_outsize(
                self.outw, self.kw, self.sx, self.pw, cover_all=self.cover_all)
            type_check.expect(x_type.shape[3] == expected_w)

    def forward_cpu(self, x):
        n, c, h, w = x[0].shape
        if self.outh is None:
            self.outh = conv.get_deconv_outsize(
                h, self.kh, self.sy, self.ph, cover_all=self.cover_all)
        if self.outw is None:
            self.outw = conv.get_deconv_outsize(
                w, self.kw, self.sx, self.pw, cover_all=self.cover_all)

        up_y = numpy.zeros((n, c, self.outh, self.outw), dtype=numpy.float32)
        up_y = conv.im2col_cpu(
            up_y, self.kh, self.kw, self.sy, self.sx, self.ph, self.pw,
            cover_all=self.cover_all)
        for n in six.moves.range(up_y.shape[0]):
            for c in six.moves.range(up_y.shape[1]):
                for ky in six.moves.range(up_y.shape[4]):
                    for kx in six.moves.range(up_y.shape[5]):
                        sy = self.indexes[n, c, ky, kx] // up_y.shape[3]
                        sx = self.indexes[n, c, ky, kx] % up_y.shape[3]
                        up_y[n, c, sy, sx, ky, kx] = x[0][n, c, ky, kx]
        up_y = conv.col2im_cpu(up_y, self.sy, self.sx, self.ph,
                               self.pw, self.outh, self.outw)
        return up_y,

    def forward_gpu(self, x):
        n, c, h, w = x[0].shape
        if self.outh is None:
            self.outh = conv.get_deconv_outsize(
                h, self.kh, self.sy, self.ph, cover_all=self.cover_all)
        if self.outw is None:
            self.outw = conv.get_deconv_outsize(
                w, self.kw, self.sx, self.pw, cover_all=self.cover_all)
        up_y = cupy.zeros((n, c, self.outh, self.outw), dtype=numpy.float32)
        up_y = conv.im2col_gpu(
            up_y, self.kh, self.kw, self.sy, self.sx, self.ph, self.pw,
            cover_all=self.cover_all)
        up_y = up_y.transpose(0, 1, 4, 5, 2, 3)
        n, c, sy, sx, ky, kx = up_y.shape
        indexes = cupy.asarray(self.indexes, dtype=numpy.int32)
        cupy.ElementwiseKernel(
            'int32 indexes, float32 x, int32 n, int32 c, int32 sy, int32 sx,'
            'int32 ky, int32 kx', 'raw float32 up_y',
            '''
            int yn = i / c / sy / sx;
            int yc = (i / sy / sx) % c;
            int ysy = (i / sx) % sy;
            int ysx = i % sx;
            up_y[yn * c * sy * sx * ky * kx + \
              yc * sy * sx * ky * kx + \
              ysy * sx * ky * kx + \
              ysx * ky * kx + \
              indexes] = x;
            ''',
            'upsampling_2d_fwd')(indexes, x[0], n, c, sy, sx, ky, kx, up_y)
        up_y = up_y.transpose(0, 1, 4, 5, 2, 3)
        up_y = conv.col2im_gpu(up_y, self.sy, self.sx, self.ph, self.pw,
                               self.outh, self.outw)
        return up_y,

    def backward_cpu(self, x, gy):
        gcol = conv.im2col_cpu(
            gy[0], self.kh, self.kw, self.sy, self.sx, self.ph, self.pw,
            cover_all=self.cover_all)

        gcol = gcol.transpose(0, 1, 4, 5, 2, 3)
        n, c, sy, sx, ky, kx = gcol.shape
        gcol = gcol.reshape((n, c, sy, sx, ky * kx))
        gx = numpy.empty((n, c, sy, sx), dtype=x[0].dtype)
        for n in six.moves.range(gcol.shape[0]):
            for c in six.moves.range(gcol.shape[1]):
                for sy in six.moves.range(gcol.shape[2]):
                    for sx in six.moves.range(gcol.shape[3]):
                        gx[n, c, sy, sx] = \
                            gcol[n, c, sy, sx][self.indexes[n, c, sy, sx]]
        return gx,

    def backward_gpu(self, x, gy):
        gcol = conv.im2col_gpu(
            gy[0], self.kh, self.kw, self.sy, self.sx, self.ph, self.pw,
            cover_all=self.cover_all)

        gcol = gcol.transpose(0, 1, 4, 5, 2, 3)
        n, c, sy, sx, ky, kx = gcol.shape
        gcol = gcol.reshape((n, c, sy, sx, ky * kx))
        indexes = cupy.asarray(self.indexes, dtype=numpy.int32)
        gx = cupy.empty((n, c, sy, sx), dtype=x[0].dtype)
        cupy.ElementwiseKernel(
            'int32 indexes, raw float32 gcol, int32 n, int32 c, int32 sy,'
            'int32 sx, int32 ky, int32 kx',
            'raw float32 gx',
            '''
            int ind_n = i / c / sy / sx;
            int ind_c = (i / sy / sx) % c;
            int ind_sy = (i / sx) % sy;
            int ind_sx = i % sx;
            int gcol_ky = indexes / kx;
            int gcol_kx = indexes % kx;
            float top_gx = gcol[ind_n * c * sy * sx * ky * kx + \
                                ind_c * sy * sx * ky * kx + \
                                ind_sy * sx * ky * kx + \
                                ind_sx * ky * kx + \
                                gcol_ky * kx + \
                                gcol_kx];
            gx[ind_n * c * sy * sx + \
               ind_c * sy * sx + \
               ind_sy * sx + \
               ind_sx] = top_gx;
            ''',
            'upsampling_2d_bwd')(indexes, gcol, n, c, sy, sx, ky, kx, gx)

        return gx,


def upsampling_2d(
        x, indexes, ksize, stride=None, pad=0, outsize=None, cover_all=True):
    """Upsampling using pooling indices.

    This function produces an upsampled image using pooling indices.

    .. admonition:: Example

        It should be noted that you need to specify ``use_cudnn=False`` when
        you create MaxPooling2D object because if cuDNN used for operating
        max pooling, ``indexes`` is never created and stored.

        >>> p = F.MaxPooling2D(2, 2, use_cudnn=False)
        >>> x = numpy.arange(1, 37).reshape(1, 1, 6, 6).astype('f')
        >>> x = chainer.Variable(x)
        >>> pooled_x = p(x)
        >>> upsampled_x = F.upsampling_2d(
        ...     pooled_x, p.indexes, p.kh, p.sy, p.ph, x.shape[2:])
        >>> upsampled_x.shape
        (1, 1, 6, 6)
        >>> upsampled_x.data
        array([[[[  0.,   0.,   0.,   0.,   0.,   0.],
                 [  0.,   8.,   0.,  10.,   0.,  12.],
                 [  0.,   0.,   0.,   0.,   0.,   0.],
                 [  0.,  20.,   0.,  22.,   0.,  24.],
                 [  0.,   0.,   0.,   0.,   0.,   0.],
                 [  0.,  32.,   0.,  34.,   0.,  36.]]]], dtype=float32)

    Args:
        x (~chainer.Variable): Input variable.
        indexes (~numpy.ndarray or ~cupy.ndarray): Index array that was used
            to calculate x with MaxPooling2D.
        ksize (int or (int, int)): ksize attribtue of MaxPooling2D object that
            is used to calculate x
        stride (int or (int, int)): stride attribtue of MaxPooling2D object
            that is used to calculate x
        pad (int or (int, int)): pad attribtue of MaxPooling2D object that is
            used to calculate x
        outsize ((int, int)): Expected output size (height, width).
        cover_all (bool): Whether cover_all is used in the MaxPooling2D object
            or not.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Upsampling2D(indexes, ksize, stride, pad, outsize, cover_all)(x)

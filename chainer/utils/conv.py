import numpy
import six

from chainer import cuda


def get_conv_outsize(size, k, s, p, cover_all=False):
    if cover_all:
        return (size + p * 2 - k + s - 1) // s + 1
    else:
        return (size + p * 2 - k) // s + 1


def get_deconv_outsize(size, k, s, p, cover_all=False):
    if cover_all:
        return s * (size - 1) + k - s + 1 - 2 * p
    else:
        return s * (size - 1) + k - 2 * p


def im2col_cpu(img, kh, kw, sy, sx, ph, pw, pval=0, cover_all=False):
    n, c, h, w = img.shape
    out_h = get_conv_outsize(h, kh, sy, ph, cover_all)
    out_w = get_conv_outsize(w, kw, sx, pw, cover_all)

    img = numpy.pad(img,
                    ((0, 0), (0, 0), (ph, ph + sy - 1), (pw, pw + sx - 1)),
                    mode='constant', constant_values=(pval,))
    col = numpy.ndarray((n, c, kh, kw, out_h, out_w), dtype=img.dtype)

    for i in six.moves.range(kh):
        i_lim = i + sy * out_h
        for j in six.moves.range(kw):
            j_lim = j + sx * out_w
            col[:, :, i, j, :, :] = img[:, :, i:i_lim:sy, j:j_lim:sx]

    return col


def im2col_gpu(img, kh, kw, sy, sx, ph, pw, cover_all=False):
    n, c, h, w = img.shape
    out_h = get_conv_outsize(h, kh, sy, ph, cover_all)
    out_w = get_conv_outsize(w, kw, sx, pw, cover_all)

    col = cuda.cupy.empty((n, c, kh, kw, out_h, out_w), dtype=img.dtype)
    cuda.elementwise(
        'raw T img, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw',
        'T col',
        '''
           int c0 = i / (kh * kw * out_h * out_w);
           int ky = i / (kw * out_h * out_w) % kh;
           int kx = i / (out_h * out_w) % kw;
           int out_y = i / out_w % out_h;
           int out_x = i % out_w;

           int in_y = ky + out_y * sy - ph;
           int in_x = kx + out_x * sx - pw;
           if (in_y >= 0 && in_y < h && in_x >= 0 && in_x < w) {
             col = img[in_x + w * (in_y + h * c0)];
           } else {
             col = 0;
           }
        ''',
        'im2col')(img.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, col)
    return col


def col2im_cpu(col, sy, sx, ph, pw, h, w):
    n, c, kh, kw, out_h, out_w = col.shape

    img = numpy.zeros((n, c, h + 2 * ph + sy - 1, w + 2 * pw + sx - 1),
                      dtype=col.dtype)
    for i in six.moves.range(kh):
        i_lim = i + sy * out_h
        for j in six.moves.range(kw):
            j_lim = j + sx * out_w
            img[:, :, i:i_lim:sy, j:j_lim:sx] += col[:, :, i, j, :, :]

    return img[:, :, ph:h + ph, pw:w + pw]


def col2im_gpu(col, sy, sx, ph, pw, h, w):
    n, c, kh, kw, out_h, out_w = col.shape

    img = cuda.cupy.empty((n, c, h, w), dtype=col.dtype)
    cuda.elementwise(
        'raw T col, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw',
        'T img',
        '''
           int c0 = i / (h * w);
           int y  = i / w % h + ph;
           int x  = i % w + pw;

           int out_y_0 = max(0,     (y - kh + sy) / sy);
           int out_y_1 = min(out_h, (y      + sy) / sy);
           int out_x_0 = max(0,     (x - kw + sx) / sx);
           int out_x_1 = min(out_w, (x      + sx) / sx);

           T val = 0;
           for (int out_y = out_y_0; out_y < out_y_1; ++out_y) {
             int ky = y - out_y * sy;
             for (int out_x = out_x_0; out_x < out_x_1; ++out_x) {
               int kx = x - out_x * sx;
               int k = out_y + out_h * (kx + kw * (ky + kh * c0));
               val += col[out_x + out_w * k];
             }
           }
           img = val;
        ''',
        'col2im')(col.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, img)
    return img

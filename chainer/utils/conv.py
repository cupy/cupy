import itertools
import numpy
import six

from chainer import cuda
from chainer.utils import conv_kernel


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


def im2col_nd_cpu(img, ks, ss, ps, pval=0, cover_all=False):
    # Assured consistency of dimensions of parameters by caller.
    n, c = img.shape[0:2]       # (n, c, d_1, d_2, ..., d_N)
    ds = img.shape[2:]
    outs = tuple([get_conv_outsize(d, k, s, p, cover_all)
                  for (d, k, s, p) in zip(ds, ks, ss, ps)])

    # Pad around image.
    pad_width = ((0, 0), (0, 0)) + tuple(
        [(p, p + s - 1) for (s, p) in zip(ss, ps)])
    img = numpy.pad(img, pad_width, mode='constant', constant_values=(pval,))

    # Make patch array with which we will compute correlation with filter.
    # shape: (n, c, k_1, k_2, ..., k_N, out_1, out_2, ..., out_N)
    shape = (n, c) + ks + outs
    col = numpy.ndarray(shape, dtype=img.dtype)

    # Fill the patch array.
    N = len(ds)
    colon = slice(None)
    for kxs in itertools.product(*[six.moves.range(k) for k in ks]):
        # col[:, :, kx_1, kx_2, ..., kx_N, :, :, ..., :]
        col_index = (colon, colon) + kxs + (colon,) * N
        # img[:, :, kx_1:kx_lim_1:s_1, ..., kx_N:kx_lim_N:s_N]
        kx_lims = tuple([kx + s * out for (kx, s, out) in zip(kxs, ss, outs)])
        img_index = (colon, colon) + tuple(
            [slice(kx, kx_lim, s)
             for (kx, kx_lim, s) in zip(kxs, kx_lims, ss)])
        col[col_index] = img[img_index]

    return col


def im2col_gpu(img, kh, kw, sy, sx, ph, pw, cover_all=False):
    n, c, h, w = img.shape
    out_h = get_conv_outsize(h, kh, sy, ph, cover_all)
    out_w = get_conv_outsize(w, kw, sx, pw, cover_all)

    col = cuda.empty((n, c, kh, kw, out_h, out_w), dtype=img.dtype)
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


def im2col_nd_gpu(img, ks, ss, ps, cover_all=False):
    # Assured consistency of dimensions of parameters by caller.
    n, c = img.shape[0:2]       # (n, c, d_1, d_2, ..., d_N)
    ds = img.shape[2:]
    N = len(ds)
    outs = tuple([get_conv_outsize(d, k, s, p, cover_all)
                  for (d, k, s, p) in zip(ds, ks, ss, ps)])

    # col_shape: (n, c, k_1, k_2, ..., k_N, out_1, out_2, ..., out_N)
    shape = (n, c) + ks + outs
    col = cuda.empty(shape, dtype=img.dtype)

    # TODO(takagi) Better to cache this code?
    in_params, out_params, operation, name = \
        conv_kernel.generate_im2col_nd_kernel(N)
    cuda.elementwise(in_params, out_params, operation, name)(
        img.reduced_view(), *(ds + outs + ks + ss + ps + (col,)))

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


def col2im_nd_cpu(col, ss, ps, ds):
    # Assured consistency of dimensions of parameters by caller.
    n, c = col.shape[:2]  # (n, c, kx_1, ..., kx_N, out_1, ..., out_N)
    mid = (len(col.shape) - 2) / 2 + 2
    ks = col.shape[2:mid]
    outs = col.shape[mid:]
    colon = slice(None)

    # Image with padded size.
    img_shape = (n, c) + tuple([d + 2 * p + s - 1
                                for (d, p, s) in zip(ds, ps, ss)])
    img = numpy.zeros(img_shape, dtype=col.dtype)
    for kxs in itertools.product(*[six.moves.range(k) for k in ks]):
        # (:, :, kx_1:kx_lim_1:s_1, ..., kx_N:kx_lim_N:s_N)
        kx_lims = tuple([kx + s * out for (kx, s, out) in zip(kxs, ss, outs)])
        img_index = (colon, colon) + tuple(
            [slice(kx, kx_lim, s)
             for (kx, kx_lim, s) in zip(kxs, kx_lims, ss)])
        # (:, :, kx_1, kx_2, ..., kx_N, :, :, ..., :)
        col_index = (colon, colon) + kxs + (colon,) * len(outs)
        img[img_index] += col[col_index]

    # (:, :, p_1:d_1 + p_1, p_2:d_2 + p_2, ..., p_N:d_N + p_N]
    img_index = (colon, colon) + tuple(
        [slice(p, d + p) for (p, d) in zip(ps, ds)])
    return img[img_index]


def col2im_gpu(col, sy, sx, ph, pw, h, w):
    n, c, kh, kw, out_h, out_w = col.shape

    img = cuda.empty((n, c, h, w), dtype=col.dtype)
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
               val = val + col[out_x + out_w * k];
             }
           }
           img = val;
        ''',
        'col2im')(col.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, img)
    return img


def col2im_nd_gpu(col, ss, ps, ds):
    # Assured consistency of dimensions of parameters by caller.
    n, c = col.shape[:2]        # (n, c, k_1, ..., k_N, out_1, ..., out_N)
    mid = (len(col.shape) - 2) / 2 + 2
    ks = col.shape[2:mid]
    outs = col.shape[mid:]
    N = len(ds)

    img_shape = (n, c) + ds     # (n, c, d_1, d_2, ..., d_N)
    img = cuda.empty(img_shape, dtype=col.dtype)

    # TODO(takagi) Better to cache this code?
    in_params, out_params, operation, name =\
        conv_kernel.generate_col2im_nd_kernel(N)
    cuda.elementwise(in_params, out_params, operation, name)(
        col.reduced_view(), *(ds + outs + ks + ss + ps + (img,)))

    return img

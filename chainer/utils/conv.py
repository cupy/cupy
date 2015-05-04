import numpy
from chainer import cuda

def _outsize(size, k, s, p):
    return (size + p * 2 - k) / s + 1

def im2col_cpu(img, kh, kw, sy, sx, ph, pw):
    n, c, h, w = img.shape
    out_h = _outsize(h, kh, sy, ph)
    out_w = _outsize(w, kw, sx, pw)

    img = numpy.pad(img, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode='constant')
    col = numpy.ndarray((n, c, kh, kw, out_h, out_w), dtype=img.dtype)

    for i in xrange(kh):
        i_lim = i + sy * out_h
        for j in xrange(kw):
            j_lim = j + sx * out_w
            col[:, :, i, j, :, :] = img[:, :, i:i_lim:sy, j:j_lim:sx]

    return col

def col2im_cpu(col, sy, sx, ph, pw, h, w):
    n, c, kh, kw, out_h, out_w = col.shape

    img = numpy.zeros((n, c, h + 2 * ph, w + 2 * pw), dtype=col.dtype)
    for i in xrange(kh):
        i_lim = i + sy * out_h
        for j in xrange(kw):
            j_lim = j + sx * out_w
            img[:, :, i:i_lim:sy, j:j_lim:sx] += col[:, :, i, j, :, :]

    return img[:, :, ph:-ph, pw:-pw]

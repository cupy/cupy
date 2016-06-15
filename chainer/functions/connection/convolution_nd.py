import numpy
import operator

from functools import reduce
from six import moves

from chainer import cuda
from chainer import function
from chainer.utils import conv
from chainer.utils import type_check


def _ensure_tuple(x, n):
    if hasattr(x, '__getitem__'):
        return x
    return tuple([x] * n)


class ConvolutionND(function.Function):

    def __init__(self, N, stride=1, pad=0, use_cudnn=True, cover_all=False):
        self.N = N
        self.stride = _ensure_tuple(stride, N)
        assert len(self.stride) == N
        self.pad = _ensure_tuple(pad, N)
        assert len(self.pad) == N
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
            x_type.ndim == self.N + 2,
            w_type.ndim == self.N + 2,
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
        ks = W.shape[2:]
        ss = self.stride
        ps = self.pad
        N = self.N

        # Make patch array.
        self.col = conv.im2col_nd_cpu(x, ks, ss, ps, cover_all=self.cover_all)

        # Compute correlation.
        axes = tuple(moves.range(1, N+2))  # (1, 2, ..., N+1)
        y = numpy.tensordot(self.col, W, (axes, axes)).astype(x.dtype)

        # Apply bias if given.
        if b is not None:
            y += b

        # Roll c_O before the second in (n, y_1, y_2, ..., y_N, c_O).
        return numpy.rollaxis(y, N+1, 1),

    def forward_gpu(self, inputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        out_c = W.shape[0]      # (c_O, _, k_1, k_2, ..., k_N)
        ks = W.shape[2:]
        n, c = x.shape[:2]      # (n, c_I, d_1, d_2, ..., d_N)
        ds = x.shape[2:]
        ss = self.stride
        ps = self.pad
        N = self.N

        # Compute output image's dimensions.
        outs = tuple(
            [conv.get_conv_outsize(d, k, s, p, cover_all=self.cover_all)
             for (d, k, s, p) in zip(ds, ks, ss, ps)])

        # Make empty array for result.
        y_shape = (n, out_c) + outs  # (n, c_O, out_1, out_2, ..., out_N)
        y = cuda.empty(y_shape, dtype=x.dtype)

        # TODO(takagi) cuDNN version here.

        # Make patch array.
        self.col = conv.im2col_nd_gpu(x, ks, ss, ps, cover_all=self.cover_all)

        # Compute correlation.
        W_mat = W.reshape(out_c, -1)
        col_mats = self.col.reshape(n, -1, reduce(operator.mul, outs))
        y_mats = y.reshape(n, out_c, -1)
        # TODO(takagi): Use streams or batch gemm
        for i in moves.range(n):
            y_mats[i] = W_mat.dot(col_mats[i])

        # Apply bias if given.
        # TODO(takagi): Support unshared bias
        if b is not None:
            colon = slice(None)
            index = (colon,) + (None,) * N  # (:, None, ..., None)
            y += b[index]

        return y,

    def backward_cpu(self, inputs, grad_outputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        gy = grad_outputs[0]   # (n, c_O, out_1, out_2, ..., out_N)
        ds = x.shape[2:]       # (n, c_I, d_1, d_2, ..., d_N)
        ss = self.stride
        ps = self.pad
        N = self.N

        # Compute filter weight gradient.
        # (n, _, out_1, out_2, ..., out_N)
        out_axes = (0,) + tuple(moves.range(2, N+2))
        # (n, _, _, ..., _, out_1, out_2, ..., out_N)
        col_axes = (0,) + tuple(moves.range(N+2, N*2+2))
        gW = numpy.tensordot(
            gy, self.col, (out_axes, col_axes)).astype(W.dtype)

        # Compute patch array gradient.
        gcol = numpy.tensordot(W, gy, (0, 1)).astype(x.dtype)
        gcol = numpy.rollaxis(gcol, N + 1)

        # Compute input gradient.
        gx = conv.col2im_nd_cpu(gcol, ss, ps, ds)

        # Compute bias gradient if given and return gradients.
        if b is None:
            return gx, gW
        else:
            # (n, _, out_1, out_2, ..., out_N)
            axis = (0,) + tuple(moves.range(2, N+2))
            gb = gy.sum(axis=axis)
            return gx, gW, gb

    def backward_gpu(self, inputs, grad_putputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        gy = grad_putputs[0]   # (n, c_O, out_1, out_2, ..., out_N)
        out_c = gy.shape[1]
        outs = gy.shape[2:]
        n, c = x.shape[:2]     # (n, c_I, d_1, d_2, ..., d_N)
        ds = x.shape[2:]
        ks = W.shape[2:]       # (_, _, k_1, k_2, ..., k_N)
        ss = self.stride
        ps = self.pad
        N = self.N

        # Compute filter weight gradient.
        gW = cuda.empty_like(W)

        # TODO(takagi) cuDNN version here.

        gW_mat = gW.reshape(out_c, reduce(operator.mul, ks, c))
        col_mats = self.col.reshape(
            n, reduce(operator.mul, ks, c), reduce(operator.mul, outs))
        gy_mats = gy.reshape(n, out_c, reduce(operator.mul, outs))
        # TODO(takagi): Use streams or batch gemm
        gW_mat[...] = 0
        for i in moves.range(n):
            gW_mat += cuda.dot(gy_mats[i], col_mats[i].T)

        # Compute patch array gradient.
        W_mat = W.reshape(out_c, -1)
        gcol = cuda.empty_like(self.col)
        gcol_mats = gcol.reshape(
            n, reduce(operator.mul, ks, c), reduce(operator.mul, outs))
        for i in moves.range(n):
            gcol_mats[i] = cuda.dot(W_mat.T, gy_mats[i])

        # Compute input gradient.
        gx = conv.col2im_nd_gpu(gcol, ss, ps, ds)

        # Compute bias gradient if given.
        if b is not None:
            # (n, _, out_1, out_2, ..., out_N)
            axis = (0,) + tuple(moves.range(2, N+2))
            gb = gy.sum(axis=axis)

        if b is None:
            return gx, gW
        else:
            return gx, gW, gb


def convolution_nd(x, W, b=None, stride=1, pad=0, use_cudnn=True,
                   cover_all=False):
    """N-dimensional convolution function.

    This is an implementation of N-dimensional convolution which is generalized
    two-dimensional convolution in ConvNets. It takes three variables: an input
    image ``x``, a filter weight ``W`` and a bias vector ``b``.

    Notation: here is a notation for dimensionalities.

    - :math:`N` is the number of spacial dimensions.
    - :math:`n` is the batch size.
    - :math:`c_I` and :math:`c_O` are the number of the input and output,
      respectively.
    - :math:`d_1, d_2, ..., d_N` are the sizes of each axis of the input image.
    - :math:`k_1, k_2, ..., k_N` are the sizes of each axis of the filters.

    Args:
        x (~chainer.Variable): Input variable of shape
            :math:`(n, c_I, d_1, d_2, ..., d_N)`.
        W (~chainer.Variable):mi Weight variable of shape
            :math:`(c_O, c_I, k_1, k_2, ..., k_N)`.
        b (~chainer.Variable): Bias variable of length :math:`(c_O,)`
            (optional).
        stride (int or tuple of ints): Stride of filter applications
            :math:`(s_1, s_2, ..., s_N)`. ``stride=s`` is equivalent to
             ``(s, s, ..., s)``.
        pad (int or tuple of ints): Spatial padding width for input arrays
            :math:`(p_1, p_2, ..., p_N)`. ``pad=p`` is equivalent to
            ``(p, p, ..., p)``.
        use_cudnn (bool): If ``True``, then this function uses cuDNN if
             available.
        cover_all (bool): If ``True``, all spatial locations are convoluted
            into some output pixels. It may make the output size larger.

    Returns:
        ~chainer.Variable: Output variable.

    .. seealso:: :class:`ConvolutionND`, :function:`convolution_2d`
    """
    N = len(x.data.shape[2:])
    func = ConvolutionND(N, stride, pad, use_cudnn, cover_all)
    if b is None:
        return func(x, W)
    else:
        return func(x, W, b)

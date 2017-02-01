import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _transpose(xs, length):
    if length == 0:
        return ()

    xp = cuda.get_array_module(*xs)
    lengths = numpy.empty(length, dtype='i')
    end = length
    for i, x in enumerate(xs):
        len_x = len(x)
        if len_x == end:
            continue
        lengths[len_x:end] = i
        end = len_x
    lengths[0:end] = len(xs)

    if xp is numpy:
        dtype = xs[0].dtype
        unit = xs[0].shape[1:]

        outs = tuple([xp.empty((l,) + unit, dtype=dtype) for l in lengths])
        for i, x in enumerate(xs):
            for p, xi in enumerate(x):
                outs[p][i] = xi

    else:
        offsets1 = numpy.empty(len(xs) + 1, dtype='i')
        offsets1[0] = 0
        numpy.cumsum([len(x) for x in xs], out=offsets1[1:])

        offsets2 = numpy.empty(length + 1, dtype='i')
        offsets2[0] = 0
        numpy.cumsum(lengths, dtype='i', out=offsets2[1:])

        x = xp.concatenate(xs, axis=0)
        o = xp.empty_like(x)
        unit = xs[0].size // len(xs[0])
        size = length * len(xs) * unit
        cuda.elementwise(
            'int32 len, int32 unit, raw int32 off1, raw int32 off2, raw T vs',
            'raw T hs',
            '''
            int ind = i / unit;
            int off = i - ind * unit;
            int y = ind / len;
            int x = ind - y * len;
            if (off2[x] + y < off2[x + 1]) {
              hs[(off2[x] + y) * unit + off] = vs[(off1[y] + x) * unit + off];
            }
            ''',
            'transpose_sequence'
        )(length, unit, cuda.to_gpu(offsets1), cuda.to_gpu(offsets2), x, o,
          size=size)
        outs = tuple(xp.split(o, offsets2[1:-1]))

    return outs


class TransposeSequence(function.Function):

    """Function that transposes a list of Variables."""

    def check_type_forward(self, xs_type):
        for p, n in zip(xs_type, xs_type[1:]):
            type_check.expect(
                p.shape[0] >= n.shape[0],
                p.shape[1:] == n.shape[1:],
            )

    def forward(self, xs):
        if len(xs) == 0:
            return ()
        return _transpose(xs, len(xs[0]))

    def backward(self, xs, gs):
        return _transpose(gs, len(xs))


def transpose_sequence(xs):
    """Transpose a list of Variables.

    This function transposes a list of :class:`~chainer.Variable` s and returns
    a list of :class:`Variable` s.
    For example a user gives ``[(0, 1, 2, 3), (4, 5), (6)]``, the function
    returns ``[(0, 4, 6), (1, 5), (2), (3)]``.
    Note that a given list needs to be sorted by each length of
    :class:`~chainer.Variable`.

    Args:
        xs (list of ~chainer.Variable): Variables to transpose.

    Returns:
        tuple or Variable: Transposed list.
    """
    ys = TransposeSequence()(*xs)
    if not isinstance(ys, tuple):
        ys = (ys,)
    return ys

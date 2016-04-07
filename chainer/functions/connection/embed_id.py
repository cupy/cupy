import numpy
import six

import chainer
from chainer import cuda
from chainer import function
from chainer.utils import type_check


class EmbedIDFunction(function.Function):

    def __init__(self, ignore_label=None):
        self.ignore_label = ignore_label

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, w_type = in_types
        type_check.expect(
            x_type.dtype == numpy.int32,
            x_type.ndim >= 1,
        )
        type_check.expect(
            w_type.dtype == numpy.float32,
            w_type.ndim == 2
        )

    def forward(self, inputs):
        x, W = inputs

        if chainer.is_debug():
            if not ((0 <= x).all() and
                    (x < len(W)).all()):
                msg = 'Each `x` value need to satisfty `0 <= x < len(W)`'
                raise ValueError(msg)

        if self.ignore_label is not None:
            xp = cuda.get_array_module(*inputs)
            mask = (x == self.ignore_label)
            return xp.where(
                mask[..., None], 0, W.take(xp.where(mask, 0, x), axis=0)),

        return W.take(x, axis=0),

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        x, W = inputs
        gy = grad_outputs[0]
        gW = xp.zeros_like(W)

        if xp is numpy:
            # It is equivalent to `numpy.add.at(gW, x, gy)` but ufunc.at is
            # too slow.
            for ix, igy in six.moves.zip(x.ravel(),
                                         gy.reshape(x.size, -1)):
                if ix == self.ignore_label:
                    continue
                gW[ix] += igy
        else:
            if self.ignore_label is None:
                cuda.elementwise(
                    'T gy, int32 x, int32 n_out', 'raw T gW',
                    'int w_ind[] = {x, i % n_out}; atomicAdd(&gW[w_ind], gy)',
                    'embed_id_bwd')(
                        gy, xp.expand_dims(x, -1), gW.shape[1], gW)
            else:
                cuda.elementwise(
                    'T gy, int32 x, int32 n_out, int32 ignore', 'raw T gW',
                    '''
                    if (x != ignore) {
                      int w_ind[] = {x, i % n_out};
                      atomicAdd(&gW[w_ind], gy);
                    }
                    ''',
                    'embed_id_bwd_ignore_label')(
                        gy, xp.expand_dims(x, -1), gW.shape[1],
                        self.ignore_label, gW)
        return None, gW


def embed_id(x, W, ignore_label=None):
    """Efficient linear function for one-hot input.

    This function implements so called *word embedding*. It takes two
    arguments: a set of IDs (words) ``x`` in :math:`B` dimensional integer
    vector, and a set of all ID (word) embeddings ``W`` in :math:`V \\times d`
    float32 matrix. It outputs :math:`B \\times d` matrix whose ``i``-th
    column is the ``x[i]``-th column of ``W``.

    This function is only differentiable on the input ``W``.

    Args:
        x (~chainer.Variable): Batch vectors of IDs.
        W (~chainer.Variable): Representation of each ID (a.k.a.
            word embeddings).
        ignore_label (int or None): If ``ignore_label`` is an int value,
            ``i``-th column of return value is filled with ``0``.

    Returns:
        ~chainer.Variable: Output variable.

    .. seealso:: :class:`EmbedID`

    """
    return EmbedIDFunction(ignore_label=ignore_label)(x, W)

import numpy
import six

import chainer
from chainer import cuda
from chainer import function
from chainer.utils import type_check


class SelectItem(function.Function):

    """Select elements stored in given indices."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

        x_type, t_type = in_types
        type_check.expect(
            t_type.dtype.kind == 'i',
            x_type.ndim == 2,
            t_type.ndim == 1,
            x_type.shape[0] == t_type.shape[0],
        )

    def forward(self, inputs):
        x, t = inputs
        if chainer.is_debug():
            if not ((0 <= t).all() and
                    (t < x.shape[1]).all()):
                msg = 'Each label `t` need to satisfty `0 <= t < x.shape[1]`'
                raise ValueError(msg)

        xp = cuda.get_array_module(x)
        if xp is numpy:
            # This code is equivalent to `t.choose(x.T)`, but `numpy.choose`
            # does not work when `x.shape[1] > 32`.
            return x[six.moves.range(t.size), t],
        else:
            y = cuda.elementwise(
                'S t, raw T x',
                'T y',
                'int ind[] = {i, t}; y = x[ind];',
                'getitem_fwd'
            )(t, x)
            return y,

    def backward_cpu(self, inputs, grad_outputs):
        x, t = inputs
        gloss = grad_outputs[0]
        gx = numpy.zeros_like(x)
        gx[six.moves.range(t.size), t] = gloss
        return gx, None

    def backward_gpu(self, inputs, grad_outputs):
        x, t = inputs
        gloss = grad_outputs[0]
        gx = cuda.cupy.zeros_like(x)
        gx = cuda.elementwise(
            'S t, T gloss',
            'raw T gx',
            'int ind[] = {i, t}; gx[ind] = gloss;',
            'getitem_bwd'
        )(t, gloss, gx)
        return gx, None


def select_item(x, t):
    """Select elements stored in given indices.

    This function returns ``t.choose(x.T)``, that means
    ``y[i] == x[i, t[i]]`` for all ``i``.

    Args:
        x (Variable): Variable storing arrays.
        t (Variable): Variable storing index numbers.

    Returns:
        ~chainer.Variable: Variable that holds ``t``-th element of ``x``.

    """
    return SelectItem()(x, t)

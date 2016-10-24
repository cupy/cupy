#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class SquaredDifference(function.Function):
    """Squared difference of input variables."""

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 2,
            in_types[0].dtype.kind == 'f',
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == in_types[1].shape
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x1, x2 = inputs
        self.difference = x1 - x2
        y = xp.square(self.difference)
        return y,

    def backward(self, inputs, grads):
        x1, x2 = inputs
        gy, = grads
        gx1 = gy * self.difference * 2
        gx2 = gy * self.difference * 2
        return gx1, gx2

def squared_difference(x1, x2):
    """Squared difference of input variables.

    Args:
        x1 (~chainer.Variable): Input variables to be compared.
        x2 (~chainer.Variable): Input variables to be compared.

    Returns:
        ~chainer.Variable: (x1 - x2)(x1 - x2) element-wise. Output variable. 
    """
    return SquaredDifference()(x1, x2)


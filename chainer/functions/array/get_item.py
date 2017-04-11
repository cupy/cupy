import numpy

import chainer
from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check
from chainer import variable


class GetItem(function.Function):

    """Function that slices array and extract elements."""

    def __init__(self, slices):
        if isinstance(slices, list):
            if all([isinstance(s, int) for s in slices]):
                slices = slices,
            slices = tuple(slices)
        elif not isinstance(slices, tuple):
            slices = slices,

        if chainer.is_debug():
            n_ellipses = 0
            for s in slices:
                if s is Ellipsis:
                    n_ellipses += 1
            if n_ellipses > 1:
                raise ValueError('Only one Ellipsis is allowed')

        self.slices = slices

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        n_nones = len([item for item in self.slices if item is None])
        valid_slice = len(self.slices) - n_nones
        type_check.expect(in_types[0].ndim >= valid_slice)

    def forward(self, xs):
        ary = xs[0]
        return utils.force_array(ary[self.slices]),

    def backward(self, xs, gys):
        xp = cuda.get_array_module(*xs)
        gy = gys[0]
        gx = xp.zeros_like(xs[0])
        if xp is numpy:
            numpy.add.at(gx, self.slices, gy)
        else:
            gx.scatter_add(self.slices, gy)
        return gx,


def get_item(x, slices):
    """Extract elements from array with specified shape, axes and offsets.

    Args:
        x (~chainer.Variable): A variable to be sliced.
        slices (int, slice, Ellipsis, None, integer array-like, boolean\
        array-like or tuple of them):
            It is an integer, a slice, an ellipsis,
            a numpy.newaxis, an integer array-like, a boolean array-like
            or tuple of them.

    Returns:
        Variable: :class:`~chainer.Variable` object
            which contains sliced array of ``x``.

    .. note::

        It only supports types that are supported by CUDA's atomicAdd when
        an integer array is included in ``slices``.
        The supported types are ``numpy.float32``, ``numpy.int32``,
        ``numpy.uint32``, ``numpy.uint64`` and ``numpy.ulonglong``.

    .. note::

        It does not support ``slices`` that contains multiple boolean arrays.

    .. note::

       See NumPy document for details of `indexing
       <http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html>`_.

    """
    return GetItem(slices)(x)


def install_variable_get_item():
    variable.Variable.__getitem__ = get_item

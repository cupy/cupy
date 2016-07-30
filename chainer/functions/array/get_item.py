import collections

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
        if not isinstance(slices, collections.Iterable):
            slices = tuple([slices])

        if chainer.is_debug():
            n_ellipses = 0
            for s in slices:
                if numpy.isscalar(s) or s is None or isinstance(s, slice):
                    pass
                elif s is Ellipsis:
                    n_ellipses += 1
                else:
                    raise ValueError('Only basic indexing is supported')
            if n_ellipses > 1:
                raise ValueError('Only one Ellipsis is allowed')

        self.slices = slices

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        valid_slice = len(self.slices) - self.slices.count(None)
        type_check.expect(in_types[0].ndim >= valid_slice)

    def forward(self, xs):
        ary = xs[0]
        return utils.force_array(ary[tuple(self.slices)]),

    def backward(self, xs, gys):
        xp = cuda.get_array_module(*xs)
        gy = gys[0]
        gx = xp.zeros_like(xs[0])
        gx[tuple(self.slices)] = gy
        return gx,


def get_item(x, slices):
    """Extract elements from array with specified shape, axes and offsets.

    Args:
        x (tuple of Variables): Variable to be sliced.
        slices (int, slice, None or Ellipsis or tuple of them): Basic slicing
            to slice a variable. It supports ``int``, ``slice``, ``newaxis``
            (equivalent to ``None``) and ``Ellipsis``.

    Returns:
        Variable: :class:`~chainer.Variable` object
            which contains sliced array of ``x``.

    .. note::

       See NumPy document for details of `indexing
       <http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html>`_.

    """
    return GetItem(slices)(x)


def install_variable_get_item():
    variable.Variable.__getitem__ = get_item

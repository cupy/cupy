# flake8: NOQA
# "flake8: NOQA" to suppress warning "H104  File contains nothing but comments"

# class s_(object):

import numpy
import six

import cupy
from cupy import core
from cupy.creation import from_data
from cupy.manipulation import join


class AxisConcatenator(object):
    """Translates slice objects to concatenation along an axis.

    For detailed documentation on usage, see :func:`cupy.r_`.
    This implementation is partially borrowed from NumPy's one.

    """

    def _output_obj(self, obj, ndim, ndmin, trans1d):
        k2 = ndmin - ndim
        if trans1d < 0:
            trans1d += k2 + 1
        defaxes = list(six.moves.range(ndmin))
        k1 = trans1d
        axes = defaxes[:k1] + defaxes[k2:] + \
            defaxes[k1:k2]
        return obj.transpose(axes)

    def __init__(self, axis=0, matrix=False, ndmin=1, trans1d=-1):
        self.axis = axis
        self.trans1d = trans1d
        self.matrix = matrix
        self.ndmin = ndmin

    def __getitem__(self, key):
        trans1d = self.trans1d
        ndmin = self.ndmin
        objs = []
        scalars = []
        arraytypes = []
        scalartypes = []
        if isinstance(key, six.string_types):
            raise NotImplementedError
        if not isinstance(key, tuple):
            key = (key,)

        for i, k in enumerate(key):
            scalar = False
            if isinstance(k, slice):
                raise NotImplementedError
            elif isinstance(k, six.string_types):
                if i != 0:
                    raise ValueError(
                        'special directives must be the first entry.')
                raise NotImplementedError
            elif type(k) in numpy.ScalarType:
                newobj = from_data.array(k, ndmin=ndmin)
                scalars.append(i)
                scalar = True
                scalartypes.append(newobj.dtype)
            else:
                newobj = from_data.array(k, copy=False, ndmin=ndmin)
                if ndmin > 1:
                    ndim = from_data.array(k, copy=False).ndim
                    if trans1d != -1 and ndim < ndmin:
                        newobj = self._output_obj(newobj, ndim, ndmin, trans1d)

            objs.append(newobj)
            if not scalar and isinstance(newobj, core.ndarray):
                arraytypes.append(newobj.dtype)

        final_dtype = numpy.find_common_type(arraytypes, scalartypes)
        if final_dtype is not None:
            for k in scalars:
                objs[k] = objs[k].astype(final_dtype)

        return join.concatenate(tuple(objs), axis=self.axis)

    def __len__(self):
        return 0


class CClass(AxisConcatenator):

    def __init__(self):
        super(CClass, self).__init__(-1, ndmin=2, trans1d=0)


c_ = CClass()
"""Translates slice objects to concatenation along the second axis.

This is a CuPy object that corresponds to :func:`cupy.r_`, which is
useful because of its common occurrence. In particular, arrays will be
stacked along their last axis after being upgraded to at least 2-D with
1's post-pended to the shape (column vectors made out of 1-D arrays).

For detailed documentation, see :func:`r_`.

This implementation is partially borrowed from NumPy's one.

Args:
    Not a function, so takes no parameters

Returns:
    cupy.ndarray: Joined array.

.. seealso:: :func:`numpy.c_`

Examples
--------
>>> a = cupy.array([[1, 2, 3]], dtype=np.int32)
>>> b = cupy.array([[4, 5, 6]], dtype=np.int32)
>>> cupy.c_[a, 0, 0, b]
array([[1, 2, 3, 0, 0, 4, 5, 6]], dtype=int32)

"""


class RClass(AxisConcatenator):

    def __init__(self):
        super(RClass, self).__init__()


r_ = RClass()
"""Translates slice objects to concatenation along the first axis.

This is a simple way to build up arrays quickly.
If the index expression contains comma separated arrays, then stack
them along their first axis.

This object can build up from normal CuPy arrays.
Therefore, the other objects (e.g. writing strings like '2,3,4',
or using imaginary numbers like [1,2,3j],
or using string integers like '-1') are not implemented yet
compared with NumPy.

This implementation is partially borrowed from NumPy's one.

Args:
    Not a function, so takes no parameters

Returns:
    cupy.ndarray: Joined array.

.. seealso:: :func:`numpy.r_`

Examples
--------
>>> a = cupy.array([1, 2, 3], dtype=np.int32)
>>> b = cupy.array([4, 5, 6], dtype=np.int32)
>>> cupy.r_[a, 0, 0, b]
array([1, 2, 3, 0, 0, 4, 5, 6], dtype=int32)

"""

# TODO(okuta): Implement indices


def ix_(*args):
    """Construct an open mesh from multiple sequences.

    This function takes N 1-D sequences and returns N outputs with N
    dimensions each, such that the shape is 1 in all but one dimension
    and the dimension with the non-unit shape value cycles through all
    N dimensions.

    Using `ix_` one can quickly construct index arrays that will index
    the cross product. ``a[cupy.ix_([1,3],[2,5])]`` returns the array
    ``[[a[1,2] a[1,5]], [a[3,2] a[3,5]]]``.

    Args:
        *args: 1-D sequences

    Returns:
        tuple of ndarrays:
        N arrays with N dimensions each, with N the number of input sequences.
        Together these arrays form an open mesh.

    Examples
    --------
    >>> a = cupy.arange(10).reshape(2, 5)
    >>> a
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])
    >>> ixgrid = cupy.ix_([0,1], [2,4])
    >>> ixgrid
    (array([[0],
           [1]]), array([[2, 4]]))

     .. seealso:: :func:`numpy.ix_`

    """
    out = []
    nd = len(args)
    for k, new in enumerate(args):
        new = from_data.asarray(new)
        if new.ndim != 1:
            raise ValueError("Cross index must be 1 dimensional")
        if new.size == 0:
            # Explicitly type empty arrays to avoid float default
            new = new.astype(numpy.intp)
        if cupy.issubdtype(new.dtype, cupy.bool_):
            new, = new.nonzero()
        new = new.reshape((1,) * k + (new.size,) + (1,) * (nd - k - 1))
        out.append(new)
    return tuple(out)

# TODO(okuta): Implement ravel_multi_index


# TODO(okuta): Implement unravel_index


# TODO(okuta): Implement diag_indices


# TODO(okuta): Implement diag_indices_from


# TODO(okuta): Implement mask_indices


# TODO(okuta): Implement tril_indices


# TODO(okuta): Implement tril_indices_from


# TODO(okuta): Implement triu_indices


# TODO(okuta): Implement triu_indices_from

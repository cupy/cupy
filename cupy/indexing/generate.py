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

This is a CuPy object that corresponds to :obj:`cupy.r_`, which is
useful because of its common occurrence. In particular, arrays will be
stacked along their last axis after being upgraded to at least 2-D with
1's post-pended to the shape (column vectors made out of 1-D arrays).

For detailed documentation, see :obj:`r_`.

This implementation is partially borrowed from NumPy's one.

Returns:
    cupy.ndarray: Joined array.

.. seealso:: :obj:`numpy.c_`

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

Returns:
    cupy.ndarray: Joined array.

.. seealso:: :obj:`numpy.r_`

Examples
--------
>>> a = cupy.array([1, 2, 3], dtype=np.int32)
>>> b = cupy.array([4, 5, 6], dtype=np.int32)
>>> cupy.r_[a, 0, 0, b]
array([1, 2, 3, 0, 0, 4, 5, 6], dtype=int32)

"""


def indices(dimensions, dtype=int):
    """Returns an array representing the indices of a grid.

    Computes an array where the subarrays contain index values 0,1,...
    varying only along the corresponding axis.

    Args:
        dimensions: The shape of the grid.
        dtype: Data type specifier. It is int by default.

    Returns:
        ndarray:
        The array of grid indices,
        ``grid.shape = (len(dimensions),) + tuple(dimensions)``.

    Examples
    --------
    >>> grid = cupy.indices((2, 3))
    >>> grid.shape
    (2, 2, 3)
    >>> grid[0]        # row indices
    array([[0, 0, 0],
           [1, 1, 1]])
    >>> grid[1]        # column indices
    array([[0, 1, 2],
           [0, 1, 2]])

    .. seealso:: :func:`numpy.indices`

    """
    dimensions = tuple(dimensions)
    N = len(dimensions)
    shape = (1,) * N
    res = cupy.empty((N,) + dimensions, dtype=dtype)
    for i, dim in enumerate(dimensions):
        res[i] = cupy.arange(dim, dtype=dtype).reshape(
            shape[:i] + (dim,) + shape[i + 1:]
        )
    return res


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
            raise ValueError('Cross index must be 1 dimensional')
        if new.size == 0:
            # Explicitly type empty arrays to avoid float default
            new = new.astype(numpy.intp)
        if cupy.issubdtype(new.dtype, cupy.bool_):
            new, = new.nonzero()
        new = new.reshape((1,) * k + (new.size,) + (1,) * (nd - k - 1))
        out.append(new)
    return tuple(out)

# TODO(okuta): Implement ravel_multi_index


def unravel_index(indices, dims, order='C'):
    """Converts array of flat indices into a tuple of coordinate arrays.

    Args:
        indices (cupy.ndarray): An integer array whose elements are indices
            into the flattened version of an array of dimensions :obj:`dims`.
        dims (tuple of ints): The shape of the array to use for unraveling
            indices.
        order ('C' or 'F'): Determines whether the indices should be viewed as
            indexing in row-major (C-style) or column-major (Fortran-style)
            order.

    Returns:
        tuple of ndarrays:
        Each array in the tuple has the same shape as the indices array.

    Examples
    --------
    >>> cupy.unravel_index(cupy.array([22, 41, 37]), (7, 6))
    (array([3, 6, 6]), array([4, 5, 1]))
    >>> cupy.unravel_index(cupy.array([31, 41, 13]), (7, 6), order='F')
    (array([3, 6, 6]), array([4, 5, 1]))

    .. seealso:: :func:`numpy.unravel_index`

    """
    if order is None:
        order = 'C'

    if order == 'C':
        dims = reversed(dims)
    elif order == 'F':
        pass
    else:
        raise TypeError('order not understood')

    if not cupy.can_cast(indices, cupy.int64, 'same_kind'):
        raise TypeError(
            'Iterator operand 0 dtype could not be cast '
            'from dtype(\'{}\') to dtype(\'{}\') '
            'according to the rule \'same_kind\''.format(
                indices.dtype, cupy.int64().dtype))

    if (indices < 0).any():
        raise ValueError('invalid entry in index array')

    unraveled_coords = []
    for dim in dims:
        unraveled_coords.append(indices % dim)
        indices = indices // dim

    if (indices > 0).any():
        raise ValueError('invalid entry in index array')

    if order == 'C':
        unraveled_coords = reversed(unraveled_coords)
    return tuple(unraveled_coords)


# TODO(okuta): Implement diag_indices


# TODO(okuta): Implement diag_indices_from


# TODO(okuta): Implement mask_indices


# TODO(okuta): Implement tril_indices


# TODO(okuta): Implement tril_indices_from


# TODO(okuta): Implement triu_indices


# TODO(okuta): Implement triu_indices_from

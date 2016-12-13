# flake8: NOQA
# "flake8: NOQA" to suppress warning "H104  File contains nothing but comments"


# class c_(object):
# class r_(object):
# class s_(object):

import numpy
import cupy
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
        new = cupy.asarray(new)
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

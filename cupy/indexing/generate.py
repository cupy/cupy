# flake8: NOQA
# "flake8: NOQA" to suppress warning "H104  File contains nothing but comments"

# class s_(object):

import numpy

import cupy

import six


class AxisConcatenator(object):
    """Translates slice objects to concatenation along an axis.

    For detailed documentation on usage, see `r_`.

    """

    def _output_obj(self, newobj, tempobj, ndmin, trans1d):
        k2 = ndmin - tempobj.ndim
        if (trans1d < 0):
            trans1d += k2 + 1
        defaxes = list(six.moves.range(ndmin))
        k1 = trans1d
        axes = defaxes[:k1] + defaxes[k2:] + \
            defaxes[k1:k2]
        newobj = newobj.transpose(axes)
        return newobj

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

        for k in six.moves.range(len(key)):
            scalar = False
            if isinstance(key[k], slice):
                raise NotImplementedError
            elif isinstance(key[k], six.string_types):
                if k != 0:
                    raise ValueError(
                        'special directives must be the first entry.')
                raise NotImplementedError
            elif type(key[k]) in numpy.ScalarType:
                newobj = cupy.array(key[k], ndmin=ndmin)
                scalars.append(k)
                scalar = True
                scalartypes.append(newobj.dtype)
            else:
                newobj = key[k]
                tempobj = cupy.array(newobj, copy=False)
                newobj = cupy.array(newobj, copy=False, ndmin=ndmin)
                if ndmin > 1:
                    if trans1d != -1 and tempobj.ndim < ndmin:
                        newobj = self._output_obj(newobj, ndmin, trans1d)
                    del tempobj
                elif ndmin == 1:
                    if tempobj.ndim < ndmin:
                        newobj = self._output_obj(
                            newobj, tempobj, ndmin, trans1d)
                    del tempobj

            objs.append(newobj)
            if not scalar and isinstance(newobj, cupy.ndarray):
                arraytypes.append(newobj.dtype)

        final_dtype = numpy.find_common_type(arraytypes, scalartypes)
        if final_dtype is not None:
            for k in scalars:
                objs[k] = objs[k].astype(final_dtype)

        return cupy.concatenate(tuple(objs), axis=self.axis)

    def __len__(self):
        return 0


class CClass(AxisConcatenator):

    def __init__(self):
        """Translates slice objects to concatenation along the second axis.

        This is CuPy object that corresponds to np.r_['-1,2,0', index expression], which is
        useful because of its common occurrence. In particular, arrays will be
        stacked along their last axis after being upgraded to at least 2-D with
        1's post-pended to the shape (column vectors made out of 1-D arrays).

        For detailed documentation, see `r_`.
        """
        AxisConcatenator.__init__(self, -1, ndmin=2, trans1d=0)

c_ = CClass()


class RClass(AxisConcatenator):

    def __init__(self):
        """Translates slice objects to concatenation along the first axis.

        This is a simple way to build up arrays quickly. There are two use cases.

        1. If the index expression contains comma separated arrays, then stack
           them along their first axis.
        2. If the index expression contains slice notation or scalars then create
           a 1-D array with a range indicated by the slice notation. (Not Implemented)

        If slice notation is used, the syntax ``start:stop:step`` is equivalent
        to ``np.arange(start, stop, step)`` inside of the brackets. However, if
        ``step`` is an imaginary number (i.e. 100j) then its integer portion is
        interpreted as a number-of-points desired and the start and stop are
        inclusive. In other words ``start:stop:stepj`` is interpreted as
        ``np.linspace(start, stop, step, endpoint=1)`` inside of the brackets.
        After expansion of slice notation, all comma separated sequences are
        concatenated together. (Not Implemented)

        Optional character strings placed as the first element of the index
        expression can be used to change the output. The strings 'r' or 'c' result
        in matrix output. If the result is 1-D and 'r' is specified a 1 x N (row)
        matrix is produced. If the result is 1-D and 'c' is specified, then a N x 1
        (column) matrix is produced. If the result is 2-D then both provide the
        same matrix result. (Not Implemented)

        A string integer specifies which axis to stack multiple comma separated
        arrays along. A string of two comma-separated integers allows indication
        of the minimum number of dimensions to force each entry into as the
        second integer (the axis to concatenate along is still the first integer).
        (Not Implemented)

        A string with three comma-separated integers allows specification of the
        axis to concatenate along, the minimum number of dimensions to force the
        entries to, and which axis should contain the start of the arrays which
        are less than the specified number of dimensions. In other words the third
        integer allows you to specify where the 1's should be placed in the shape
        of the arrays that have their shapes upgraded. By default, they are placed
        in the front of the shape tuple. The third argument allows you to specify
        where the start of the array should be instead. Thus, a third argument of
        '0' would place the 1's at the end of the array shape. Negative integers
        specify where in the new shape tuple the last dimension of upgraded arrays
        should be placed, so the default is '-1'. (Not Implemented)

        Args:
            Not a function, so takes no parameters

        Returns:
            cupy.ndarray: Joined array.

        .. seealso:: :func:`numpy.r_`

        """
        AxisConcatenator.__init__(self)

r_ = RClass()

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

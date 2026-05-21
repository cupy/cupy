from __future__ import annotations

import cupy
import operator
import numpy

from cupy._core._dtype import get_dtype

supported_dtypes = [get_dtype(x) for x in
                    ('single', 'double', 'csingle', 'cdouble')]

# dtype kind chars that cuSPARSE accepts for sparse ``data`` arrays:
#   '?' bool, 'f' float32, 'd' float64, 'F' complex64, 'D' complex128.
_SPARSE_DATA_KINDS = frozenset('?fdFD')


def is_sparse_data_dtype(dtype):
    """Return True if ``dtype`` can be stored in a sparse ``data`` array.

    cuSPARSE-backed ops accept only bool, float32, float64, complex64,
    and complex128 for ``data``.
    """
    return numpy.dtype(dtype).char in _SPARSE_DATA_KINDS


_upcast_memo: dict = {}


def isdense(x):
    return isinstance(x, cupy.ndarray)


def isscalarlike(x):
    """Is x either a scalar, an array scalar, or a 0-dim array?"""
    return cupy.isscalar(x) or (isdense(x) and x.ndim == 0)


def safely_cast_index_arrays(A, idx_dtype=numpy.int32, msg=""):
    """Safely cast sparse array indices to ``idx_dtype``.

    Check the shape of *A* to determine if it is safe to cast its index
    arrays to dtype *idx_dtype*.  If any dimension in shape is larger than
    fits in the dtype, casting is unsafe so raise :class:`ValueError`.
    If safe, cast the index arrays to ``idx_dtype`` and return the result
    without changing the input *A*.  The caller can assign the results to
    *A*'s attributes if desired or use the recast index arrays directly.

    Unless downcasting is needed, the original index arrays are returned.
    You can test e.g. ``A.indptr is new_indptr`` to see if downcasting
    occurred.

    Args:
        A (cupyx.scipy.sparse): The array for which index arrays should
            be (potentially) downcast.
        idx_dtype (dtype): Desired index dtype.  Defaults to ``numpy.int32``.
        msg (str, optional): String appended to the ``ValueError`` message
            when ``A.shape`` is too big to fit in ``idx_dtype``.

    Returns:
        ndarray or tuple of ndarrays:
            For CSR/CSC, ``(indices, indptr)``.
            For COO, ``(row, col)`` (CuPy is currently 2-D-only).
            For DIA, ``offsets``.

    Raises:
        ValueError: When the dtype cannot represent ``A``'s shape or
            existing index values.

    .. seealso:: :func:`scipy.sparse.safely_cast_index_arrays`
    """
    idx_dtype = numpy.dtype(idx_dtype)
    if not msg:
        msg = f"dtype {idx_dtype}"
    max_value = numpy.iinfo(idx_dtype).max

    if A.format in ('csc', 'csr'):
        # indptr is monotonically nondecreasing, so its last element is
        # the largest representable value.
        if int(A.indptr[-1]) > max_value:  # synchronize!
            raise ValueError(f"indptr values too large for {msg}")
        if max(A.shape) > max_value:
            if bool((A.indices > max_value).any()):  # synchronize!
                raise ValueError(f"indices values too large for {msg}")
        return (A.indices.astype(idx_dtype, copy=False),
                A.indptr.astype(idx_dtype, copy=False))

    if A.format == 'coo':
        if max(A.shape) > max_value:
            if (bool((A.row > max_value).any())  # synchronize!
                    or bool((A.col > max_value).any())):
                raise ValueError(f"coords values too large for {msg}")
        return (A.row.astype(idx_dtype, copy=False),
                A.col.astype(idx_dtype, copy=False))

    if A.format == 'dia':
        if max(A.shape) > max_value:
            if bool((A.offsets > max_value).any()):  # synchronize!
                raise ValueError(f"offsets values too large for {msg}")
        return A.offsets.astype(idx_dtype, copy=False)

    raise TypeError(
        f'Format {A.format} is not associated with index arrays.')


def get_index_dtype(arrays=(), maxval=None, check_contents=False):
    """Based on input (integer) arrays ``a``, determines a suitable index data
    type that can hold the data in the arrays.

    Args:
        arrays (tuple of array_like):
            Input arrays whose types/contents to check
        maxval (float, optional):
            Maximum value needed
        check_contents (bool, optional):
            Whether to check the values in the arrays and not just their types.
            Default: False (check only the types)

    Returns:
        dtype: Suitable index data type (int32 or int64)
    """

    int32min = cupy.iinfo(cupy.int32).min
    int32max = cupy.iinfo(cupy.int32).max

    dtype = cupy.int32
    if maxval is not None:
        if maxval > int32max:
            dtype = cupy.int64

    if isinstance(arrays, cupy.ndarray):
        arrays = (arrays,)

    for arr in arrays:
        arr = cupy.asarray(arr)
        if not cupy.can_cast(arr.dtype, cupy.int32):
            if check_contents:
                if arr.size == 0:
                    # a bigger type not needed
                    continue
                elif cupy.issubdtype(arr.dtype, cupy.integer):
                    maxval = arr.max()
                    minval = arr.min()
                    if minval >= int32min and maxval <= int32max:
                        # a bigger type not needed
                        continue

            dtype = cupy.int64
            break

    return dtype


def validateaxis(axis):
    if axis is not None:
        # In NumPy, you can pass in tuples for 'axis', but they are
        # not very useful for sparse matrices given their limited
        # dimensions, so let's make it explicit that they are not
        # allowed to be passed in
        if isinstance(axis, tuple):
            raise TypeError("Tuples are not accepted for the 'axis' "
                            "parameter. Please pass in one of the "
                            "following: {-2, -1, 0, 1, None}.")

        axis_type = type(axis)

        # If not a tuple, check that the provided axis is actually
        # an integer and raise a TypeError similar to NumPy's
        if not cupy.issubdtype(cupy.dtype(axis_type), cupy.integer):
            raise TypeError("axis must be an integer, not {name}"
                            .format(name=axis_type.__name__))

        if not (-2 <= axis <= 1):
            raise ValueError("axis out of range")


def upcast(*args):
    """Returns the nearest supported sparse dtype for the
    combination of one or more types.

    upcast(t0, t1, ..., tn) -> T  where T is a supported dtype

    Examples:
        >>> upcast('int32')
        <type 'numpy.int32'>
        >>> upcast('int32','float32')
        <type 'numpy.float64'>
        >>> upcast('bool',float)
        <type 'numpy.complex128'>
    """

    t = _upcast_memo.get(args)
    if t is not None:
        return t

    upcast = numpy.result_type(*args)

    for t in supported_dtypes:
        if cupy.can_cast(upcast, t):
            _upcast_memo[args] = t
            return t

    raise TypeError('no supported conversion for types: %r' % (args,))


def check_shape(args, current_shape=None):
    """Check validity of the shape"""

    if len(args) == 0:
        raise TypeError("function missing 1 required positional argument: "
                        "'shape'")

    elif len(args) == 1:
        try:
            shape_iter = iter(args[0])
        except TypeError:
            new_shape = (operator.index(args[0]), )
        else:
            new_shape = tuple(operator.index(arg) for arg in shape_iter)
    else:
        new_shape = tuple(operator.index(arg) for arg in args)

    if current_shape is None:
        if len(new_shape) != 2:
            raise ValueError('shape must be a 2-tuple of positive integers')
        elif new_shape[0] < 0 or new_shape[1] < 0:
            raise ValueError("'shape' elements cannot be negative")

    else:
        current_size = numpy.prod(current_shape)

        negative_indexes = [i for i, x in enumerate(new_shape) if x < 0]
        if len(negative_indexes) == 0:
            new_size = numpy.prod(new_shape)
            if new_size != current_size:
                raise ValueError('cannot reshape array of size {} into shape'
                                 '{}'.format(current_size, new_shape))
        elif len(negative_indexes) == 1:
            skip = negative_indexes[0]
            specified = numpy.prod(new_shape[0:skip] + new_shape[skip+1:])
            unspecified, remainder = divmod(current_size, specified)
            if remainder != 0:
                err_shape = tuple('newshape'if x < 0 else x for x in new_shape)
                raise ValueError('cannot reshape array of size {} into shape'
                                 '{}'.format(current_size, err_shape))
            new_shape = new_shape[0:skip] + (unspecified,) + new_shape[skip+1:]
        else:
            raise ValueError('can only specify one unknown dimension')

    if len(new_shape) != 2:
        raise ValueError('matrix shape must be two-dimensional')

    return new_shape

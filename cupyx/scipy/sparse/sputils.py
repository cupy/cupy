import cupy

from cupy.core._dtype import get_dtype

supported_dtypes = [get_dtype(x) for x in
                    ('single', 'double', 'csingle', 'cdouble')]

_upcast_memo = {}


def isdense(x):
    return isinstance(x, cupy.ndarray)


def isscalarlike(x):
    """Is x either a scalar, an array scalar, or a 0-dim array?"""
    return cupy.isscalar(x) or (isdense(x) and x.ndim == 0)


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

    dtype = cupy.intc
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

    upcast = cupy.find_common_type(args, [])

    for t in supported_dtypes:
        if cupy.can_cast(upcast, t):
            _upcast_memo[args] = t
            return t

    raise TypeError('no supported conversion for types: %r' % (args,))

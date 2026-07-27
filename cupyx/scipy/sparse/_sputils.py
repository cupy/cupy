from __future__ import annotations

import cupy
import operator
import numpy

from cupy._core._dtype import get_dtype

# Extra floating dtypes beyond numpy's own that CuPy can store and compute
# on.  Today this is only ``ml_dtypes.bfloat16`` (an optional dependency):
# CuPy has no float8 support.  These have no scipy.sparse equivalent (scipy
# rejects them), so their semantics follow CuPy's dense arrays.  Detected by
# identity, not dtype ``kind`` (bfloat16's kind is the opaque ``'V'``).
#
# CuPy only registers the bfloat16 kernel typename when ml_dtypes is present
# AND numpy >= 2.1.2; on older numpy the bfloat16 loops are silently a no-op
# (see cupy/_core/_scalar.pyx, dlpack.pyx and _util.pyx).  Match that guard --
# without the version check the gate would admit bfloat16 sparse data whose
# kernels do not exist, and every operation on it would miscompile.
if numpy.lib.NumpyVersion(numpy.__version__) >= '2.1.2':
    try:
        import ml_dtypes as _ml_dtypes
        _extra_float_dtypes = frozenset([numpy.dtype(_ml_dtypes.bfloat16)])
    except ImportError:
        _extra_float_dtypes = frozenset()
else:
    _extra_float_dtypes = frozenset()

# Dtypes ``upcast`` may promote to, widest last.  Mirrors scipy minus the
# extended-precision types the GPU cannot store (longdouble/clongdouble):
# without the integer entries ``upcast`` would promote every int/bool
# combination to float (e.g. hstack/bmat/kronsum of int blocks -> float32).
# The 16-bit floats (float16, then bfloat16 if available) sit just before
# float32 so a same-dtype upcast stays 16-bit rather than widening.
supported_dtypes = (
    [get_dtype(x) for x in ('bool_', 'int8', 'uint8', 'int16', 'uint16',
                            'int32', 'uint32', 'int64', 'uint64')]
    + [get_dtype('float16')]
    + sorted(_extra_float_dtypes, key=lambda d: d.itemsize)
    + [get_dtype(x) for x in ('single', 'double', 'csingle', 'cdouble')])


def is_extra_float_dtype(dtype):
    """Whether ``dtype`` is a supported non-numpy float (e.g. bfloat16)."""
    return numpy.dtype(dtype) in _extra_float_dtypes


def is_16bit_float(dtype):
    """Whether ``dtype`` is a 16-bit float (float16 or bfloat16).

    These widen losslessly to float32, which the pure-CuPy fallbacks use for
    accumulation and for the value-templated kernels: ``add.at``/``atomicAdd``
    and the kernel instantiations do not cover the 16-bit floats uniformly.
    """
    dtype = numpy.dtype(dtype)
    return dtype in _extra_float_dtypes or (
        dtype.kind == 'f' and dtype.itemsize == 2)


def is_float_dtype(dtype):
    """True for real floating-point dtypes, *including* bfloat16.

    A bare ``dtype.kind == 'f'`` test excludes bfloat16 (whose kind is the
    opaque ``'V'``); use this where a float-vs-non-float decision must treat
    bfloat16 as the float it is (e.g. ``asfptype``).
    """
    return numpy.dtype(dtype).kind == 'f' or is_extra_float_dtype(dtype)


def promote_data_types(*dtypes):
    """``numpy.promote_types`` reduction with a dense-parity fallback.

    bfloat16 mixed with a >= 16-bit int (or with float16) has no numpy
    abstract promotion -- ml_dtypes never registered it -- so
    ``numpy.promote_types`` raises ``DTypePromotionError``.  Dense CuPy still
    resolves the pair through its ufunc loops (e.g. ``bfloat16 + int32 ->
    float64``); reproduce that by promoting through an actual (empty) add so
    the elementwise sparse ops (add, multiply, maximum, comparison) match
    dense.  Only the rare unpromotable mix reaches the fallback -- every other
    combination takes the plain ``promote_types`` fast path.
    """
    try:
        result = numpy.dtype(dtypes[0])
        for dt in dtypes[1:]:
            result = numpy.promote_types(result, dt)
        return result
    except TypeError:  # numpy's DTypePromotionError subclasses TypeError
        acc = cupy.empty(0, numpy.dtype(dtypes[0]))
        for dt in dtypes[1:]:
            acc = acc + cupy.empty(0, numpy.dtype(dt))
        return acc.dtype


def promote_scalar_data_type(sparse_dtype, scalar):
    """Dtype of an elementwise sparse-data op against a *scalar*, like dense.

    For every dtype but the extra floats (bfloat16) this is value-based
    ``numpy.result_type`` -- so a Python ``int``/``float`` promotes weakly.
    ``numpy.result_type`` mishandles bfloat16, though: it *raises* for a typed
    numpy scalar (e.g. ``numpy.int16(1)`` has no abstract promotion) and
    *over-promotes* a Python float (``result_type(bfloat16, 2.0)`` -> float64
    where the ufunc gives float32).  Reproduce dense CuPy's actual promotion
    with a zero-length add so scalar ``maximum``/``minimum``/``*``/``*=`` on a
    bfloat16 matrix match dense instead of raising or widening.
    ``maximum``/``minimum``/``multiply`` share ``add``'s operand promotion;
    division forces a float through its own path.
    """
    sparse_dtype = numpy.dtype(sparse_dtype)
    if not is_extra_float_dtype(sparse_dtype):
        return numpy.result_type(sparse_dtype, scalar)
    return (cupy.empty(0, sparse_dtype) + scalar).dtype


def get_sum_dtype(dtype):
    """The dtype ``sum`` accumulates in, mirroring numpy/scipy.

    The counterpart of ``scipy.sparse._sputils.get_sum_dtype``: bool and
    signed integers widen to the platform int (int64 on the 64-bit builds
    CuPy targets), unsigned integers to the platform uint; float and complex
    are unchanged.  Used by the integer axis reductions, which accumulate in
    an explicit wide type.
    """
    dtype = numpy.dtype(dtype)
    if dtype.kind == 'u' and numpy.can_cast(dtype, numpy.uint):
        return numpy.dtype(numpy.uint)
    if numpy.can_cast(dtype, numpy.int_):
        return numpy.dtype(numpy.int_)
    return dtype


def add_at_accumulator_dtype(dtype):
    """Return an ``add.at``-compatible accumulator dtype for ``dtype``.

    ``cupy.add.at`` targets only int32/int64/uint32/uint64 and
    float16/32/64.  bool and narrow signed integers widen to int64 (bool via
    "any nonzero"); narrow unsigned integers to uint64 (exact; the cast back
    wraps like numpy); the 16-bit floats accumulate in float32 and round back
    on the cast (lossless widening).  Every other (already-accepted) dtype is
    returned unchanged, so the result is always a valid ``add.at`` target.
    """
    dtype = numpy.dtype(dtype)
    if dtype.kind == 'b' or (dtype.kind == 'i' and dtype.itemsize < 4):
        return numpy.dtype(numpy.int64)
    if dtype.kind == 'u' and dtype.itemsize < 4:
        return numpy.dtype(numpy.uint64)
    if is_16bit_float(dtype):
        return numpy.dtype(numpy.float32)
    return dtype


def is_sparse_data_dtype(dtype):
    """Return True if ``dtype`` can be stored in a sparse ``data`` array.

    Accepts every dtype CuPy dense arrays support: bool, every signed/
    unsigned integer width, float16/32/64, complex64/128, and (if installed)
    bfloat16.  cuSPARSE itself accepts only float32/64 and complex64/128;
    bool, integers and the 16-bit floats route through pure-CuPy fallbacks.
    This is a superset of scipy (which rejects float16 and bfloat16).
    Rejected: extended precision (longdouble/clongdouble) and float8, which
    the GPU cannot represent.  numpy pins the ``char`` of every fixed-width
    float/complex ('e'/'f'/'d' and 'F'/'D') and gives longdouble its own
    'g'/'G', so the char is the platform-independent test here -- the
    *width* is not (MSVC's ``long double`` is 8 bytes, so longdouble would
    pass an itemsize check).
    """
    dtype = numpy.dtype(dtype)
    if dtype in _extra_float_dtypes:
        return True
    if dtype.kind in 'biu':          # bool, signed int, unsigned int
        return True
    if dtype.kind == 'f':            # float16/32/64 (not float8/longdouble)
        return dtype.char in 'efd'
    if dtype.kind == 'c':            # complex64/complex128 (not clongdouble)
        return dtype.char in 'FD'
    return False


def check_data_dtype(dtype):
    """Raise ``ValueError`` if ``dtype`` cannot back a sparse ``data`` array.

    Shared by the constructors and :meth:`astype` so an unsupported dtype
    (float8 or extended precision the GPU cannot store) is rejected the same
    way everywhere.  The supported list is built from ``supported_dtypes``,
    so bfloat16 appears only when ``ml_dtypes`` is installed, and the wording
    mirrors scipy's ``getdtype`` rejection.
    """
    dtype = numpy.dtype(dtype)
    if not is_sparse_data_dtype(dtype):
        names = ', '.join(d.name for d in supported_dtypes)
        raise ValueError(
            f'cupyx.scipy.sparse does not support dtype {dtype.name}. '
            f'The only supported types are: {names}.')


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


def validate_axis_1d(axis):
    """Validate a reduction ``axis`` for a 1-D sparse array.

    Accepts ``None``, ``0``, ``-1``, or a length-1 tuple of ``0``/``-1``
    (scipy allows a 1-tuple axis).  Raises otherwise.  A 1-D reduction
    always collapses the single axis, so callers ignore the (absent)
    return value and reduce over everything.
    """
    if isinstance(axis, tuple):
        if len(axis) != 1:
            raise ValueError('axis out of range for 1-D array')
        # A tuple axis must hold an integer axis: numpy/scipy reject a
        # non-integer element such as ``(None,)`` (mirrors collapse_2d_axis).
        axis = operator.index(axis[0])
    if axis not in (None, 0, -1):
        raise ValueError(f'axis {axis} is out of bounds for 1-D array')


def collapse_2d_axis(axis):
    """Collapse a tuple ``axis`` for a 2-D reduction to a plain int / None.

    scipy accepts tuple axes for 2-D array/matrix reductions: a length-1
    tuple ``(i,)`` means axis ``i``, and a length-2 tuple spanning both
    axes means a full reduction (``None``).  A non-tuple ``axis`` is
    returned unchanged for the caller's normal validation.
    """
    if not isinstance(axis, tuple):
        return axis
    if len(axis) == 1:
        # Validate the element like the length-2 branch: ``operator.index``
        # rejects non-integers (e.g. ``(None,)`` / ``(0.0,)``) that would
        # otherwise slip through as a bogus axis, matching numpy/scipy.
        return operator.index(axis[0])
    if len(axis) == 2:
        norm = set()
        for a in axis:
            a = operator.index(a)  # rejects non-integers like numpy
            norm.add(a + 2 if a < 0 else a)
        if norm == {0, 1}:
            return None
    raise ValueError('axis out of range for 2-D reduction')


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


def check_shape(args, current_shape=None, *, allow_nd=(2,)):
    """Check validity of the shape.

    Args:
        allow_nd (tuple of int): Accepted dimensionalities (tuple
            lengths).  Defaults to ``(2,)`` so 2-D-only callers are
            unaffected; pass ``(1, 2)`` to also accept 1-D shapes.
    """

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
        if len(new_shape) not in allow_nd:
            raise ValueError(f'shape must have length in {allow_nd}. '
                             f'Got new_shape={new_shape}')
        elif any(d < 0 for d in new_shape):
            raise ValueError("'shape' elements cannot be negative")

    else:
        current_size = numpy.prod(current_shape)

        negative_indexes = [i for i, x in enumerate(new_shape) if x < 0]
        if len(negative_indexes) == 0:
            new_size = numpy.prod(new_shape)
            if new_size != current_size:
                raise ValueError('cannot reshape array of size {} into shape'
                                 ' {}'.format(current_size, new_shape))
        elif len(negative_indexes) == 1:
            skip = negative_indexes[0]
            specified = numpy.prod(new_shape[0:skip] + new_shape[skip+1:])
            unspecified, remainder = divmod(current_size, specified)
            if remainder != 0:
                err_shape = tuple('newshape'if x < 0 else x for x in new_shape)
                raise ValueError('cannot reshape array of size {} into shape'
                                 ' {}'.format(current_size, err_shape))
            new_shape = (new_shape[0:skip] + (int(unspecified),)
                         + new_shape[skip+1:])
        else:
            raise ValueError('can only specify one unknown dimension')

    if len(new_shape) not in allow_nd:
        raise ValueError(f'shape must have length in {allow_nd}. '
                         f'Got new_shape={new_shape}')

    return new_shape

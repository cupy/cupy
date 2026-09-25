from ._kernel import create_ufunc

# This is only line changed from upstream cupy file.
# The prefix was previously inverted (the Ascend build used "cupy_", so no
# ufunc created here ever dispatched). Keep "cupy_" for the normal backends —
# the Ascend dispatcher rewrites `cupy_<name>` to `ascend_<name>` itself.
IF CUPY_CANN_VERSION > 0:
    cdef str OP_PREFIX = "ascend_"
ELSE:
    cdef str OP_PREFIX = "cupy_"


cdef _create_bit_op(name, op, no_bool, doc='', scatter_op=None):
    types = () if no_bool else ('??->?',)
    return create_ufunc(
        OP_PREFIX + name,
        types + ('bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l',
                 'LL->L', 'qq->q', 'QQ->Q'),
        'out0 = in0 %s in1' % op,
        doc=doc, scatter_op=scatter_op)


cdef _bitwise_and = _create_bit_op(
    'bitwise_and', '&', False,
    '''Computes the bitwise AND of two arrays elementwise.

    Only integer and boolean arrays are handled.

    .. seealso:: :data:`numpy.bitwise_and`

    ''',
    scatter_op='and')


cdef _bitwise_or = _create_bit_op(
    'bitwise_or', '|', False,
    '''Computes the bitwise OR of two arrays elementwise.

    Only integer and boolean arrays are handled.

    .. seealso:: :data:`numpy.bitwise_or`

    ''',
    scatter_op='or')

cdef _bitwise_xor = _create_bit_op(
    'bitwise_xor', '^', False,
    '''Computes the bitwise XOR of two arrays elementwise.

    Only integer and boolean arrays are handled.

    .. seealso:: :data:`numpy.bitwise_xor`

    ''',
    scatter_op='xor')


cdef _invert = create_ufunc(
    'cupy_invert',
    (('?->?', 'out0 = !in0'), 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I',
     'l->l', 'L->L', 'q->q', 'Q->Q'),
    'out0 = ~in0',
    doc='''Computes the bitwise NOT of an array elementwise.

    Only integer and boolean arrays are handled.

    .. note::
        :func:`cupy.bitwise_not` is an alias for :func:`cupy.invert`.

    .. seealso:: :data:`numpy.invert`

    ''')


cdef _left_shift = _create_bit_op(
    'left_shift', '<<', True,
    '''Shifts the bits of each integer element to the left.

    Only integer arrays are handled.

    .. seealso:: :data:`numpy.left_shift`

    ''')


cdef _right_shift = _create_bit_op(
    'right_shift', '>>', True,
    '''Shifts the bits of each integer element to the right.

    Only integer arrays are handled

    .. seealso:: :data:`numpy.right_shift`

    ''')


# Variables to expose to Python
# (cythonized data cannot be exposed to Python, even with cpdef.)
bitwise_and = _bitwise_and
bitwise_or = _bitwise_or
bitwise_xor = _bitwise_xor
invert = _invert

IF CUPY_CANN_VERSION > 0:
    # ASCEND: `ascend_left_shift` / `ascend_right_shift` are registered as
    # BINARY_OP (tensor @ tensor) only -- CANN has no scalar shift operator,
    # so a call with a Python scalar operand makes the dispatcher look up
    # SCALAR_BINARY_OP and fail with KeyError. The adapters below normalize
    # scalar operands to 0-d arrays so the call always takes the registered
    # tensor-tensor path.
    #
    # NB: `import cupy` binds the partially initialized package here (this
    # module is imported from `cupy/__init__`); that is fine because the
    # `_cp` attributes are only accessed at call time.
    import cupy as _cp

    def _shift_adapter(ufunc, x1, x2, args, kwargs):
        # Follow NumPy's weak-scalar rule: a *Python* scalar takes the dtype
        # of the other operand (np.left_shift(int32_arr, 2) stays int32);
        # numpy scalars keep their own dtype via the normal promotion path.
        if not isinstance(x1, _cp.ndarray):
            if isinstance(x2, _cp.ndarray) and isinstance(x1, int):
                x1 = _cp.asarray(x1, dtype=x2.dtype)
            else:
                x1 = _cp.asarray(x1)
        if not isinstance(x2, _cp.ndarray):
            if isinstance(x1, _cp.ndarray) and isinstance(x2, int):
                x2 = _cp.asarray(x2, dtype=x1.dtype)
            else:
                x2 = _cp.asarray(x2)
        return ufunc(x1, x2, *args, **kwargs)

    def left_shift(x1, x2, *args, **kwargs):
        """Shifts the bits of each integer element to the left.

        Only integer arrays are handled.

        .. seealso:: :data:`numpy.left_shift`
        """
        return _shift_adapter(_left_shift, x1, x2, args, kwargs)

    def right_shift(x1, x2, *args, **kwargs):
        """Shifts the bits of each integer element to the right.

        Only integer arrays are handled.

        .. seealso:: :data:`numpy.right_shift`
        """
        return _shift_adapter(_right_shift, x1, x2, args, kwargs)

ELSE:
    left_shift = _left_shift
    right_shift = _right_shift

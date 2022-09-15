import cupy


def ufunc_at(ufunc, a, indices, b=None):
    from cupy._core import _routines_indexing
    from cupy._core import _kernel

    a = cupy.asarray(a)
    indices = cupy.asarray(indices)

    if ufunc.nout != 1:
        raise TypeError('TODO')

    if ufunc.nin == 1:
        if b is not None:
            raise ValueError('TODO')
        in_types = (a.dtype.type,)
        create_kernel = _routines_indexing._create_unary_scatter_kernel
    elif ufunc.nin == 2:
        if b is None:
            raise ValueError('TODO')
        b = cupy.asarray(b, dtype=a.dtype)
        in_types = (a.dtype.type, b.dtype.type)
        create_kernel = _routines_indexing._create_binary_scatter_kernel
    else:
        raise ValueError('TODO')

    op = ufunc._ops._guess_routine_from_in_types(in_types)
    preamble = ufunc._preamble + '\n' + _kernel._TypeMap(
        tuple([(f'in{i}_type', t) for i, t in enumerate(op.in_types)] +
              [(f'out{i}_type', t) for i, t in enumerate(op.out_types)])
    ).get_typedef_code()
    kernel = create_kernel('scatter_' + ufunc.name, op.routine, preamble)
    _routines_indexing._call_scatter_op_single(kernel, a, indices, b, 0, 1)

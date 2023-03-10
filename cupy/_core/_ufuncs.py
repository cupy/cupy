import numpy as np
from cupy._core._kernel import create_ufunc, _Op


def _fix_to_sctype(dtype, sctype):
    if dtype.type == sctype:
        return dtype
    elif dtype.kind == "S":
        length = dtype.itemsize
    elif dtype.kind == "U":
        length = dtype.itemsize // 4
    else:
        raise ValueError("CuPy currently only supports string conversions.")

    return np.dtype((sctype, length))


def _s_copy_resolver(op, in_dtypes, out_dtypes):
    # The result must have the same length as the input (it may be cast
    # after the copy no-op)
    return in_dtypes, in_dtypes


elementwise_copy = create_ufunc(
    'cupy_copy',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d', 'F->F', 'D->D'),
    'out0 = in0',
    default_casting='unsafe', custom_ops=[
        # It should be noted that casting isn't actually handled here, rather
        # the inner-loop of "copy" is effectively a no-op.
        _Op((np.bytes_,), (np.bytes_,), "out0 = in0", None, _s_copy_resolver),
        _Op((np.str_,), (np.str_,), "out0 = in0", None, _s_copy_resolver),
    ])

from cupy import _core


def count_nonzero(a, axis=None):
    """Counts the number of non-zero values in the array.

    .. note::

       :func:`numpy.count_nonzero` returns `int` value when `axis=None`,
       but :func:`cupy.count_nonzero` returns zero-dimensional array to reduce
       CPU-GPU synchronization.

    Args:
        a (cupy.ndarray): The array for which to count non-zeros.
        axis (int or tuple, optional): Axis or tuple of axes along which to
            count non-zeros. Default is None, meaning that non-zeros will be
            counted along a flattened version of ``a``
    Returns:
        cupy.ndarray of int: Number of non-zero values in the array
            along a given axis. Otherwise, the total number of non-zero values
            in the array is returned.
    """

    return _count_nonzero(a, axis=axis)


_count_nonzero = _core.create_reduction_func(
    'cupy_count_nonzero',
    ('?->l', 'B->l', 'h->l', 'H->l', 'i->l', 'I->l', 'l->l', 'L->l',
     'q->l', 'Q->l', 'e->l', 'f->l', 'd->l', 'F->l', 'D->l'),
    ('in0 != type_in0_raw(0)', 'a + b', 'out0 = a', None), 0)

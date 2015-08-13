from cupy import reduction


def sum(a, axis=None, dtype=None, out=None, keepdims=False, allocator=None):
    '''Returns the sum of an array along given axes.

    Args:
        a (cupy.ndarray): Array to take sum.
        axis (int or sequence of ints): Axes along which the sum is taken.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If True, the specified axes are remained as axes of
            length one.
        allocator (function): CuPy memory allocator. The allocator of ``a`` is
            used by default.

    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.sum`

    '''
    return _sum(a, axis, dtype, out, keepdims, allocator)


def prod(a, axis=None, dtype=None, out=None, keepdims=False, allocator=None):
    '''Returns the product of an array along given axes.

    Args:
        a (cupy.ndarray): Array to take product.
        axis (int or sequence of ints): Axes along which the product is taken.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If True, the specified axes are remained as axes of
            length one.
        allocator (function): CuPy memory allocator. The allocator of ``a`` is
            used by default.

    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.prod`

    '''
    return _prod(a, axis, dtype, out, keepdims, allocator)


def nansum(a, axis=None, dtype=None, out=None, keepdims=0, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def cumprod(a, axis=None, dtype=None, out=None, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def cumsum(a, axis=None, dtype=None, out=None, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def diff(a, n=1, axis=-1, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def ediff1d(ary, to_end=None, to_begin=None, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def gradient(f, *varargs, **kwargs):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def trapz(y, x=None, dx=1.0, axis=-1):
    # TODO(beam2d): Implement it
    raise NotImplementedError


_sum = reduction.create_reduction_func(
    'cupy_sum',
    ['?->l', 'B->L', 'h->l', 'H->L', 'i->l', 'I->L', 'l->l', 'L->L',
     'q->q', 'Q->Q',
     ('e->e', (None, None, None, 'float')),
     'f->f', 'd->d'],
    ('in0', 'a + b', 'out0 = a', None), 0)


_prod = reduction.create_reduction_func(
    'cupy_prod',
    ['?->l', 'B->L', 'h->l', 'H->L', 'i->l', 'I->L', 'l->l', 'L->L',
     'q->q', 'Q->Q',
     ('e->e', (None, None, None, 'float')),
     'f->f', 'd->d'],
    ('in0', 'a * b', 'out0 = a', None), 1)

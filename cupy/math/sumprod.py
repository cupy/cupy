from cupy import reduction


def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    '''Returns the sum of an array along given axes.

    Args:
        a (cupy.ndarray): Array to take sum.
        axis (int or sequence of ints): Axes along which the sum is taken.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If True, the specified axes are remained as axes of
            length one.

    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.sum`

    '''
    return _sum(a, axis, dtype, out, keepdims)


def prod(a, axis=None, dtype=None, out=None, keepdims=False):
    '''Returns the product of an array along given axes.

    Args:
        a (cupy.ndarray): Array to take product.
        axis (int or sequence of ints): Axes along which the product is taken.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If True, the specified axes are remained as axes of
            length one.

    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.prod`

    '''
    return _prod(a, axis, dtype, out, keepdims)


# TODO(okuta): Implement nansum


# TODO(okuta): Implement cumprod


# TODO(okuta): Implement cumsum


# TODO(okuta): Implement diff


# TODO(okuta): Implement ediff1d


# TODO(okuta): Implement gradient


# TODO(okuta): Implement cross


# TODO(okuta): Implement trapz


_sum = reduction.create_reduction_func(
    'cupy_sum',
    ('?->l', 'B->L', 'h->l', 'H->L', 'i->l', 'I->L', 'l->l', 'L->L',
     'q->q', 'Q->Q',
     ('e->e', (None, None, None, 'float')),
     'f->f', 'd->d'),
    ('in0', 'a + b', 'out0 = a', None), 0)


_prod = reduction.create_reduction_func(
    'cupy_prod',
    ['?->l', 'B->L', 'h->l', 'H->L', 'i->l', 'I->L', 'l->l', 'L->L',
     'q->q', 'Q->Q',
     ('e->e', (None, None, None, 'float')),
     'f->f', 'd->d'],
    ('in0', 'a * b', 'out0 = a', None), 1)

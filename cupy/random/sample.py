from cupy import cuda
from cupy.random import distributions
from cupy.random import generator


def rand(*size, **kwarg):
    """Returns an array of uniform random values over the interval ``[0, 1)``.

    Each element of the array is uniformly distributed on the half-open
    interval ``[0, 1)``. All elements are identically and independently
    distributed (i.i.d.).

    Args:
        size (tuple of ints): The shape of the array.
        dtype: Data type specifier. Only float32 and float64 types are allowed.
            The default is float64.
        allocator (function): CuPy memory allocator. :func:`cupy.cuda.alloc` is
            used by default.

    Returns:
        cupy.ndarray: A random array.

    .. seealso:: :func:`numpy.random.rand`

    """
    dtype = kwarg.pop('dtype', float)
    allocator = kwarg.pop('allocator', cuda.alloc)
    if kwarg:
        raise TypeError('rand() got unexpected keyward arguments %s'
                        % ', '.join(kwarg.keys()))
    return random_sample(size=size, dtype=dtype, allocator=allocator)


def randn(*size, **kwarg):
    """Returns an array of standand normal random values.

    Each element of the array is normally distributed with zero mean and unit
    variance. All elements are identically and independently distributed
    (i.i.d.).

    Args:
        size (tuple of ints): The shape of the array.
        dtype: Data type specifier. Only float32 and float64 types are allowed.
            The default is float64.
        allocator (function): CuPy memory allocator. :func:`cupy.cuda.alloc` is
            used by default.

    Returns:
        cupy.ndarray: An array of standanr normal random values.

    .. seealso:: :func:`numpy.random.randn`

    """
    dtype = kwarg.pop('dtype', float)
    allocator = kwarg.pop('allocator', cuda.alloc)
    if kwarg:
        raise TypeError('randn() got unexpected keyward arguments %s'
                        % ', '.join(kwarg.keys()))
    return distributions.normal(size=size, dtype=dtype, allocator=allocator)


def randint(low, high=None, size=None, dtype=int, allocator=cuda.alloc):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def random_integers(low, high=None, size=None, dtype=int,
                    allocator=cuda.alloc):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def random_sample(size=None, dtype=float, allocator=cuda.alloc):
    """Returns an array of random values over the interval ``[0, 1)``.

    This is a variant of :func:`cupy.random.rand`.

    Args:
        size (int or tuple of ints): The shape of the array.
        dtype: Data type specifier. Only float32 and float64 types are allowed.
        allocator (function): CuPy memory allocator.

    Returns:
        cupy.ndarray: An array of uniformly distributed random values.

    .. seealso:: :func:`numpy.random.random_sample`

    """
    rs = generator.get_random_state()
    return rs.random_sample(size=size, dtype=dtype, allocator=allocator)


def choice(a, size=None, replace=True, p=None, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError

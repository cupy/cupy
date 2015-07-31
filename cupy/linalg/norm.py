def norm(x, ord=None, axis=None, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def cond(x, p=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def det(a, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def matrix_rank(M, tol=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def slogdet(a, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None, allocator=None):
    """Returns the sum along the diagonals of an array.

    It computes the sum along the diagonals at ``axis1`` and ``axis2``.

    Args:
        a (cupy.ndarray): Array to take trace.
        offset (int): Index of diagonals. Zero indicates the main diagonal, a
            positive value an upper diagonal, and a negative value a lower
            diagonal.
        axis1 (int): The first axis along which the trace is taken.
        axis2 (int): The second axis along which the trace is taken.
        dtype: Data type specifier of the output.
        out (cupy.ndarray): Output array.
        allocator (function): CuPy memory allocator. The allocator of ``a`` is
            used by default.

    Returns:
        cupy.ndarray: The trace of ``a`` along axes ``(axis1, axis2)``.

    .. seealso:: :func:`numpy.trace`

    """
    d = a.diagonal(offset, axis1, axis2)
    return d.sum(-1, dtype, out, False, allocator)

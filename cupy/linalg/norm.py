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
    d = a.diagonal(offset, axis1, axis2)
    return d.sum(-1, dtype, out, False, allocator)

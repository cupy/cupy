from cupy.logic import ufunc


def allclose(a, b, rtol=1e-05, atol=1e-08):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def array_equal(a1, a2):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def array_equiv(a1, a2):
    # TODO(beam2d): Implement it
    raise NotImplementedError


greater = ufunc.create_comparison('greater', '>')
greater_equal = ufunc.create_comparison('greater_equal', '>=')
less = ufunc.create_comparison('less', '<')
less_equal = ufunc.create_comparison('less_equal', '<=')
equal = ufunc.create_comparison('equal', '==')
not_equal = ufunc.create_comparison('not_equal', '!=')

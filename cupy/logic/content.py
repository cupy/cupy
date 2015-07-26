from cupy import elementwise


def _create_float_test_ufunc(name):
    return elementwise.create_ufunc(
        'cupy_' + name, ['e->?', 'f->?', 'd->?'], 'out0 = %s(in0)' % name)


isfinite = _create_float_test_ufunc('isfinite')
isinf = _create_float_test_ufunc('isinf')
isnan = _create_float_test_ufunc('isnan')


def isneginf(x, y=None, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def isposinf(x, y=None, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError

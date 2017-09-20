import cupy


def all(a, axis=None, out=None, keepdims=False):
    assert isinstance(a, cupy.ndarray)
    return a.all(axis=axis, out=out, keepdims=keepdims)


def any(a, axis=None, out=None, keepdims=False):
    assert isinstance(a, cupy.ndarray)
    return a.any(axis=axis, out=out, keepdims=keepdims)

from cupy import reduction


def amin(a, axis=None, out=None, keepdims=False, dtype=None, allocator=None):
    return reduction.amin(a, axis, dtype, out, keepdims, allocator)


def amax(a, axis=None, out=None, keepdims=False, dtype=None, allocator=None):
    return reduction.amax(a, axis, dtype, out, keepdims, allocator)


def nanmin(a, axis=None, out=None, keepdims=False, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def nanmax(a, axis=None, out=None, keepdims=False, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def ptp(a, axis=None, out=None, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def percentile(a, q, axis=None, out=None, overwrite_input=False,
               interpolation='linear', keepdims=False, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError

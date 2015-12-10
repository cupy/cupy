# flake8: NOQA
# "flake8: NOQA" to suppress warning "H104  File contains nothing but comments"

# TODO(okuta): Implement fliplr


# TODO(okuta): Implement flipud


def roll(a, shift, axis=None):
    if axis is None:
        size = a.size
    else:
        axis = int(axis)
        if axis >= a.ndim:
            raise ValueError('axis must be >= 0 and < %d' % a.ndim)
        size = a.shape[axis]
    if size == 0:
        return a
    shift %= size
    indexes = cupy.concatenate(
        (cupy.arange(size - shift, size), cupy.arange(size - shift)))
    res = a.take(indexes, axis)

    if axis is None:
        res = res.reshape(a.shape)
    return res

# TODO(okuta): Implement rot90

import numpy

from cupy import elementwise


def copyto(dst, src, casting='same_kind', where=None):
    if not numpy.can_cast(src.dtype, dst.dtype, casting):
        raise TypeError('Cannot cast %s to %s in %s casting mode' %
                        (src.dtype, dst.dtype, casting))
    if where is None:
        elementwise.copy(src, dst)
    else:
        elementwise.copy_where(src, where, dst)

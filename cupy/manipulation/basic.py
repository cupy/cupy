import numpy

from cupy import elementwise


def copyto(dst, src, casting='same_kind', where=None):
    """Copies values from one array to another with broadcasting.

    This function can be called for arrays on different devices. In this case,
    casting, ``where``, and broadcasting is not supported, and an exception is
    raised if these are used.

    Args:
        dst (cupy.ndarray): Target array.
        src (cupy.ndarray): Source array.
        casting (str): Casting rule. See :func:`numpy.can_cast` for detail.
        where (cupy.ndarray of bool): If specified, this array acts as a mask,
            and an element is copied only if the corresponding element of
            ``where`` is True.

    .. seealso:: :func:`numpy.copyto`

    """
    if not numpy.can_cast(src.dtype, dst.dtype, casting):
        raise TypeError('Cannot cast %s to %s in %s casting mode' %
                        (src.dtype, dst.dtype, casting))
    if dst.size == 0:
        return

    if where is None:
        if _can_memcpy(dst, src):
            dst.data.copy_from(src.data, src.nbytes)
        else:
            elementwise.copy(src, dst)
    else:
        elementwise.copy_where(src, where, dst)


def _can_memcpy(dst, src):
    c_contiguous = dst.flags.c_contiguous and src.flags.c_contiguous
    f_contiguous = dst.flags.f_contiguous and src.flags.f_contiguous
    return (c_contiguous or f_contiguous) and dst.dtype == src.dtype and \
        dst.size == src.size

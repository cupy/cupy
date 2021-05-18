import numpy

from cupy import _core
from cupy._core import _fusion_interface
from cupy._core import fusion
from cupy._sorting import search


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

    src_type = type(src)
    src_is_python_scalar = src_type in (
        int, bool, float, complex,
        fusion._FusionVarScalar, _fusion_interface._ScalarProxy)
    if src_is_python_scalar:
        src_dtype = numpy.dtype(type(src))
        can_cast = numpy.can_cast(src, dst.dtype, casting)
    else:
        src_dtype = src.dtype
        can_cast = numpy.can_cast(src_dtype, dst.dtype, casting)

    if not can_cast:
        raise TypeError('Cannot cast %s to %s in %s casting mode' %
                        (src_dtype, dst.dtype, casting))
    if fusion._is_fusing():
        if where is None:
            _core.elementwise_copy(src, dst)
        else:
            fusion._call_ufunc(search._where_ufunc, where, src, dst, dst)
        return

    if dst.size == 0:
        return

    if src_is_python_scalar and where is None:
        dst.fill(src)
        return

    if where is None:
        if _can_memcpy(dst, src):
            dst.data.copy_from_async(src.data, src.nbytes)
        else:
            device = dst.device
            with device:
                if src.device != device:
                    src = src.copy()
                _core.elementwise_copy(src, dst)
    else:
        _core.elementwise_copy_where(src, where, dst)


def _can_memcpy(dst, src):
    c_contiguous = dst.flags.c_contiguous and src.flags.c_contiguous
    f_contiguous = dst.flags.f_contiguous and src.flags.f_contiguous
    return (c_contiguous or f_contiguous) and dst.dtype == src.dtype and \
        dst.size == src.size

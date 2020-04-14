import numpy

import cupy
from cupy import core
from cupy.core import fusion
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
        int, bool, float, complex, fusion._FusionVarScalar)
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
            core.elementwise_copy(src, dst)
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
                core.elementwise_copy(src, dst)
    else:
        core.elementwise_copy_where(src, where, dst)


def _can_memcpy(dst, src):
    c_contiguous = dst.flags.c_contiguous and src.flags.c_contiguous
    f_contiguous = dst.flags.f_contiguous and src.flags.f_contiguous
    return (c_contiguous or f_contiguous) and dst.dtype == src.dtype and \
        dst.size == src.size


_putmask_kernel = core._kernel.ElementwiseKernel(
    'Q mask, raw S values, uint64 len_vals', 'T out',
    '''
    if (mask) out = (T) values[i % len_vals];
    ''',
    'putmask_kernel'
)


def putmask(a, mask, values):
    """
    Changes elements of an array inplace, based on conditional mask and
    input values.

    Sets ``a.flat[n] = values[n]`` for each n where ``mask.flat[n]==True``.
    If `values` is not the same size as `a` and `mask` then it will repeat.

    Args
        a (cupy.ndarray): Target array.
        mask (cupy.ndarray): Boolean mask array. It has to be
            the same shape as `a`.
        values (cupy.ndarray or scalar): Values to put into `a` where `mask`
            is True. If `values` is smaller than `a`, then it will be
            repeated.

    Examples
    --------
    >>> x = cupy.arange(6).reshape(2, 3)
    >>> cupy.putmask(x, x>2, x**2)
    >>> x
    array([[ 0,  1,  2],
           [ 9, 16, 25]])

    If `values` is smaller than `a` it is repeated:

    >>> x = cupy.arange(6)
    >>> cupy.putmask(x, x>2, [-33, -44])
    >>> x
    array([  0,  1,  2, -44, -33, -44])

    .. seealso:: :func:`numpy.putmask`

    """

    if not isinstance(a, cupy.ndarray):
        raise ValueError('`a` should be of type cupy.ndarray')
    if not isinstance(mask, cupy.ndarray):
        raise ValueError('`mask` should be of type cupy.ndarray')
    if not (cupy.isscalar(values) or isinstance(values, cupy.ndarray)):
        raise ValueError('`values` should be of type cupy.ndarray')

    if not a.shape == mask.shape:
        raise ValueError('mask and data must be the same size')

    if cupy.isscalar(values):
        a[mask] = values

    elif not numpy.can_cast(values.dtype, a.dtype):
        raise TypeError('Cannot cast array data from'
                        ' {} to {} according to the rule \'safe\''
                        .format(values.dtype, a.dtype))

    elif a.shape == values.shape:
        a[mask] = values[mask]

    else:
        if values.ndim > 1:
            values = values.ravel()
        _putmask_kernel(mask, values, len(values), a)

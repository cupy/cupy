import cupy
from cupy import _core


def asarray_chkfinite(a, dtype=None, order=None):
    """Converts the given input to an array,
    and raises an error if the input contains NaNs or Infs.

    Args:
        a: array like.
        dtype: data type, optional
        order: {'C', 'F', 'A', 'K'}, optional

    Returns:
        cupy.ndarray: An array on the current device.

    .. note::
        This function performs device synchronization.

    .. seealso:: :func:`numpy.asarray_chkfinite`

    """

    a = cupy.asarray(a, dtype=dtype, order=order)
    if not cupy.isfinite(a).all():
        raise ValueError(
            "array must not contain Infs or NaNs")
    return a


def asfarray(a, dtype=cupy.float_):
    """Converts array elements to float type.

    Args:
        a (cupy.ndarray): Source array.
        dtype: str or dtype object, optional

    Returns:
        cupy.ndarray: The input array ``a`` as a float ndarray.

    .. seealso:: :func:`numpy.asfarray`

    """
    if not cupy.issubdtype(dtype, cupy.inexact):
        dtype = cupy.float_
    return cupy.asarray(a, dtype=dtype)


def asfortranarray(a, dtype=None):
    """Return an array laid out in Fortran order in memory.

    Args:
        a (~cupy.ndarray): The input array.
        dtype (str or dtype object, optional): By default, the data-type is
            inferred from the input data.

    Returns:
        ~cupy.ndarray: The input `a` in Fortran, or column-major, order.

    .. seealso:: :func:`numpy.asfortranarray`

    """
    return _core.asfortranarray(a, dtype)


def require(a, dtype=None, requirements=None):
    """Return an array which satisfies the requirements.

    Args:
        a (~cupy.ndarray): The input array.
        dtype (str or dtype object, optional): The required data-type.
            If None preserve the current dtype.
        requirements (str or list of str): The requirements can be any
            of the following

            * 'F_CONTIGUOUS' ('F', 'FORTRAN') - ensure a Fortran-contiguous \
                array. \

            * 'C_CONTIGUOUS' ('C', 'CONTIGUOUS') - ensure a C-contiguous array.

            * 'OWNDATA' ('O')      - ensure an array that owns its own data.

    Returns:
        ~cupy.ndarray: The input array ``a`` with specified requirements and
        type if provided.

    .. seealso:: :func:`numpy.require`

    """

    possible_flags = {'C': 'C', 'C_CONTIGUOUS': 'C', 'CONTIGUOUS': 'C',
                      'F': 'F', 'F_CONTIGUOUS': 'F', 'FORTRAN': 'F',
                      'O': 'OWNDATA', 'OWNDATA': 'OWNDATA'}

    if not requirements:
        try:
            return cupy.asanyarray(a, dtype=dtype)
        except TypeError:
            raise(ValueError("Incorrect dtype \"{}\" provided".format(dtype)))
    else:
        try:
            requirements = {possible_flags[x.upper()] for x in requirements}
        except KeyError:
            raise(ValueError("Incorrect flag \"{}\" in requirements".format(
                             (set(requirements) -
                              set(possible_flags.keys())).pop())))

    order = 'A'
    if requirements >= {'C', 'F'}:
        raise ValueError('Cannot specify both "C" and "F" order')
    elif 'F' in requirements:
        order = 'F_CONTIGUOUS'
        requirements.remove('F')
    elif 'C' in requirements:
        order = 'C_CONTIGUOUS'
        requirements.remove('C')

    copy = 'OWNDATA' in requirements
    try:
        arr = cupy.array(a, dtype=dtype, order=order, copy=copy, subok=False)
    except TypeError:
        raise(ValueError("Incorrect dtype \"{}\" provided".format(dtype)))
    return arr

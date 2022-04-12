import numpy

from cupy import _core
from cupy._core import fusion


def array(obj, dtype=None, copy=True, order='K', subok=False, ndmin=0):
    """Creates an array on the current device.

    This function currently does not support the ``subok`` option.

    Args:
        obj: :class:`cupy.ndarray` object or any other object that can be
            passed to :func:`numpy.array`.
        dtype: Data type specifier.
        copy (bool): If ``False``, this function returns ``obj`` if possible.
            Otherwise this function always returns a new array.
        order ({'C', 'F', 'A', 'K'}): Row-major (C-style) or column-major
            (Fortran-style) order.
            When ``order`` is ``'A'``, it uses ``'F'`` if ``a`` is column-major
            and uses ``'C'`` otherwise.
            And when ``order`` is ``'K'``, it keeps strides as closely as
            possible.
            If ``obj`` is :class:`numpy.ndarray`, the function returns ``'C'``
            or ``'F'`` order array.
        subok (bool): If ``True``, then sub-classes will be passed-through,
            otherwise the returned array will be forced to be a base-class
            array (default).
        ndmin (int): Minimum number of dimensions. Ones are inserted to the
            head of the shape if needed.

    Returns:
        cupy.ndarray: An array on the current device.

    .. note::
       This method currently does not support ``subok`` argument.

    .. note::
       If ``obj`` is an `numpy.ndarray` instance that contains big-endian data,
       this function automatically swaps its byte order to little-endian,
       which is the NVIDIA and AMD GPU architecture's native use.

    .. seealso:: :func:`numpy.array`

    """
    return _core.array(obj, dtype, copy, order, subok, ndmin)


def asarray(a, dtype=None, order=None):
    """Converts an object to array.

    This is equivalent to ``array(a, dtype, copy=False)``.
    This function currently does not support the ``order`` option.

    Args:
        a: The source object.
        dtype: Data type specifier. It is inferred from the input by default.
        order ({'C', 'F'}):
            Whether to use row-major (C-style) or column-major (Fortran-style)
            memory representation. Defaults to ``'C'``. ``order`` is ignored
            for objects that are not :class:`cupy.ndarray`, but have the
            ``__cuda_array_interface__`` attribute.

    Returns:
        cupy.ndarray: An array on the current device. If ``a`` is already on
        the device, no copy is performed.

    .. note::
       If ``a`` is an `numpy.ndarray` instance that contains big-endian data,
       this function automatically swaps its byte order to little-endian,
       which is the NVIDIA and AMD GPU architecture's native use.

    .. seealso:: :func:`numpy.asarray`

    """
    return _core.array(a, dtype, False, order)


def asanyarray(a, dtype=None, order=None):
    """Converts an object to array.

    This is currently equivalent to :func:`cupy.asarray`, since there is no
    subclass of :class:`cupy.ndarray` in CuPy. Note that the original
    :func:`numpy.asanyarray` returns the input array as is if it is an instance
    of a subtype of :class:`numpy.ndarray`.

    .. seealso:: :func:`cupy.asarray`, :func:`numpy.asanyarray`

    """
    return _core.array(a, dtype, False, order)


def ascontiguousarray(a, dtype=None):
    """Returns a C-contiguous array.

    Args:
        a (cupy.ndarray): Source array.
        dtype: Data type specifier.

    Returns:
        cupy.ndarray: If no copy is required, it returns ``a``. Otherwise, it
        returns a copy of ``a``.

    .. seealso:: :func:`numpy.ascontiguousarray`

    """
    return _core.ascontiguousarray(a, dtype)

# TODO(okuta): Implement asmatrix


def copy(a, order='K'):
    """Creates a copy of a given array on the current device.

    This function allocates the new array on the current device. If the given
    array is allocated on the different device, then this function tries to
    copy the contents over the devices.

    Args:
        a (cupy.ndarray): The source array.
        order ({'C', 'F', 'A', 'K'}): Row-major (C-style) or column-major
            (Fortran-style) order.
            When ``order`` is ``'A'``, it uses ``'F'`` if ``a`` is column-major
            and uses ``'C'`` otherwise.
            And when ``order`` is ``'K'``, it keeps strides as closely as
            possible.

    Returns:
        cupy.ndarray: The copy of ``a`` on the current device.

    .. seealso:: :func:`numpy.copy`, :meth:`cupy.ndarray.copy`

    """
    if fusion._is_fusing():
        if order != 'K':
            raise NotImplementedError(
                'cupy.copy does not support `order` in fusion yet.')
        return fusion._call_ufunc(_core.elementwise_copy, a)

    # If the current device is different from the device of ``a``, then this
    # function allocates a new array on the current device, and copies the
    # contents over the devices.
    return a.copy(order=order)


def frombuffer(*args, **kwargs):
    """Interpret a buffer as a 1-dimensional array.

    .. note::
        Uses NumPy's ``frombuffer`` and coerces the result to a CuPy array.

    .. seealso:: :func:`numpy.frombuffer`

    """
    return asarray(numpy.frombuffer(*args, **kwargs))


def fromfile(*args, **kwargs):
    """Reads an array from a file.

    .. note::
        Uses NumPy's ``fromfile`` and coerces the result to a CuPy array.

    .. note::
       If you let NumPy's ``fromfile`` read the file in big-endian, CuPy
       automatically swaps its byte order to little-endian, which is the NVIDIA
       and AMD GPU architecture's native use.

    .. seealso:: :func:`numpy.fromfile`

    """
    return asarray(numpy.fromfile(*args, **kwargs))


def fromfunction(*args, **kwargs):
    """Construct an array by executing a function over each coordinate.

    .. note::
        Uses NumPy's ``fromfunction`` and coerces the result to a CuPy array.

    .. seealso:: :func:`numpy.fromfunction`
    """
    return asarray(numpy.fromfunction(*args, **kwargs))


def fromiter(*args, **kwargs):
    """Create a new 1-dimensional array from an iterable object.

    .. note::
        Uses NumPy's ``fromiter`` and coerces the result to a CuPy array.

    .. seealso:: :func:`numpy.fromiter`
    """
    return asarray(numpy.fromiter(*args, **kwargs))


def fromstring(*args, **kwargs):
    """A new 1-D array initialized from text data in a string.

    .. note::
        Uses NumPy's ``fromstring`` and coerces the result to a CuPy array.

    .. seealso:: :func:`numpy.fromstring`
    """
    return asarray(numpy.fromstring(*args, **kwargs))


def loadtxt(*args, **kwargs):
    """Load data from a text file.

    .. note::
        Uses NumPy's ``loadtxt`` and coerces the result to a CuPy array.

    .. seealso:: :func:`numpy.loadtxt`
    """
    return asarray(numpy.loadtxt(*args, **kwargs))


def genfromtxt(*args, **kwargs):
    """Load data from text file, with missing values handled as specified.

    .. note::
        Uses NumPy's ``genfromtxt`` and coerces the result to a CuPy array.

    .. seealso:: :func:`numpy.genfromtxt`
    """
    return asarray(numpy.genfromtxt(*args, **kwargs))

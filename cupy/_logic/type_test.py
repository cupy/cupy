import numpy

import cupy


def iscomplex(x):
    """Returns a bool array, where True if input element is complex.

    What is tested is whether the input has a non-zero imaginary part, not if
    the input type is complex.

    Args:
        x (cupy.ndarray): Input array.

    Returns:
        cupy.ndarray: Boolean array of the same shape as ``x``.

    .. seealso::
        :func:`isreal`, :func:`iscomplexobj`

    Examples
    --------
    >>> cupy.iscomplex(cupy.array([1+1j, 1+0j, 4.5, 3, 2, 2j]))
    array([ True, False, False, False, False,  True])

    """
    if numpy.isscalar(x):
        return numpy.iscomplex(x)
    if not isinstance(x, cupy.ndarray):
        return cupy.asarray(numpy.iscomplex(x))
    if x.dtype.kind == 'c':
        return x.imag != 0
    return cupy.zeros(x.shape, bool)


def iscomplexobj(x):
    """Check for a complex type or an array of complex numbers.

    The type of the input is checked, not the value. Even if the input
    has an imaginary part equal to zero, `iscomplexobj` evaluates to True.

    Args:
        x (cupy.ndarray): Input array.

    Returns:
        bool: The return value, True if ``x`` is of a complex type or
        has at least one complex element.

    .. seealso::
        :func:`isrealobj`, :func:`iscomplex`

    Examples
    --------
    >>> cupy.iscomplexobj(cupy.array([3, 1+0j, True]))
    True
    >>> cupy.iscomplexobj(cupy.array([3, 1, True]))
    False

    """
    if not isinstance(x, cupy.ndarray):
        return numpy.iscomplexobj(x)
    return x.dtype.kind == 'c'


def isfortran(a):
    """Returns True if the array is Fortran contiguous but *not* C contiguous.

    If you only want to check if an array is Fortran contiguous use
    ``a.flags.f_contiguous`` instead.

    Args:
        a (cupy.ndarray): Input array.

    Returns:
        bool: The return value, True if ``a`` is Fortran contiguous but not C
        contiguous.

    .. seealso::
       :func:`~numpy.isfortran`

    Examples
    --------

    cupy.array allows to specify whether the array is written in C-contiguous
    order (last index varies the fastest), or FORTRAN-contiguous order in
    memory (first index varies the fastest).

    >>> a = cupy.array([[1, 2, 3], [4, 5, 6]], order='C')
    >>> a
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> cupy.isfortran(a)
    False

    >>> b = cupy.array([[1, 2, 3], [4, 5, 6]], order='F')
    >>> b
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> cupy.isfortran(b)
    True

    The transpose of a C-ordered array is a FORTRAN-ordered array.

    >>> a = cupy.array([[1, 2, 3], [4, 5, 6]], order='C')
    >>> a
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> cupy.isfortran(a)
    False
    >>> b = a.T
    >>> b
    array([[1, 4],
           [2, 5],
           [3, 6]])
    >>> cupy.isfortran(b)
    True

    C-ordered arrays evaluate as False even if they are also FORTRAN-ordered.

    >>> cupy.isfortran(np.array([1, 2], order='F'))
    False

    """
    return a.flags.f_contiguous and not a.flags.c_contiguous


def isreal(x):
    """Returns a bool array, where True if input element is real.

    If element has complex type with zero complex part, the return value
    for that element is True.

    Args:
        x (cupy.ndarray): Input array.

    Returns:
        cupy.ndarray: Boolean array of same shape as ``x``.

    .. seealso::
        :func:`iscomplex`, :func:`isrealobj`

    Examples
    --------
    >>> cupy.isreal(cp.array([1+1j, 1+0j, 4.5, 3, 2, 2j]))
    array([False,  True,  True,  True,  True, False])

    """
    if numpy.isscalar(x):
        return numpy.isreal(x)
    if not isinstance(x, cupy.ndarray):
        return cupy.asarray(numpy.isreal(x))
    if x.dtype.kind == 'c':
        return x.imag == 0
    return cupy.ones(x.shape, bool)


def isrealobj(x):
    """Return True if x is a not complex type or an array of complex numbers.

    The type of the input is checked, not the value. So even if the input
    has an imaginary part equal to zero, `isrealobj` evaluates to False
    if the data type is complex.

    Args:
        x (cupy.ndarray): The input can be of any type and shape.

    Returns:
        bool: The return value, False if ``x`` is of a complex type.

    .. seealso::
        :func:`iscomplexobj`, :func:`isreal`

    Examples
    --------
    >>> cupy.isrealobj(cupy.array([3, 1+0j, True]))
    False
    >>> cupy.isrealobj(cupy.array([3, 1, True]))
    True

    """
    if not isinstance(x, cupy.ndarray):
        return numpy.isrealobj(x)
    return x.dtype.kind != 'c'

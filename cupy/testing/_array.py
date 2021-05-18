import numpy.testing

import cupy


# NumPy-like assertion functions that accept both NumPy and CuPy arrays

def assert_allclose(actual, desired, rtol=1e-7, atol=0, err_msg='',
                    verbose=True):
    """Raises an AssertionError if objects are not equal up to desired tolerance.

    Args:
         actual(numpy.ndarray or cupy.ndarray): The actual object to check.
         desired(numpy.ndarray or cupy.ndarray): The desired, expected object.
         rtol(float): Relative tolerance.
         atol(float): Absolute tolerance.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting
             values are appended to the error message.

    .. seealso:: :func:`numpy.testing.assert_allclose`

    """  # NOQA
    numpy.testing.assert_allclose(
        cupy.asnumpy(actual), cupy.asnumpy(desired),
        rtol=rtol, atol=atol, err_msg=err_msg, verbose=verbose)


def assert_array_almost_equal(x, y, decimal=6, err_msg='', verbose=True):
    """Raises an AssertionError if objects are not equal up to desired precision.

    Args:
         x(numpy.ndarray or cupy.ndarray): The actual object to check.
         y(numpy.ndarray or cupy.ndarray): The desired, expected object.
         decimal(int): Desired precision.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting
             values are appended to the error message.

    .. seealso:: :func:`numpy.testing.assert_array_almost_equal`
    """  # NOQA
    numpy.testing.assert_array_almost_equal(
        cupy.asnumpy(x), cupy.asnumpy(y), decimal=decimal,
        err_msg=err_msg, verbose=verbose)


def assert_array_almost_equal_nulp(x, y, nulp=1):
    """Compare two arrays relatively to their spacing.

    Args:
         x(numpy.ndarray or cupy.ndarray): The actual object to check.
         y(numpy.ndarray or cupy.ndarray): The desired, expected object.
         nulp(int): The maximum number of unit in the last place for tolerance.

    .. seealso:: :func:`numpy.testing.assert_array_almost_equal_nulp`
    """
    numpy.testing.assert_array_almost_equal_nulp(
        cupy.asnumpy(x), cupy.asnumpy(y), nulp=nulp)


def assert_array_max_ulp(a, b, maxulp=1, dtype=None):
    """Check that all items of arrays differ in at most N Units in the Last Place.

    Args:
         a(numpy.ndarray or cupy.ndarray): The actual object to check.
         b(numpy.ndarray or cupy.ndarray): The desired, expected object.
         maxulp(int): The maximum number of units in the last place
             that elements of ``a`` and ``b`` can differ.
         dtype(numpy.dtype): Data-type to convert ``a`` and ``b`` to if given.

    .. seealso:: :func:`numpy.testing.assert_array_max_ulp`
    """  # NOQA
    numpy.testing.assert_array_max_ulp(
        cupy.asnumpy(a), cupy.asnumpy(b), maxulp=maxulp, dtype=dtype)


def assert_array_equal(x, y, err_msg='', verbose=True, strides_check=False):
    """Raises an AssertionError if two array_like objects are not equal.

    Args:
         x(numpy.ndarray or cupy.ndarray): The actual object to check.
         y(numpy.ndarray or cupy.ndarray): The desired, expected object.
         strides_check(bool): If ``True``, consistency of strides is also
             checked.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values
             are appended to the error message.

    .. seealso:: :func:`numpy.testing.assert_array_equal`
    """
    numpy.testing.assert_array_equal(
        cupy.asnumpy(x), cupy.asnumpy(y), err_msg=err_msg,
        verbose=verbose)

    if strides_check:
        if x.strides != y.strides:
            msg = ['Strides are not equal:']
            if err_msg:
                msg = [msg[0] + ' ' + err_msg]
            if verbose:
                msg.append(' x: {}'.format(x.strides))
                msg.append(' y: {}'.format(y.strides))
            raise AssertionError('\n'.join(msg))


def assert_array_list_equal(xlist, ylist, err_msg='', verbose=True):
    """Compares lists of arrays pairwise with ``assert_array_equal``.

    Args:
         x(array_like): Array of the actual objects.
         y(array_like): Array of the desired, expected objects.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values
             are appended to the error message.

    Each element of ``x`` and ``y`` must be either :class:`numpy.ndarray`
    or :class:`cupy.ndarray`. ``x`` and ``y`` must have same length.
    Otherwise, this function raises ``AssertionError``.
    It compares elements of ``x`` and ``y`` pairwise
    with :func:`assert_array_equal` and raises error if at least one
    pair is not equal.

    .. seealso:: :func:`numpy.testing.assert_array_equal`
    """
    x_type = type(xlist)
    y_type = type(ylist)
    if x_type is not y_type:
        raise AssertionError(
            'Matching types of list or tuple are expected, '
            'but were different types '
            '(xlist:{} ylist:{})'.format(x_type, y_type))
    if x_type not in (list, tuple):
        raise AssertionError(
            'List or tuple is expected, but was {}'.format(x_type))
    if len(xlist) != len(ylist):
        raise AssertionError('List size is different')
    for x, y in zip(xlist, ylist):
        numpy.testing.assert_array_equal(
            cupy.asnumpy(x), cupy.asnumpy(y), err_msg=err_msg,
            verbose=verbose)


def assert_array_less(x, y, err_msg='', verbose=True):
    """Raises an AssertionError if array_like objects are not ordered by less than.

    Args:
         x(numpy.ndarray or cupy.ndarray): The smaller object to check.
         y(numpy.ndarray or cupy.ndarray): The larger object to compare.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values
             are appended to the error message.

    .. seealso:: :func:`numpy.testing.assert_array_less`
    """  # NOQA
    numpy.testing.assert_array_less(
        cupy.asnumpy(x), cupy.asnumpy(y), err_msg=err_msg,
        verbose=verbose)

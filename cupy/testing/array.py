import numpy.testing

import cupy


# NumPy-like assertion functions that accept both NumPy and CuPy arrays

def assert_allclose(actual, desired, rtol=1e-7, atol=0, err_msg='',
                    verbose=True):
    numpy.testing.assert_allclose(
        cupy.asnumpy(actual), cupy.asnumpy(desired),
        rtol=rtol, atol=atol, err_msg=err_msg, verbose=verbose)


def assert_array_almost_equal(x, y, decimal=6, err_msg='', verbose=True):
    numpy.testing.assert_array_almost_equal(
        cupy.asnumpy(x), cupy.asnumpy(y), decimal=decimal,
        err_msg=err_msg, verbose=verbose)


def assert_arrays_almost_equal_nulp(x, y, nulp=1):
    numpy.testing.assert_arrays_almost_equal_nulp(
        cupy.asnumpy(x), cupy.asnumpy(y), nulp=nulp)


def assert_array_max_ulp(a, b, maxulp=1, dtype=None):
    numpy.testing.assert_array_max_ulp(
        cupy.asnumpy(a), cupy.asnumpy(b), maxulp=maxulp, dtype=dtype)


def assert_array_equal(x, y, err_msg='', verbose=True):
    numpy.testing.assert_array_equal(
        cupy.asnumpy(x), cupy.asnumpy(y), err_msg=err_msg,
        verbose=verbose)


def assert_array_list_equal(xlist, ylist, err_msg='', verbose=True):
    if len(xlist) != len(ylist):
        raise AssertionError('List size is different')
    for x, y in zip(xlist, ylist):
        numpy.testing.assert_array_equal(
            cupy.asnumpy(x), cupy.asnumpy(y), err_msg=err_msg,
            verbose=verbose)


def assert_array_less(x, y, err_msg='', verbose=True):
    numpy.testing.assert_array_less(
        cupy.asnumpy(x), cupy.asnumpy(y), err_msg=err_msg,
        verbose=verbose)

import pickle
import unittest

import numpy
import pytest
try:
    import scipy.sparse
    scipy_available = True
except ImportError:
    scipy_available = False

import cupy
from cupy import testing
from cupyx.scipy import sparse


p_format = ["csr", "csc"]


@pytest.mark.parametrize('format', p_format)
@pytest.mark.parametrize('density', [0.1, 0.2, 0.5, 0.9, 1.0])
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('n_rows', [100, 1000])
def test_major_slice(format, density, dtype, n_rows):
    a = cupy.sparse.random(n_rows, 10, format=format, density=density, dtype=dtype)
    cupy.testing.assert_array_equal(a[5:9].todense(), a.todense()[5:9])

    cupy.cuda.Stream.null.synchronize()


@pytest.mark.parametrize('format', p_format)
@pytest.mark.parametrize('density', [0.1, 0.2, 0.5, 0.9, 1.0])
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('n_rows', [100, 1000])
def test_major_slice_minor_slice(format, density, dtype, n_rows):
    a = cupy.sparse.random(n_rows, 10, format=format, density=density, dtype=dtype)
    maj = slice(1, 5)
    min = slice(1, 5)

    expected = a.todense()[maj, min].ravel()
    actual = a[maj, min].todense().ravel()
    cupy.testing.assert_array_equal(actual, expected)

    cupy.cuda.Stream.null.synchronize()


@pytest.mark.parametrize('format', p_format)
@pytest.mark.parametrize('density', [0.1, 0.2, 0.5, 0.9, 1.0])
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('n_rows', [100, 1000])
def test_major_slice_minor_all(format, density, dtype, n_rows):
    a = cupy.sparse.random(n_rows, 10, format=format, density=density, dtype=dtype)
    maj = slice(1, 5)
    min = slice(None)

    expected = a.todense()[maj, min].ravel()
    actual = a[maj, min].todense().ravel()
    cupy.testing.assert_array_equal(actual, expected)

    cupy.cuda.Stream.null.synchronize()


@pytest.mark.parametrize('format', p_format)
@pytest.mark.parametrize('density', [0.1, 0.2, 0.5, 0.9, 1.0])
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('n_rows', [100, 1000])
def test_major_slice_minor_scalar(format, density, dtype, n_rows):
    a = cupy.sparse.random(n_rows, 10, format=format, density=density, dtype=dtype)
    maj = slice(1, 5)
    min = 5

    expected = a.todense()[maj, min].ravel()
    actual = a[maj, min].todense().ravel()
    cupy.testing.assert_array_equal(actual, expected)

    cupy.cuda.Stream.null.synchronize()


@pytest.mark.parametrize('format', p_format)
@pytest.mark.parametrize('density', [0.1, 0.2, 0.5, 0.9, 1.0])
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('n_rows', [100, 1000])
def test_major_scalar_minor_slice(format, density, dtype, n_rows):
    a = cupy.sparse.random(n_rows, 10, format=format, density=density, dtype=dtype)
    maj = 5
    min = slice(1, 5)

    expected = a.todense()[maj, min].ravel()
    actual = a[maj, min].todense().ravel()
    cupy.testing.assert_array_equal(actual, expected)

    cupy.cuda.Stream.null.synchronize()


@pytest.mark.parametrize('format', p_format)
@pytest.mark.parametrize('density', [0.1, 0.2, 0.5, 0.9, 1.0])
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('n_rows', [100, 1000])
def test_major_scalar_minor_all(format, density, dtype, n_rows):
    a = cupy.sparse.random(n_rows, 10, format=format, density=density, dtype=dtype)
    maj = 5
    min = slice(None)

    expected = a.todense()[maj, min].ravel()
    actual = a[maj, min].todense().ravel()
    cupy.testing.assert_array_equal(actual, expected)

    cupy.cuda.Stream.null.synchronize()


@pytest.mark.parametrize('format', p_format)
@pytest.mark.parametrize('density', [0.1, 0.2, 0.5, 0.9, 1.0])
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('n_rows', [100, 1000])
def test_major_scalar_minor_scalar(format, density, dtype, n_rows):
    a = cupy.sparse.random(n_rows, 10, format=format, density=density, dtype=dtype)
    maj = 5
    min = 5

    expected = a.todense()[maj, min].ravel()
    actual = a[maj, min].ravel()
    cupy.testing.assert_array_equal(actual, expected)
    cupy.cuda.Stream.null.synchronize()


@pytest.mark.parametrize('format', p_format)
@pytest.mark.parametrize('density', [0.1, 0.2, 0.5, 0.9, 1.0])
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('n_rows', [100, 1000])
def test_major_all_minor_scalar(format, density, dtype, n_rows):

    a = cupy.sparse.random(n_rows, 10, format=format, density=density, dtype=dtype)
    maj = slice(None)
    min = 5

    expected = a.todense()[maj, min].ravel()
    actual = a[maj, min].todense().ravel()
    cupy.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('format', p_format)
@pytest.mark.parametrize('density', [0.1, 0.2, 0.5, 0.9, 1.0])
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('n_rows', [100, 1000])
def test_major_all_minor_slice(format, density, dtype, n_rows):
    a = cupy.sparse.random(n_rows, 10, format=format, density=density, dtype=dtype)
    maj = slice(None)
    min = slice(5, 10)

    expected = a.todense()[maj, min].ravel()
    actual = a[maj, min].todense().ravel()
    cupy.testing.assert_array_equal(actual, expected)
    cupy.cuda.Stream.null.synchronize()


@pytest.mark.parametrize('format', p_format)
@pytest.mark.parametrize('density', [0.1, 0.2, 0.5, 0.9, 1.0])
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('n_rows', [100, 1000])
def test_major_all_minor_all(format, density, dtype, n_rows):
    a = cupy.sparse.random(n_rows, 10, format=format, density=density, dtype=dtype)
    maj = slice(None)
    min = slice(None)

    expected = a.todense()[maj, min].ravel()
    actual = a[maj, min].todense().ravel()
    cupy.testing.assert_array_equal(actual, expected)
    cupy.cuda.Stream.null.synchronize()


@pytest.mark.parametrize('format', p_format)
@pytest.mark.parametrize('density', [0.1, 0.2, 0.5, 0.9, 1.0])
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('n_rows', [100, 1000])
def test_major_all(format, density, dtype, n_rows):
    a = cupy.sparse.random(n_rows, 10, format=format, density=density, dtype=dtype)
    maj = slice(None)

    expected = a.todense()[maj].ravel()
    actual = a[maj].todense().ravel()
    cupy.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('format', p_format)
@pytest.mark.parametrize('density', [0.1, 0.2, 0.5, 0.9, 1.0])
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('n_rows', [100, 1000])
def test_major_scalar(format, density, dtype, n_rows):
    a = cupy.sparse.random(n_rows, 10, format=format, density=density, dtype=dtype)
    cupy.testing.assert_array_equal(a[5].todense().ravel(), a.todense()[5].ravel())
    cupy.cuda.Stream.null.synchronize()


@pytest.mark.parametrize('format', ['csr']) #'csc'])
@pytest.mark.parametrize('density', [0.1, 0.2, 0.5, 0.9, 1.0])
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('n_rows', [100, 1000])
def test_major_fancy(format, density, dtype, n_rows):
    a = cupy.sparse.random(n_rows, 10, format=format, density=density, dtype=dtype)

    idx = [1, 5, 4]

    expected = a.todense()[idx].ravel()
    actual = a[idx].todense().ravel()
    cupy.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('format', ['csr', 'csc']) #'csc'])
@pytest.mark.parametrize('density', [0.1, 0.2, 0.5, 0.9, 1.0])
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('n_rows', [100, 1000])
def test_major_fancy_minor_fancy(format, density, dtype, n_rows):
    a = cupy.sparse.random(n_rows, 10, format=format, density=density, dtype=dtype)

    maj = [1, 5, 4]
    min = [1, 5, 4]

    expected = a.todense()[maj, min].ravel()
    actual = a[maj, min].ravel()

    cupy.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('format', ['csr', 'csc']) #'csc'])
@pytest.mark.parametrize('density', [0.1, 0.2, 0.5, 0.9, 1.0])
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('n_rows', [100, 1000])
def test_major_fancy_minor_all(format, density, dtype, n_rows):
    a = cupy.sparse.random(n_rows, 10, format=format, density=density, dtype=dtype)

    maj = [1, 5, 4]
    min = slice(None)

    expected = a.todense()[maj, min].ravel()
    actual = a[maj, min].todense().ravel()

    cupy.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('format', ['csr', 'csc']) #'csc'])
@pytest.mark.parametrize('density', [0.1, 0.2, 0.5, 0.9, 1.0])
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('n_rows', [100, 1000])
def test_major_fancy_minor_scalar(format, density, dtype, n_rows):
    a = cupy.sparse.random(n_rows, 10, format=format, density=density, dtype=dtype)

    maj = [1, 5, 4]
    min = 5

    expected = a.todense()[maj, min].ravel()
    actual = a[maj, min].todense().ravel()

    cupy.testing.assert_array_equal(actual, expected)



@pytest.mark.parametrize('format', ['csr', 'csc']) #'csc'])
@pytest.mark.parametrize('density', [0.1, 0.2, 0.5, 0.9, 1.0])
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('n_rows', [100, 1000])
def test_major_scalar_minor_fancy(format, density, dtype, n_rows):
    a = cupy.sparse.random(n_rows, 10, format=format, density=density, dtype=dtype)

    maj = 5
    min = [1, 5, 4]

    expected = a.todense()[maj, min].ravel()
    actual = a[maj, min].todense().ravel()

    cupy.testing.assert_array_equal(actual, expected)

@pytest.mark.parametrize('format', ['csr', 'csc']) #'csc'])
@pytest.mark.parametrize('density',  [0.1, 0.2, 0.5, 0.9, 1.0])
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('n_rows', [100, 1000])
def test_major_all_minor_fancy(format, density, dtype, n_rows):
    a = cupy.sparse.random(n_rows, 10, format=format, density=density, dtype=dtype)

    maj = slice(None)
    min = [1, 5, 4]

    expected = a.todense()[maj, min].ravel()
    actual = a[maj, min].todense().ravel()
    cupy.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('format', ['csr']) #'csc'])
@pytest.mark.parametrize('density', [0.1, 0.2, 0.5, 0.9, 1.0])
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('n_rows', [100, 1000])
def test_major_bool_fancy(format, density, dtype, n_rows):
    a = cupy.sparse.random(3, 10, format=format, density=density, dtype=dtype)

    maj = [True, False, True]

    expected = a.todense()[maj].ravel()
    actual = a[maj].todense().ravel()
    cupy.testing.assert_array_equal(actual, expected)


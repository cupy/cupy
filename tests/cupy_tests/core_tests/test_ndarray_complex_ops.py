import unittest

import numpy
import pytest

import cupy
from cupy import testing


@testing.gpu
class TestConj(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_almost_equal()
    def test_conj(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype)
        return x.conj()

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_array_almost_equal()
    def test_conj_pass(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype)
        y = x.conj()
        assert x is y
        return y

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_almost_equal()
    def test_conjugate(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype)
        return x.conjugate()

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_array_almost_equal()
    def test_conjugate_pass(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype)
        y = x.conjugate()
        assert x is y
        return y


@testing.gpu
class TestAngle(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_almost_equal()
    def test_angle(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype)
        return xp.angle(x)


@testing.gpu
class TestRealImag(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_almost_equal(accept_error=False)
    def test_real(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype)
        return x.real

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_almost_equal(accept_error=False)
    def test_real_zero_dim(self, xp, dtype):
        x = xp.array(1, dtype=dtype)
        return x.real

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_almost_equal(accept_error=False)
    def test_real_non_contiguous(self, xp, dtype):
        x = testing.shaped_arange((2, 3, 2), xp, dtype).transpose(0, 2, 1)
        return x.real

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_almost_equal(accept_error=False)
    def test_imag(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype)
        return x.imag

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_almost_equal(accept_error=False)
    def test_imag_zero_dim(self, xp, dtype):
        x = xp.array(1, dtype=dtype)
        return x.imag

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_almost_equal(accept_error=False)
    def test_imag_non_contiguous(self, xp, dtype):
        x = testing.shaped_arange((2, 3, 2), xp, dtype).transpose(0, 2, 1)
        return x.imag

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_almost_equal(accept_error=False)
    def test_real_setter(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype)
        x.real = testing.shaped_reverse_arange((2, 3), xp, dtype).real
        return x

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_almost_equal(accept_error=False)
    def test_real_setter_zero_dim(self, xp, dtype):
        x = xp.array(1, dtype=dtype)
        x.real = 2
        return x

    @testing.for_dtypes('FD')
    @testing.numpy_cupy_array_almost_equal(accept_error=False)
    def test_real_setter_non_contiguous(self, xp, dtype):
        x = testing.shaped_arange((2, 3, 2), xp, dtype).transpose(0, 2, 1)
        x.real = testing.shaped_reverse_arange((2, 2, 3), xp, dtype).real
        return x

    @testing.for_dtypes('FD')
    @testing.numpy_cupy_array_almost_equal(accept_error=False)
    def test_imag_setter(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype)
        x.imag = testing.shaped_reverse_arange((2, 3), xp, dtype).real
        return x

    @testing.for_dtypes('FD')
    @testing.numpy_cupy_array_almost_equal(accept_error=False)
    def test_imag_setter_zero_dim(self, xp, dtype):
        x = xp.array(1, dtype=dtype)
        x.imag = 2
        return x

    @testing.for_dtypes('FD')
    @testing.numpy_cupy_array_almost_equal(accept_error=False)
    def test_imag_setter_non_contiguous(self, xp, dtype):
        x = testing.shaped_arange((2, 3, 2), xp, dtype).transpose(0, 2, 1)
        x.imag = testing.shaped_reverse_arange((2, 2, 3), xp, dtype).real
        return x

    @testing.for_all_dtypes(no_complex=True)
    def test_imag_setter_raise(self, dtype):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((2, 3), xp, dtype)
            with pytest.raises(TypeError):
                x.imag = testing.shaped_reverse_arange((2, 3), xp, dtype)

    @testing.for_all_dtypes()
    def test_real_inplace(self, dtype):
        x = cupy.zeros((2, 3), dtype=dtype)
        x.real[:] = 1
        expected = cupy.ones((2, 3), dtype=dtype)
        assert cupy.all(x == expected)

    @testing.for_all_dtypes()
    def test_imag_inplace(self, dtype):
        x = cupy.zeros((2, 3), dtype=dtype)

        # TODO(kmaehashi) The following line should raise error for real
        # dtypes, but currently ignored silently.
        x.imag[:] = 1

        expected = cupy.zeros((2, 3), dtype=dtype) + (
            1j if x.dtype.kind == 'c' else 0)
        assert cupy.all(x == expected)


@testing.gpu
class TestScalarConversion(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_scalar_conversion(self, dtype):
        scalar = 1 + 1j if numpy.dtype(dtype).kind == 'c' else 1
        x_1d = cupy.array([scalar]).astype(dtype)
        assert complex(x_1d) == scalar

        x_0d = x_1d.reshape(())
        assert complex(x_0d) == scalar

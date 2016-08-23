import unittest

import numpy

from cupy import testing


@testing.gpu
class TestConj(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_dtypes('fdFD', name='dtype')
    @testing.numpy_cupy_array_almost_equal()
    def test_conj(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype)
        if numpy.dtype(dtype).kind == 'c':
            x += 1j * testing.shaped_reverse_arange((2, 3), xp, dtype)
        return x.conj()


@testing.gpu
class TestAngle(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_dtypes('FD', name='dtype')
    @testing.numpy_cupy_array_almost_equal()
    def test_conj(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype)
        if numpy.dtype(dtype).kind == 'c':
            x += 1j * testing.shaped_reverse_arange((2, 3), xp, dtype)
        return xp.angle(x)


@testing.gpu
class TestRealImag(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_dtypes('fdFD', name='dtype')
    @testing.numpy_cupy_array_almost_equal(accept_error=False)
    def test_real(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype)
        if numpy.dtype(dtype).kind == 'c':
            x += 1j * testing.shaped_reverse_arange((2, 3), xp, dtype)
        return x.real

    @testing.for_dtypes('fdFD', name='dtype')
    @testing.numpy_cupy_array_almost_equal(accept_error=False)
    def test_imag(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype)
        if numpy.dtype(dtype).kind == 'c':
            x += 1j * testing.shaped_reverse_arange((2, 3), xp, dtype)
        return x.imag

    @testing.for_dtypes('fdFD', name='dtype')
    @testing.numpy_cupy_array_almost_equal(accept_error=False)
    def test_real_setter(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype)
        if numpy.dtype(dtype).kind == 'c':
            x += 1j * testing.shaped_arange((2, 3), xp, dtype)
        x.real = testing.shaped_reverse_arange(
            (2, 3), xp, numpy.dtype(dtype).char.lower())
        return x

    @testing.for_dtypes('FD', name='dtype')
    @testing.numpy_cupy_array_almost_equal(accept_error=False)
    def test_imag_setter(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype)
        if numpy.dtype(dtype).kind == 'c':
            x += 1j * testing.shaped_arange((2, 3), xp, dtype)
        x.imag = testing.shaped_reverse_arange(
            (2, 3), xp, numpy.dtype(dtype).char.lower())
        return x

    @testing.for_dtypes('df', name='dtype')
    @testing.numpy_cupy_raises(exception_class=TypeError)
    def test_imag_setter_raise(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype)
        if numpy.dtype(dtype).kind == 'c':
            x += 1j * testing.shaped_arange((2, 3), xp, dtype)
        x.imag = testing.shaped_reverse_arange(
            (2, 3), xp, numpy.dtype(dtype).char.lower())
        return x

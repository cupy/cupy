import unittest

import numpy
import six

from cupy import testing

import cupy


@testing.gpu
class TestConj(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_dtypes('fdFD', name='dtype')
    @testing.numpy_cupy_array_almost_equal()
    def test_conj(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype)
        if dtype.kind == 'c':
            x += 1j * testing.shaped_reverse_arange((2, 3), xp, dtype)
        return x.conj()


@testing.gpu
class TestAngle(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_dtypes('FD', name='dtype')
    @testing.numpy_cupy_array_almost_equal()
    def test_conj(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype)
        if dtype.kind == 'c':
            x += 1j * testing.shaped_reverse_arange((2, 3), xp, dtype)
        return xp.angle(x)


@testing.gpu
class TestRealImag(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_dtypes('fdFD', name='dtype')
    @testing.numpy_cupy_array_almost_equal()
    def test_real(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype)
        if dtype.kind == 'c':
            x += 1j * testing.shaped_reverse_arange((2, 3), xp, dtype)
        return x.real

    @testing.for_dtypes('fdFD', name='dtype')
    @testing.numpy_cupy_array_almost_equal()
    def test_imag(self, xp, dtype):
        x = testing.shaped_arange((2, 3), xp, dtype)
        if dtype.kind == 'c':
            x += 1j * testing.shaped_reverse_arange((2, 3), xp, dtype)
        return x.imag

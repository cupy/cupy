import unittest

import pytest
import scipy.signal  # NOQA

import cupy
import cupyx
from cupy import testing


@testing.gpu
@testing.parameterize(*testing.product({
    'mode': ['valid', 'same', 'full']
}))
@testing.with_requires('scipy')
class TestChooseConvMethod(unittest.TestCase):

    @testing.for_dtypes('efdFD')
    @testing.numpy_cupy_equal(scipy_name='scp')
    def test_choose_conv_method1(self, xp, scp, dtype):
        a = testing.shaped_arange((10,), xp, dtype)
        b = testing.shaped_arange((5,), xp, dtype)
        return scp.signal.choose_conv_method(a, b, mode=self.mode)

    @testing.for_dtypes('efdFD')
    @testing.numpy_cupy_equal(scipy_name='scp')
    def test_choose_conv_method2(self, xp, scp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        b = testing.shaped_arange((10,), xp, dtype)
        return scp.signal.choose_conv_method(a, b, mode=self.mode)

    @testing.for_dtypes('efdFD')
    @testing.numpy_cupy_equal(scipy_name='scp')
    def test_choose_conv_method_same(self, xp, scp, dtype):
        a = testing.shaped_arange((10,), xp, dtype)
        return scp.signal.choose_conv_method(a, a, mode=self.mode)

    @testing.for_int_dtypes()
    def test_choose_conv_method_int(self, dtype):
        a = testing.shaped_arange((10,), cupy, dtype)
        b = testing.shaped_arange((5,), cupy, dtype)
        assert cupyx.scipy.signal.choose_conv_method(
            a, b, mode=self.mode) == 'direct'

    @testing.for_all_dtypes()
    def test_choose_conv_method_ndim(self, dtype):
        a = testing.shaped_arange((3, 4, 5), cupy, dtype)
        b = testing.shaped_arange((1, 2), cupy, dtype)
        with pytest.raises(NotImplementedError):
            cupyx.scipy.signal.choose_conv_method(a, b, mode=self.mode)

    @testing.for_all_dtypes()
    def test_choose_conv_method_zero_dim(self, dtype):
        a = testing.shaped_arange((), cupy, dtype)
        b = testing.shaped_arange((5,), cupy, dtype)
        with pytest.raises(NotImplementedError):
            cupyx.scipy.signal.choose_conv_method(a, b, mode=self.mode)

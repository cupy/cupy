import unittest

import pytest

import cupy
import cupyx
from cupy import testing


@testing.gpu
@testing.parameterize(*testing.product({
    'mode': ['valid', 'same', 'full']
}))
class TestChooseConvMethod(unittest.TestCase):

    @testing.for_dtypes('efdFD')
    def test_choose_conv_method1(self, dtype):
        a = testing.shaped_arange((10000,), cupy, dtype)
        b = testing.shaped_arange((5000,), cupy, dtype)
        assert cupyx.scipy.signal.choose_conv_method(
            a, b, mode=self.mode) == 'fft'

    @testing.for_dtypes('efdFD')
    def test_choose_conv_method2(self, dtype):
        a = testing.shaped_arange((5000,), cupy, dtype)
        b = testing.shaped_arange((10000,), cupy, dtype)
        assert cupyx.scipy.signal.choose_conv_method(
            a, b, mode=self.mode) == 'fft'

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

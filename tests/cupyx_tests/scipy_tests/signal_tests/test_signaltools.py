import unittest

import pytest
import scipy.signal  # NOQA

import cupy
import cupyx
from cupy import testing


@testing.gpu
@testing.parameterize(*testing.product({
    'mode': ['valid', 'same', 'full'],
    'shape': [((), (5,)), ((3, 4, 5), (1, 2))]
}))
@testing.with_requires('scipy')
class TestChooseConvMethod(unittest.TestCase):

    @testing.for_dtypes('efdFD')
    @testing.numpy_cupy_equal(scipy_name='scp')
    def test_choose_conv_method1(self, xp, scp, dtype):
        a = testing.shaped_arange((10,), xp, dtype)
        b = testing.shaped_arange((5,), xp, dtype)
        return scp.signal.choose_conv_method(a, b, mode=self.mode)

    @testing.for_dtypes('bBhHiIlLqQpP')
    def test_choose_conv_method2(self, dtype):
        a = testing.shaped_arange((10,), cupy, dtype)
        b = testing.shaped_arange((20,), cupy, dtype)
        assert cupyx.scipy.signal.choose_conv_method(
            a, b, mode=self.mode) == 'direct'

    @testing.for_all_dtypes()
    def test_choose_conv_method_ndim(self, dtype):
        shape_a, shape_b = self.shape
        a = testing.shaped_arange(shape_a, cupy, dtype)
        b = testing.shaped_arange(shape_b, cupy, dtype)
        with pytest.raises(NotImplementedError):
            cupyx.scipy.signal.choose_conv_method(a, b, mode=self.mode)

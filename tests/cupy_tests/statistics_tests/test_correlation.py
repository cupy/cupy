import unittest

import numpy
import pytest

import cupy
from cupy.cuda import runtime
from cupy import testing


@testing.gpu
class TestCorrcoef(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_corrcoef(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return xp.corrcoef(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_corrcoef_diag_exception(self, xp, dtype):
        a = testing.shaped_arange((1, 3), xp, dtype)
        return xp.corrcoef(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_corrcoef_y(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        y = testing.shaped_arange((2, 3), xp, dtype)
        return xp.corrcoef(a, y=y)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_corrcoef_rowvar(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        y = testing.shaped_arange((2, 3), xp, dtype)
        return xp.corrcoef(a, y=y, rowvar=False)


@testing.gpu
class TestCov(unittest.TestCase):

    def generate_input(self, a_shape, y_shape, xp, dtype):
        a = testing.shaped_arange(a_shape, xp, dtype)
        y = None
        if y_shape is not None:
            y = testing.shaped_arange(y_shape, xp, dtype)
        return a, y

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def check(self, a_shape, y_shape=None, rowvar=True, bias=False,
              ddof=None, xp=None, dtype=None):
        a, y = self.generate_input(a_shape, y_shape, xp, dtype)
        return xp.cov(a, y, rowvar, bias, ddof)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def check_warns(self, a_shape, y_shape=None, rowvar=True, bias=False,
                    ddof=None, xp=None, dtype=None):
        with testing.assert_warns(RuntimeWarning):
            a, y = self.generate_input(a_shape, y_shape, xp, dtype)
            return xp.cov(a, y, rowvar, bias, ddof)

    @testing.for_all_dtypes()
    def check_raises(self, a_shape, y_shape=None, rowvar=True, bias=False,
                     ddof=None, dtype=None):
        for xp in (numpy, cupy):
            a, y = self.generate_input(a_shape, y_shape, xp, dtype)
            with pytest.raises(ValueError):
                xp.cov(a, y, rowvar, bias, ddof)

    def test_cov(self):
        self.check((2, 3))
        self.check((2,), (2,))
        self.check((1, 3), (1, 3), rowvar=False)
        self.check((2, 3), (2, 3), rowvar=False)
        self.check((2, 3), bias=True)
        self.check((2, 3), ddof=2)

    def test_cov_warns(self):
        self.check_warns((2, 3), ddof=3)
        self.check_warns((2, 3), ddof=4)

    def test_cov_raises(self):
        self.check_raises((2, 3), ddof=1.2)
        self.check_raises((3, 4, 2))
        self.check_raises((2, 3), (3, 4, 2))

    def test_cov_empty(self):
        self.check((0, 1))


@testing.gpu
@testing.parameterize(*testing.product({
    'mode': ['valid', 'same', 'full'],
    'shape1': [(5,), (6,), (20,), (21,)],
    'shape2': [(5,), (6,), (20,), (21,)],
}))
class TestCorrelateShapeCombination(unittest.TestCase):

    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-4)
    def test_correlate(self, xp, dtype):
        a = testing.shaped_arange(self.shape1, xp, dtype)
        b = testing.shaped_arange(self.shape2, xp, dtype)
        return xp.correlate(a, b, mode=self.mode)


@testing.gpu
@testing.parameterize(*testing.product({
    'mode': ['valid', 'full', 'same']
}))
class TestCorrelate(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_correlate_non_contiguous(self, xp, dtype):
        if runtime.is_hip and self.mode == 'full':
            pytest.xfail(reason='ROCm/HIP may have a bug')
        a = testing.shaped_arange((300,), xp, dtype)
        b = testing.shaped_arange((100,), xp, dtype)
        return xp.correlate(a[::200], b[10::70], mode=self.mode)

    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-4)
    def test_correlate_large_non_contiguous(self, xp, dtype):
        a = testing.shaped_arange((10000,), xp, dtype)
        b = testing.shaped_arange((1000,), xp, dtype)
        return xp.correlate(a[200::], b[10::700], mode=self.mode)

    @testing.for_all_dtypes_combination(names=['dtype1', 'dtype2'])
    @testing.numpy_cupy_allclose(rtol=1e-2)
    @pytest.mark.xfail(runtime.is_hip, reason='ROCm/HIP may have a bug')
    def test_correlate_diff_types(self, xp, dtype1, dtype2):
        a = testing.shaped_random((200,), xp, dtype1)
        b = testing.shaped_random((100,), xp, dtype2)
        return xp.correlate(a, b, mode=self.mode)


@testing.gpu
@testing.parameterize(*testing.product({
    'mode': ['valid', 'same', 'full']
}))
class TestCorrelateInvalid(unittest.TestCase):

    @testing.with_requires('numpy>=1.18')
    @testing.for_all_dtypes()
    def test_correlate_empty(self, dtype):
        for xp in (numpy, cupy):
            a = xp.zeros((0,), dtype)
            with pytest.raises(ValueError):
                xp.correlate(a, a, mode=self.mode)

    @testing.for_all_dtypes()
    def test_correlate_ndim(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((5, 10, 2), xp, dtype)
            b = testing.shaped_arange((3, 4, 4), xp, dtype)
            with pytest.raises(ValueError):
                xp.correlate(a, b, mode=self.mode)

    @testing.for_all_dtypes()
    def test_correlate_zero_dim(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((), xp, dtype)
            b = testing.shaped_arange((1,), xp, dtype)
            with pytest.raises(ValueError):
                xp.correlate(a, b, mode=self.mode)

import unittest

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

    def _check(self, a_shape, y_shape=None, rowvar=True, bias=False, ddof=None,
               xp=None, dtype=None):
        a = testing.shaped_arange(a_shape, xp, dtype)
        y = None
        if y_shape is not None:
            y = testing.shaped_arange(y_shape, xp, dtype)
        return xp.cov(a, y, rowvar, bias, ddof)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def check(self, *args, **kw):
        return self._check(*args, **kw)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def check_warns(self, *args, **kw):
        with testing.assert_warns(RuntimeWarning):
            return self._check(*args, **kw)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_raises()
    def check_raises(self, *args, **kw):
        self._check(*args, **kw)

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

    @testing.with_requires('numpy>=1.10')
    def test_cov_empty(self):
        self.check((0, 1))

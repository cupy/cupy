import unittest

from cupy import testing


@testing.gpu
class TestMatrix(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.numpy_cupy_array_equal()
    def test_diag1(self, xp):
        a = testing.shaped_arange((3, 3), xp)
        return xp.diag(a)

    @testing.numpy_cupy_array_equal()
    def test_diag2(self, xp):
        a = testing.shaped_arange((3, 3), xp)
        return xp.diag(a, 1)

    @testing.numpy_cupy_array_equal()
    def test_diag3(self, xp):
        a = testing.shaped_arange((3, 3), xp)
        return xp.diag(a, -2)

    @testing.numpy_cupy_array_equal()
    def test_diagflat1(self, xp):
        a = testing.shaped_arange((3, 3), xp)
        return xp.diagflat(a)

    @testing.numpy_cupy_array_equal()
    def test_diagflat2(self, xp):
        a = testing.shaped_arange((3, 3), xp)
        return xp.diagflat(a, 1)

    @testing.numpy_cupy_array_equal()
    def test_diagflat3(self, xp):
        a = testing.shaped_arange((3, 3), xp)
        return xp.diagflat(a, -2)

    @testing.numpy_cupy_array_equal()
    def test_diagflat_from_scalar(self, xp):
        return xp.diagflat(3)

    @testing.numpy_cupy_array_equal()
    def test_diagflat_from_scalar_with_k0(self, xp):
        return xp.diagflat(3, 0)

    @testing.numpy_cupy_array_equal()
    def test_diagflat_from_scalar_with_k1(self, xp):
        return xp.diagflat(3, 1)


@testing.parameterize(
    {'shape': (2,)},
    {'shape': (3, 3)},
    {'shape': (4, 3)},
)
@testing.gpu
class TestTri(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_tri(self, xp, dtype):
        return xp.tri(*self.shape, k=0, dtype=dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_tri_nega(self, xp, dtype):
        return xp.tri(*self.shape, k=-1, dtype=dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_tri_posi(self, xp, dtype):
        return xp.tri(*self.shape, k=1, dtype=dtype)


@testing.parameterize(
    {'shape': (2,)},
    {'shape': (3, 3)},
    {'shape': (4, 3)},
    {'shape': (2, 3, 4)},
)
@testing.gpu
class TestTriLowerAndUpper(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_tril(self, xp, dtype):
        m = testing.shaped_arange(self.shape, xp, dtype)
        return xp.tril(m)

    @testing.numpy_cupy_array_equal()
    def test_tril_array_like(self, xp):
        return xp.tril([[1, 2], [3, 4]])

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_tril_nega(self, xp, dtype):
        m = testing.shaped_arange(self.shape, xp, dtype)
        return xp.tril(m, -1)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_tril_posi(self, xp, dtype):
        m = testing.shaped_arange(self.shape, xp, dtype)
        return xp.tril(m, 1)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_triu(self, xp, dtype):
        m = testing.shaped_arange(self.shape, xp, dtype)
        return xp.triu(m)

    @testing.numpy_cupy_array_equal()
    def test_triu_array_like(self, xp):
        return xp.triu([[1, 2], [3, 4]])

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_triu_nega(self, xp, dtype):
        m = testing.shaped_arange(self.shape, xp, dtype)
        return xp.triu(m, -1)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_triu_posi(self, xp, dtype):
        m = testing.shaped_arange(self.shape, xp, dtype)
        return xp.triu(m, 1)

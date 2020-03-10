import unittest

import numpy
import pytest

import cupy
from cupy import testing


@testing.gpu
class TestSort(unittest.TestCase):

    # Test ranks

    def test_sort_zero_dim(self):
        for xp in (numpy, cupy):
            a = testing.shaped_random((), xp)
            with pytest.raises(numpy.AxisError):
                a.sort()

    def test_external_sort_zero_dim(self):
        for xp in (numpy, numpy):
            a = testing.shaped_random((), xp)
            with pytest.raises(numpy.AxisError):
                xp.sort(a)

    @testing.numpy_cupy_array_equal()
    def test_sort_two_or_more_dim(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        a.sort()
        return a

    @testing.numpy_cupy_array_equal()
    def test_external_sort_two_or_more_dim(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        return xp.sort(a)

    # Test dtypes

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=False)
    @testing.numpy_cupy_allclose()
    def test_sort_dtype(self, xp, dtype):
        a = testing.shaped_random((10,), xp, dtype)
        a.sort()
        return a

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=False)
    @testing.numpy_cupy_allclose()
    def test_external_sort_dtype(self, xp, dtype):
        a = testing.shaped_random((10,), xp, dtype)
        return xp.sort(a)

    @testing.for_dtypes([numpy.float16, numpy.bool_])
    def test_sort_unsupported_dtype(self, dtype):
        a = testing.shaped_random((10,), cupy, dtype)
        with self.assertRaises(NotImplementedError):
            a.sort()

    @testing.for_dtypes([numpy.float16, numpy.bool_])
    def test_external_sort_unsupported_dtype(self, dtype):
        a = testing.shaped_random((10,), cupy, dtype)
        with self.assertRaises(NotImplementedError):
            return cupy.sort(a)

    # Test contiguous arrays

    @testing.numpy_cupy_allclose()
    def test_sort_contiguous(self, xp):
        a = testing.shaped_random((10,), xp)  # C contiguous view
        a.sort()
        return a

    def test_sort_non_contiguous(self):
        a = testing.shaped_random((10,), cupy)[::2]  # Non contiguous view
        with self.assertRaises(NotImplementedError):
            a.sort()

    @testing.numpy_cupy_allclose()
    def test_external_sort_contiguous(self, xp):
        a = testing.shaped_random((10,), xp)  # C contiguous view
        return xp.sort(a)

    @testing.numpy_cupy_allclose()
    def test_external_sort_non_contiguous(self, xp):
        a = testing.shaped_random((10,), xp)[::2]  # Non contiguous view
        return xp.sort(a)

    # Test axis

    @testing.numpy_cupy_array_equal()
    def test_sort_axis(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        a.sort(axis=0)
        return a

    @testing.numpy_cupy_array_equal()
    def test_external_sort_axis(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        return xp.sort(a, axis=0)

    @testing.numpy_cupy_array_equal()
    def test_sort_negative_axis(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        a.sort(axis=-2)
        return a

    @testing.numpy_cupy_array_equal()
    def test_external_sort_negative_axis(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        return xp.sort(a, axis=-2)

    @testing.numpy_cupy_array_equal()
    def test_external_sort_none_axis(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        return xp.sort(a, axis=None)

    @testing.numpy_cupy_raises()
    def test_sort_invalid_axis1(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        a.sort(axis=3)

    def test_sort_invalid_axis2(self):
        a = testing.shaped_random((2, 3, 3), cupy)
        with self.assertRaises(numpy.AxisError):
            a.sort(axis=3)

    @testing.numpy_cupy_raises()
    def test_external_sort_invalid_axis1(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        xp.sort(a, axis=3)

    def test_external_sort_invalid_axis2(self):
        a = testing.shaped_random((2, 3, 3), cupy)
        with self.assertRaises(numpy.AxisError):
            cupy.sort(a, axis=3)

    @testing.numpy_cupy_raises()
    def test_sort_invalid_negative_axis1(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        a.sort(axis=-4)

    def test_sort_invalid_negative_axis2(self):
        a = testing.shaped_random((2, 3, 3), cupy)
        with self.assertRaises(numpy.AxisError):
            a.sort(axis=-4)

    @testing.numpy_cupy_raises()
    def test_external_sort_invalid_negative_axis1(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        xp.sort(a, axis=-4)

    def test_external_sort_invalid_negative_axis2(self):
        a = testing.shaped_random((2, 3, 3), cupy)
        with self.assertRaises(numpy.AxisError):
            cupy.sort(a, axis=-4)


@testing.gpu
class TestLexsort(unittest.TestCase):

    # Test ranks

    # TODO(niboshi): Fix xfail
    @pytest.mark.xfail(reason='Explicit error types required')
    @testing.numpy_cupy_raises()
    def test_lexsort_zero_dim(self, xp):
        a = testing.shaped_random((), xp)
        return xp.lexsort(a)

    @testing.numpy_cupy_array_equal
    def test_lexsort_one_dim(self, xp):
        a = testing.shaped_random((2,), xp)
        return xp.lexsort(a)

    @testing.numpy_cupy_array_equal
    def test_lexsort_two_dim(self, xp):
        a = xp.array([[9, 4, 0, 4, 0, 2, 1],
                      [1, 5, 1, 4, 3, 4, 4]])  # from numpy.lexsort example
        return xp.lexsort(a)

    def test_lexsort_three_or_more_dim(self):
        a = testing.shaped_random((2, 10, 10), cupy)
        with self.assertRaises(NotImplementedError):
            return cupy.lexsort(a)

    # Test dtypes

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=False)
    @testing.numpy_cupy_allclose()
    def test_lexsort_dtype(self, xp, dtype):
        a = testing.shaped_random((2, 10), xp, dtype)
        return xp.lexsort(a)

    @testing.for_dtypes([numpy.float16, numpy.bool_])
    def test_lexsort_unsupported_dtype(self, dtype):
        a = testing.shaped_random((2, 10), cupy, dtype)
        with self.assertRaises(TypeError):
            return cupy.lexsort(a)


@testing.parameterize(*testing.product({
    'external': [False, True],
}))
@testing.gpu
class TestArgsort(unittest.TestCase):

    def argsort(self, a, axis=-1):
        if self.external:
            xp = cupy.get_array_module(a)
            return xp.argsort(a, axis=axis)
        else:
            return a.argsort(axis=axis)

    # Test base cases

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=False)
    @testing.numpy_cupy_array_equal()
    def test_argsort_zero_dim(self, xp, dtype):
        a = testing.shaped_random((), xp, dtype)
        return self.argsort(a)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=False)
    @testing.numpy_cupy_array_equal()
    def test_argsort_one_dim(self, xp, dtype):
        a = testing.shaped_random((10,), xp, dtype)
        return self.argsort(a)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=False)
    @testing.numpy_cupy_array_equal()
    def test_argsort_multi_dim(self, xp, dtype):
        a = testing.shaped_random((2, 3, 3), xp, dtype)
        return self.argsort(a)

    @testing.numpy_cupy_array_equal()
    def test_argsort_non_contiguous(self, xp):
        a = xp.array([1, 0, 2, 3])[::2]
        return self.argsort(a)

    # Test unsupported dtype

    @testing.for_dtypes([numpy.float16, numpy.bool_])
    def test_argsort_unsupported_dtype(self, dtype):
        a = testing.shaped_random((10,), cupy, dtype)
        with self.assertRaises(NotImplementedError):
            return self.argsort(a)

    # Test axis

    @testing.numpy_cupy_array_equal()
    def test_argsort_axis(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        return self.argsort(a, axis=0)

    @testing.numpy_cupy_array_equal()
    def test_argsort_negative_axis(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        return self.argsort(a, axis=2)

    @testing.numpy_cupy_array_equal()
    def test_argsort_none_axis(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        return self.argsort(a, axis=None)

    @testing.numpy_cupy_raises()
    def test_argsort_invalid_axis1(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        return self.argsort(a, axis=3)

    def test_argsort_invalid_axis2(self):
        a = testing.shaped_random((2, 3, 3), cupy)
        with self.assertRaises(numpy.AxisError):
            return self.argsort(a, axis=3)

    @testing.numpy_cupy_array_equal()
    def test_argsort_zero_dim_axis(self, xp):
        a = testing.shaped_random((), xp)
        return self.argsort(a, axis=0)

    def test_argsort_zero_dim_invalid_axis(self):
        for xp in (numpy, cupy):
            a = testing.shaped_random((), xp)
            with pytest.raises(numpy.AxisError):
                self.argsort(a, axis=1)

    @testing.numpy_cupy_raises()
    def test_argsort_invalid_negative_axis1(self, xp):
        a = testing.shaped_random((2, 3, 3), xp)
        return self.argsort(a, axis=-4)

    def test_argsort_invalid_negative_axis2(self):
        a = testing.shaped_random((2, 3, 3), cupy)
        with self.assertRaises(numpy.AxisError):
            return self.argsort(a, axis=-4)

    # Misc tests

    def test_argsort_original_array_not_modified_one_dim(self):
        a = testing.shaped_random((10,), cupy)
        b = cupy.array(a)
        self.argsort(a)
        testing.assert_allclose(a, b)

    def test_argsort_original_array_not_modified_multi_dim(self):
        a = testing.shaped_random((2, 3, 3), cupy)
        b = cupy.array(a)
        self.argsort(a)
        testing.assert_allclose(a, b)


@testing.gpu
class TestMsort(unittest.TestCase):

    # Test base cases

    # TODO(niboshi): Fix xfail
    @pytest.mark.xfail(reason='Explicit error types required')
    @testing.numpy_cupy_raises()
    def test_msort_zero_dim(self, xp):
        a = testing.shaped_random((), xp)
        return xp.msort(a)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=False)
    @testing.numpy_cupy_array_equal()
    def test_msort_one_dim(self, xp, dtype):
        a = testing.shaped_random((10,), xp, dtype)
        return xp.msort(a)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=False)
    @testing.numpy_cupy_array_equal()
    def test_msort_multi_dim(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype)
        return xp.msort(a)

    # Test unsupported dtype

    @testing.for_dtypes([numpy.float16, numpy.bool_])
    def test_msort_unsupported_dtype(self, dtype):
        a = testing.shaped_random((10,), cupy, dtype)
        with self.assertRaises(NotImplementedError):
            return cupy.msort(a)


@testing.parameterize(*testing.product({
    'external': [False, True],
    'length': [10, 20000],
}))
@testing.gpu
class TestPartition(unittest.TestCase):

    def partition(self, a, kth, axis=-1):
        if self.external:
            xp = cupy.get_array_module(a)
            return xp.partition(a, kth, axis=axis)
        else:
            a.partition(kth, axis=axis)
            return a

    # Test base cases

    def test_partition_zero_dim(self):
        for xp in (numpy, cupy):
            a = testing.shaped_random((), xp)
            kth = 2
            with pytest.raises(numpy.AxisError):
                self.partition(a, kth)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_equal()
    def test_partition_one_dim(self, xp, dtype):
        if self.length == 10 and dtype in [xp.float16, xp.bool_]:
            return cupy.zeros(1)  # dummy

        a = testing.shaped_random((self.length,), xp, dtype)
        kth = 2
        x = self.partition(a, kth)
        self.assertTrue(xp.all(x[0:kth] <= x[kth:kth + 1]))
        self.assertTrue(xp.all(x[kth:kth + 1] <= x[kth + 1:]))
        return x[kth]

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_partition_multi_dim(self, xp, dtype):
        if self.length == 10 and dtype in [xp.float16, xp.bool_]:
            return cupy.zeros(1)  # dummy

        a = testing.shaped_random((10, 10, self.length), xp, dtype)
        kth = 2
        x = self.partition(a, kth)
        self.assertTrue(xp.all(x[:, :, 0:kth] <= x[:, :, kth:kth + 1]))
        self.assertTrue(xp.all(x[:, :, kth:kth + 1] <= x[:, :, kth + 1:]))
        return x[:, :, kth:kth + 1]

    # Test unsupported dtype

    @testing.for_dtypes([numpy.float16, numpy.bool_, numpy.complex64,
                         numpy.complex128])
    def test_partition_unsupported_dtype(self, dtype):
        if self.length != 10 and not cupy.issubdtype(dtype, complex):
            return

        a = testing.shaped_random((self.length,), cupy, dtype)
        kth = 2
        with self.assertRaises(NotImplementedError):
            return self.partition(a, kth)

    # Test non-contiguous array

    @testing.numpy_cupy_equal()
    def test_partition_non_contiguous(self, xp):
        a = testing.shaped_random((self.length,), xp)[::-1]
        kth = 2
        if not self.external:
            if xp is cupy:
                with self.assertRaises(NotImplementedError):
                    return self.partition(a, kth)
            return 0  # dummy
        else:
            x = self.partition(a, kth)
            self.assertTrue(xp.all(x[0:kth] <= x[kth:kth + 1]))
            self.assertTrue(xp.all(x[kth:kth + 1] <= x[kth + 1:]))
            return x[kth]

    # Test kth

    @testing.numpy_cupy_equal()
    def test_partition_sequence_kth(self, xp):
        a = testing.shaped_random((self.length,), xp)
        kth = (2, 4)
        x = self.partition(a, kth)
        return x[kth[0]], x[kth[1]]

    @testing.numpy_cupy_equal()
    def test_partition_negative_kth(self, xp):
        a = testing.shaped_random((self.length,), xp)
        kth = -3
        x = self.partition(a, kth)
        return x[kth]

    @testing.numpy_cupy_raises()
    def test_partition_invalid_kth(self, xp):
        a = testing.shaped_random((self.length,), xp)
        kth = self.length
        return self.partition(a, kth)

    @testing.numpy_cupy_raises()
    def test_partition_invalid_negative_kth(self, xp):
        a = testing.shaped_random((self.length,), xp)
        kth = -self.length - 1
        return self.partition(a, kth)

    # Test axis

    @testing.numpy_cupy_array_equal()
    def test_partition_axis(self, xp):
        a = testing.shaped_random((self.length, 10, 10), xp)
        kth = 2
        axis = 0
        x = self.partition(a, kth, axis=axis)
        return x[kth, :, :]

    @testing.numpy_cupy_array_equal()
    def test_partition_negative_axis(self, xp):
        a = testing.shaped_random((10, 10, self.length), xp)
        kth = 2
        axis = -1
        x = self.partition(a, kth, axis=axis)
        return x[:, :, kth]

    @testing.numpy_cupy_equal()
    def test_partition_none_axis(self, xp):
        if self.external:
            a = testing.shaped_random((2, self.length), xp)
            kth = 2
            axis = None
            x = self.partition(a, kth, axis=axis)
            return x[kth]
        else:
            return None

    @testing.numpy_cupy_raises()
    def test_partition_invalid_axis1(self, xp):
        a = testing.shaped_random((2, 2, self.length), xp)
        kth = 2
        axis = 3
        return self.partition(a, kth, axis=axis)

    def test_partition_invalid_axis2(self):
        a = testing.shaped_random((2, 2, self.length), cupy)
        with self.assertRaises(numpy.AxisError):
            kth = 2
            axis = 3
            return self.partition(a, kth, axis=axis)

    @testing.numpy_cupy_raises()
    def test_partition_invalid_negative_axis1(self, xp):
        a = testing.shaped_random((2, 2, self.length), xp)
        kth = 2
        axis = -4
        return self.partition(a, kth, axis=axis)

    def test_partition_invalid_negative_axis2(self):
        a = testing.shaped_random((2, 2, self.length), cupy)
        with self.assertRaises(numpy.AxisError):
            kth = 2
            axis = -4
            return self.partition(a, kth, axis=axis)


@testing.parameterize(*testing.product({
    'external': [False, True],
}))
@testing.gpu
class TestArgpartition(unittest.TestCase):

    def argpartition(self, a, kth, axis=-1):
        if self.external:
            xp = cupy.get_array_module(a)
            return xp.argpartition(a, kth, axis=axis)
        else:
            return a.argpartition(kth, axis=axis)

    # Test base cases

    def test_argpartition_zero_dim(self):
        for xp in (numpy, cupy):
            a = testing.shaped_random((), xp)
            kth = 2
            with pytest.raises(ValueError):
                self.argpartition(a, kth)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_equal()
    def test_argpartition_one_dim(self, xp, dtype):
        a = testing.shaped_random((10,), xp, dtype, 100)
        kth = 2
        idx = self.argpartition(a, kth)
        self.assertTrue((a[idx[:kth]] < a[idx[kth]]).all())
        self.assertTrue((a[idx[kth]] < a[idx[kth + 1:]]).all())
        return idx[kth]

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_argpartition_multi_dim(self, xp, dtype):
        a = testing.shaped_random((3, 3, 10), xp, dtype, 100)
        kth = 2
        idx = self.argpartition(a, kth)
        rows = [[[0]], [[1]], [[2]]]
        cols = [[[0], [1], [2]]]
        self.assertTrue((a[rows, cols, idx[:, :, :kth]] <
                         a[rows, cols, idx[:, :, kth:kth + 1]]).all())
        self.assertTrue((a[rows, cols, idx[:, :, kth:kth + 1]] <
                         a[rows, cols, idx[:, :, kth + 1:]]).all())
        return idx[:, :, kth:kth + 1]

    # Test unsupported dtype

    @testing.for_dtypes([numpy.float16, numpy.bool_])
    def test_argpartition_unsupported_dtype(self, dtype):
        a = testing.shaped_random((10,), cupy, dtype, 100)
        kth = 2
        with self.assertRaises(NotImplementedError):
            return self.argpartition(a, kth)

    # Test non-contiguous array

    @testing.numpy_cupy_equal()
    def test_argpartition_non_contiguous(self, xp):
        a = testing.shaped_random((10,), xp, 'i', 100)[::2]
        kth = 2
        idx = self.argpartition(a, kth)
        self.assertTrue((a[idx[:kth]] < a[idx[kth]]).all())
        self.assertTrue((a[idx[kth]] < a[idx[kth + 1:]]).all())
        return idx[kth]

    # Test kth

    @testing.numpy_cupy_equal()
    def test_argpartition_sequence_kth(self, xp):
        a = testing.shaped_random((10,), xp, scale=100)
        kth = (2, 4)
        idx = self.argpartition(a, kth)
        for _kth in kth:
            self.assertTrue((a[idx[:_kth]] < a[idx[_kth]]).all())
            self.assertTrue((a[idx[_kth]] < a[idx[_kth + 1:]]).all())
        return (idx[2], idx[4])

    @testing.numpy_cupy_equal()
    def test_argpartition_negative_kth(self, xp):
        a = testing.shaped_random((10,), xp, scale=100)
        kth = -3
        idx = self.argpartition(a, kth)
        self.assertTrue((a[idx[:kth]] < a[idx[kth]]).all())
        self.assertTrue((a[idx[kth]] < a[idx[kth + 1:]]).all())
        return idx[kth]

    @testing.numpy_cupy_raises()
    def test_argpartition_invalid_kth(self, xp):
        a = testing.shaped_random((10,), xp, scale=100)
        kth = 10
        return self.argpartition(a, kth)

    @testing.numpy_cupy_raises()
    def test_argpartition_invalid_negative_kth(self, xp):
        a = testing.shaped_random((10,), xp, scale=100)
        kth = -11
        return self.argpartition(a, kth)

    # Test axis

    @testing.numpy_cupy_array_equal()
    def test_argpartition_axis(self, xp):
        a = testing.shaped_random((10, 3, 3), xp, scale=100)
        kth = 2
        axis = 0
        idx = self.argpartition(a, kth, axis=axis)
        rows = [[[0], [1], [2]]]
        cols = [[[0, 1, 2]]]
        self.assertTrue((a[idx[:kth, :, :], rows, cols] <
                         a[idx[kth:kth + 1, :, :], rows, cols]).all())
        self.assertTrue((a[idx[kth:kth + 1, :, :], rows, cols] <
                         a[idx[kth + 1:, :, :], rows, cols]).all())
        return idx[kth:kth + 1, :, :]

    @testing.numpy_cupy_array_equal()
    def test_argpartition_negative_axis(self, xp):
        a = testing.shaped_random((3, 3, 10), xp, scale=100)
        kth = 2
        axis = -1
        idx = self.argpartition(a, kth, axis=axis)
        rows = [[[0]], [[1]], [[2]]]
        cols = [[[0], [1], [2]]]
        self.assertTrue((a[rows, cols, idx[:, :, :kth]] <
                         a[rows, cols, idx[:, :, kth:kth + 1]]).all())
        self.assertTrue((a[rows, cols, idx[:, :, kth:kth + 1]] <
                         a[rows, cols, idx[:, :, kth + 1:]]).all())
        return idx[:, :, kth:kth + 1]

    @testing.numpy_cupy_equal()
    def test_argpartition_none_axis(self, xp):
        a = testing.shaped_random((2, 2), xp, scale=100)
        kth = 2
        axis = None
        idx = self.argpartition(a, kth, axis=axis)
        a1 = a.flatten()
        self.assertTrue((a1[idx[:kth]] < a1[idx[kth]]).all())
        self.assertTrue((a1[idx[kth]] < a1[idx[kth + 1:]]).all())
        return idx[kth]

    @testing.numpy_cupy_raises()
    def test_argpartition_invalid_axis1(self, xp):
        a = testing.shaped_random((2, 2, 2), xp, scale=100)
        kth = 1
        axis = 3
        return self.argpartition(a, kth, axis=axis)

    def test_argpartition_invalid_axis2(self):
        a = testing.shaped_random((2, 2, 2), cupy, scale=100)
        kth = 1
        axis = 3
        with self.assertRaises(numpy.AxisError):
            self.argpartition(a, kth, axis=axis)

    @testing.numpy_cupy_raises()
    def test_argpartition_invalid_negative_axis1(self, xp):
        a = testing.shaped_random((2, 2, 2), xp, scale=100)
        kth = 1
        axis = -4
        return self.argpartition(a, kth, axis=axis)

    def test_argpartition_invalid_negative_axis2(self):
        a = testing.shaped_random((2, 2, 2), cupy, scale=100)
        kth = 1
        axis = -4
        with self.assertRaises(numpy.AxisError):
            self.argpartition(a, kth, axis=axis)

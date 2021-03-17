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
        for xp in (numpy, cupy):
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

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_sort_dtype(self, xp, dtype):
        a = testing.shaped_random((10,), xp, dtype)
        a.sort()
        return a

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_external_sort_dtype(self, xp, dtype):
        a = testing.shaped_random((10,), xp, dtype)
        return xp.sort(a)

    # Test contiguous arrays

    @testing.numpy_cupy_array_equal()
    def test_sort_contiguous(self, xp):
        a = testing.shaped_random((10,), xp)  # C contiguous view
        a.sort()
        return a

    def test_sort_non_contiguous(self):
        a = testing.shaped_random((10,), cupy)[::2]  # Non contiguous view
        with self.assertRaises(NotImplementedError):
            a.sort()

    @testing.numpy_cupy_array_equal()
    def test_external_sort_contiguous(self, xp):
        a = testing.shaped_random((10,), xp)  # C contiguous view
        return xp.sort(a)

    @testing.numpy_cupy_array_equal()
    def test_external_sort_non_contiguous(self, xp):
        a = testing.shaped_random((10,), xp)[::2]  # Non contiguous view
        return xp.sort(a)

    # Test axis

    @testing.numpy_cupy_array_equal()
    def test_sort_axis1(self, xp):
        a = testing.shaped_random((2, 3, 4), xp)
        a.sort(axis=0)
        return a

    @testing.numpy_cupy_array_equal()
    def test_sort_axis2(self, xp):
        a = testing.shaped_random((2, 3, 4), xp)
        a.sort(axis=1)
        return a

    @testing.numpy_cupy_array_equal()
    def test_sort_axis3(self, xp):
        a = testing.shaped_random((2, 3, 4), xp)
        a.sort(axis=2)
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

    def test_sort_invalid_axis1(self):
        for xp in (numpy, cupy):
            a = testing.shaped_random((2, 3, 3), xp)
            with pytest.raises(numpy.AxisError):
                a.sort(axis=3)

    def test_sort_invalid_axis2(self):
        a = testing.shaped_random((2, 3, 3), cupy)
        with self.assertRaises(numpy.AxisError):
            a.sort(axis=3)

    def test_external_sort_invalid_axis1(self):
        for xp in (numpy, cupy):
            a = testing.shaped_random((2, 3, 3), xp)
            with pytest.raises(numpy.AxisError):
                xp.sort(a, axis=3)

    def test_external_sort_invalid_axis2(self):
        a = testing.shaped_random((2, 3, 3), cupy)
        with self.assertRaises(numpy.AxisError):
            cupy.sort(a, axis=3)

    def test_sort_invalid_negative_axis1(self):
        for xp in (numpy, cupy):
            a = testing.shaped_random((2, 3, 3), xp)
            with pytest.raises(numpy.AxisError):
                a.sort(axis=-4)

    def test_sort_invalid_negative_axis2(self):
        a = testing.shaped_random((2, 3, 3), cupy)
        with self.assertRaises(numpy.AxisError):
            a.sort(axis=-4)

    def test_external_sort_invalid_negative_axis1(self):
        for xp in (numpy, cupy):
            a = testing.shaped_random((2, 3, 3), xp)
            with pytest.raises(numpy.AxisError):
                xp.sort(a, axis=-4)

    def test_external_sort_invalid_negative_axis2(self):
        a = testing.shaped_random((2, 3, 3), cupy)
        with self.assertRaises(numpy.AxisError):
            cupy.sort(a, axis=-4)

    # Test NaN ordering

    @testing.for_dtypes('efdFD')
    @testing.numpy_cupy_array_equal()
    def test_nan1(self, xp, dtype):
        a = testing.shaped_random((10,), xp, dtype)
        a[2] = a[6] = xp.nan
        out = xp.sort(a)
        return out

    @testing.for_dtypes('efdFD')
    @testing.numpy_cupy_array_equal()
    def test_nan2(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        a[0, 2, 1] = a[1, 0, 3] = xp.nan
        out = xp.sort(a, axis=0)
        return out

    @testing.for_dtypes('efdFD')
    @testing.numpy_cupy_array_equal()
    def test_nan3(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        a[0, 2, 1] = a[1, 0, 3] = xp.nan
        out = xp.sort(a, axis=1)
        return out

    @testing.for_dtypes('efdFD')
    @testing.numpy_cupy_array_equal()
    def test_nan4(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        a[0, 2, 1] = a[1, 0, 3] = xp.nan
        out = xp.sort(a, axis=2)
        return out


@testing.gpu
class TestLexsort(unittest.TestCase):

    # Test ranks

    # TODO(niboshi): Fix xfail
    @pytest.mark.xfail(reason='Explicit error types required')
    def test_lexsort_zero_dim(self):
        for xp in (numpy, cupy):
            a = testing.shaped_random((), xp)
            with pytest.raises(numpy.AxisError):
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

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_lexsort_dtype(self, xp, dtype):
        a = testing.shaped_random((2, 10), xp, dtype)
        return xp.lexsort(a)

    # Test NaN ordering

    @testing.for_dtypes('efdFD')
    @testing.numpy_cupy_array_equal()
    def test_nan1(self, xp, dtype):
        a = testing.shaped_random((2, 10), xp, dtype)
        a[0, 2] = a[0, 6] = xp.nan
        return xp.lexsort(a)

    @testing.for_dtypes('efdFD')
    @testing.numpy_cupy_array_equal()
    def test_nan2(self, xp, dtype):
        a = testing.shaped_random((2, 10), xp, dtype)
        a[1, 2] = a[0, 6] = xp.nan
        return xp.lexsort(a)

    @testing.for_dtypes('efdFD')
    @testing.numpy_cupy_array_equal()
    def test_nan3(self, xp, dtype):
        a = testing.shaped_random((2, 10), xp, dtype)
        a[1, 2] = a[1, 6] = xp.nan
        return xp.lexsort(a)

    # Test non C-contiguous input

    @testing.numpy_cupy_array_equal()
    def test_view(self, xp):
        # from #3232
        a = testing.shaped_random((4, 8), xp, dtype=xp.float64)
        a = a.T[::-1]
        return xp.lexsort(a)

    @testing.numpy_cupy_array_equal()
    def test_F_order(self, xp):
        a = testing.shaped_random((4, 8), xp, dtype=xp.float64)
        a = xp.asfortranarray(a)
        assert a.flags.f_contiguous
        assert not a.flags.c_contiguous
        return xp.lexsort(a)


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

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_argsort_zero_dim(self, xp, dtype):
        a = testing.shaped_random((), xp, dtype)
        return self.argsort(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_argsort_one_dim(self, xp, dtype):
        a = testing.shaped_random((10,), xp, dtype)
        return self.argsort(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_argsort_multi_dim(self, xp, dtype):
        a = testing.shaped_random((2, 3, 3), xp, dtype)
        return self.argsort(a)

    @testing.numpy_cupy_array_equal()
    def test_argsort_non_contiguous(self, xp):
        a = xp.array([1, 0, 2, 3])[::2]
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

    def test_argsort_invalid_axis1(self):
        for xp in (numpy, cupy):
            a = testing.shaped_random((2, 3, 3), xp)
            with pytest.raises(numpy.AxisError):
                self.argsort(a, axis=3)

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

    def test_argsort_invalid_negative_axis1(self):
        for xp in (numpy, cupy):
            a = testing.shaped_random((2, 3, 3), xp)
            with pytest.raises(numpy.AxisError):
                self.argsort(a, axis=-4)

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

    # Test NaN ordering

    @testing.for_dtypes('efdFD')
    @testing.numpy_cupy_array_equal()
    def test_nan1(self, xp, dtype):
        a = testing.shaped_random((10,), xp, dtype)
        a[2] = a[6] = xp.nan
        return self.argsort(a)

    @testing.for_dtypes('efdFD')
    @testing.numpy_cupy_array_equal()
    def test_nan2(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        a[0, 2, 1] = a[1, 1, 3] = xp.nan
        return self.argsort(a)


@testing.gpu
class TestMsort(unittest.TestCase):

    # Test base cases

    def test_msort_zero_dim(self):
        for xp in (numpy, cupy):
            a = testing.shaped_random((), xp)
            with pytest.raises(numpy.AxisError):
                xp.msort(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_msort_one_dim(self, xp, dtype):
        a = testing.shaped_random((10,), xp, dtype)
        return xp.msort(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_msort_multi_dim(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype)
        return xp.msort(a)


@testing.gpu
class TestSort_complex(unittest.TestCase):

    def test_sort_complex_zero_dim(self):
        for xp in (numpy, cupy):
            a = testing.shaped_random((), xp)
            with pytest.raises(numpy.AxisError):
                xp.sort_complex(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_sort_complex_1dim(self, xp, dtype):
        a = testing.shaped_random((100,), xp, dtype)
        return a, xp.sort_complex(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_sort_complex_ndim(self, xp, dtype):
        a = testing.shaped_random((2, 5, 3), xp, dtype)
        return a, xp.sort_complex(a)

    @testing.for_dtypes('efdFD')
    @testing.numpy_cupy_array_equal()
    def test_sort_complex_nan(self, xp, dtype):
        a = testing.shaped_random((2, 3, 5), xp, dtype)
        a[0, 2, 1] = a[1, 0, 3] = xp.nan
        return a, xp.sort_complex(a)


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

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_partition_one_dim(self, xp, dtype):
        a = testing.shaped_random((self.length,), xp, dtype)
        kth = 2
        x = self.partition(a, kth)
        assert xp.all(x[0:kth] <= x[kth:kth + 1])
        assert xp.all(x[kth:kth + 1] <= x[kth + 1:])
        return x[kth]

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_partition_multi_dim(self, xp, dtype):
        a = testing.shaped_random((10, 10, self.length), xp, dtype)
        kth = 2
        x = self.partition(a, kth)
        assert xp.all(x[:, :, 0:kth] <= x[:, :, kth:kth + 1])
        assert xp.all(x[:, :, kth:kth + 1] <= x[:, :, kth + 1:])
        return x[:, :, kth:kth + 1]

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
            assert xp.all(x[0:kth] <= x[kth:kth + 1])
            assert xp.all(x[kth:kth + 1] <= x[kth + 1:])
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

    def test_partition_invalid_kth(self):
        for xp in (numpy, cupy):
            a = testing.shaped_random((self.length,), xp)
            kth = self.length
            with pytest.raises(ValueError):
                self.partition(a, kth)

    def test_partition_invalid_negative_kth(self):
        for xp in (numpy, cupy):
            a = testing.shaped_random((self.length,), xp)
            kth = -self.length - 1
            with pytest.raises(ValueError):
                self.partition(a, kth)

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

    def test_partition_invalid_axis1(self):
        for xp in (numpy, cupy):
            a = testing.shaped_random((2, 2, self.length), xp)
            kth = 2
            axis = 3
            with pytest.raises(numpy.AxisError):
                self.partition(a, kth, axis=axis)

    def test_partition_invalid_axis2(self):
        a = testing.shaped_random((2, 2, self.length), cupy)
        with self.assertRaises(numpy.AxisError):
            kth = 2
            axis = 3
            return self.partition(a, kth, axis=axis)

    def test_partition_invalid_negative_axis1(self):
        for xp in (numpy, cupy):
            a = testing.shaped_random((2, 2, self.length), xp)
            kth = 2
            axis = -4
            with pytest.raises(numpy.AxisError):
                self.partition(a, kth, axis=axis)

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

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_argpartition_one_dim(self, xp, dtype):
        a = testing.shaped_random((10,), xp, dtype, 100)
        kth = 2
        idx = self.argpartition(a, kth)
        assert (a[idx[:kth]] <= a[idx[kth]]).all()
        assert (a[idx[kth]] <= a[idx[kth + 1:]]).all()
        return a[idx[kth]]

    # TODO(leofang): test all dtypes -- this workaround needs to be kept,
    # likely due to #3287? Need investigation.
    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_argpartition_multi_dim(self, xp, dtype):
        a = testing.shaped_random((3, 3, 10), xp, dtype, 100)
        kth = 2
        idx = self.argpartition(a, kth)
        rows = [[[0]], [[1]], [[2]]]
        cols = [[[0], [1], [2]]]
        assert (a[rows, cols, idx[:, :, :kth]] <
                a[rows, cols, idx[:, :, kth:kth + 1]]).all()
        assert (a[rows, cols, idx[:, :, kth:kth + 1]] <
                a[rows, cols, idx[:, :, kth + 1:]]).all()
        return idx[:, :, kth:kth + 1]

    # Test non-contiguous array

    @testing.numpy_cupy_equal()
    def test_argpartition_non_contiguous(self, xp):
        a = testing.shaped_random((10,), xp, 'i', 100)[::2]
        kth = 2
        idx = self.argpartition(a, kth)
        assert (a[idx[:kth]] < a[idx[kth]]).all()
        assert (a[idx[kth]] < a[idx[kth + 1:]]).all()
        return idx[kth]

    # Test kth

    @testing.numpy_cupy_equal()
    def test_argpartition_sequence_kth(self, xp):
        a = testing.shaped_random((10,), xp, scale=100)
        kth = (2, 4)
        idx = self.argpartition(a, kth)
        for _kth in kth:
            assert (a[idx[:_kth]] < a[idx[_kth]]).all()
            assert (a[idx[_kth]] < a[idx[_kth + 1:]]).all()
        return (idx[2], idx[4])

    @testing.numpy_cupy_equal()
    def test_argpartition_negative_kth(self, xp):
        a = testing.shaped_random((10,), xp, scale=100)
        kth = -3
        idx = self.argpartition(a, kth)
        assert (a[idx[:kth]] < a[idx[kth]]).all()
        assert (a[idx[kth]] < a[idx[kth + 1:]]).all()
        return idx[kth]

    def test_argpartition_invalid_kth(self):
        for xp in (numpy, cupy):
            a = testing.shaped_random((10,), xp, scale=100)
            kth = 10
            with pytest.raises(ValueError):
                self.argpartition(a, kth)

    def test_argpartition_invalid_negative_kth(self):
        for xp in (numpy, cupy):
            a = testing.shaped_random((10,), xp, scale=100)
            kth = -11
            with pytest.raises(ValueError):
                self.argpartition(a, kth)

    # Test axis

    @testing.numpy_cupy_array_equal()
    def test_argpartition_axis(self, xp):
        a = testing.shaped_random((10, 3, 3), xp, scale=100)
        kth = 2
        axis = 0
        idx = self.argpartition(a, kth, axis=axis)
        rows = [[[0], [1], [2]]]
        cols = [[[0, 1, 2]]]
        assert (a[idx[:kth, :, :], rows, cols] <
                a[idx[kth:kth + 1, :, :], rows, cols]).all()
        assert (a[idx[kth:kth + 1, :, :], rows, cols] <
                a[idx[kth + 1:, :, :], rows, cols]).all()
        return idx[kth:kth + 1, :, :]

    @testing.numpy_cupy_array_equal()
    def test_argpartition_negative_axis(self, xp):
        a = testing.shaped_random((3, 3, 10), xp, scale=100)
        kth = 2
        axis = -1
        idx = self.argpartition(a, kth, axis=axis)
        rows = [[[0]], [[1]], [[2]]]
        cols = [[[0], [1], [2]]]
        assert (a[rows, cols, idx[:, :, :kth]] <
                a[rows, cols, idx[:, :, kth:kth + 1]]).all()
        assert (a[rows, cols, idx[:, :, kth:kth + 1]] <
                a[rows, cols, idx[:, :, kth + 1:]]).all()
        return idx[:, :, kth:kth + 1]

    @testing.numpy_cupy_equal()
    def test_argpartition_none_axis(self, xp):
        a = testing.shaped_random((2, 2), xp, scale=100)
        kth = 2
        axis = None
        idx = self.argpartition(a, kth, axis=axis)
        a1 = a.flatten()
        assert (a1[idx[:kth]] < a1[idx[kth]]).all()
        assert (a1[idx[kth]] < a1[idx[kth + 1:]]).all()
        return idx[kth]

    def test_argpartition_invalid_axis1(self):
        for xp in (numpy, cupy):
            a = testing.shaped_random((2, 2, 2), xp, scale=100)
            kth = 1
            axis = 3
            with pytest.raises(numpy.AxisError):
                self.argpartition(a, kth, axis=axis)

    def test_argpartition_invalid_axis2(self):
        a = testing.shaped_random((2, 2, 2), cupy, scale=100)
        kth = 1
        axis = 3
        with self.assertRaises(numpy.AxisError):
            self.argpartition(a, kth, axis=axis)

    def test_argpartition_invalid_negative_axis1(self):
        for xp in (numpy, cupy):
            a = testing.shaped_random((2, 2, 2), xp, scale=100)
            kth = 1
            axis = -4
            with pytest.raises(numpy.AxisError):
                self.argpartition(a, kth, axis=axis)

    def test_argpartition_invalid_negative_axis2(self):
        a = testing.shaped_random((2, 2, 2), cupy, scale=100)
        kth = 1
        axis = -4
        with self.assertRaises(numpy.AxisError):
            self.argpartition(a, kth, axis=axis)

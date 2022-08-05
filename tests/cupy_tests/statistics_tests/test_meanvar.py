import math

import numpy
import pytest

import cupy
from cupy import testing

ignore_runtime_warnings = pytest.mark.filterwarnings(
    "ignore", category=RuntimeWarning)


@testing.gpu
class TestMedian:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_median_noaxis(self, xp, dtype):
        a = testing.shaped_random((3, 4, 5), xp, dtype)
        return xp.median(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_median_axis1(self, xp, dtype):
        a = testing.shaped_random((3, 4, 5), xp, dtype)
        return xp.median(a, axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_median_axis2(self, xp, dtype):
        a = testing.shaped_random((3, 4, 5), xp, dtype)
        return xp.median(a, axis=2)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_median_overwrite_input(self, xp, dtype):
        a = testing.shaped_random((3, 4, 5), xp, dtype)
        return xp.median(a, overwrite_input=True)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_median_keepdims_axis1(self, xp, dtype):
        a = testing.shaped_random((3, 4, 5), xp, dtype)
        return xp.median(a, axis=1, keepdims=True)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_median_keepdims_noaxis(self, xp, dtype):
        a = testing.shaped_random((3, 4, 5), xp, dtype)
        return xp.median(a, keepdims=True)

    def test_median_invalid_axis(self):
        for xp in [numpy, cupy]:
            a = testing.shaped_random((3, 4, 5), xp)
            with pytest.raises(numpy.AxisError):
                return xp.median(a, -a.ndim - 1, keepdims=False)

            with pytest.raises(numpy.AxisError):
                return xp.median(a, a.ndim, keepdims=False)

            with pytest.raises(numpy.AxisError):
                return xp.median(a, (-a.ndim - 1, 1), keepdims=False)

            with pytest.raises(numpy.AxisError):
                return xp.median(a, (0, a.ndim,), keepdims=False)

    @testing.for_dtypes('efdFD')
    @testing.numpy_cupy_allclose()
    def test_median_nan(self, xp, dtype):
        a = xp.array(
            [[xp.nan, 1, 2, 3],
             [4, 5, 6, 7],
             [8, 9, 10, xp.nan]],
            dtype=dtype,
        )
        return xp.median(a, axis=1)


@testing.parameterize(
    *testing.product({
        'shape': [(3, 4, 5)],
        'axis': [(0, 1), (0, -1), (1, 2), (1,)],
        'keepdims': [True, False]
    })
)
@testing.gpu
class TestMedianAxis:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_median_axis_sequence(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        return xp.median(a, self.axis, keepdims=self.keepdims)


@testing.parameterize(
    *testing.product({
        'shape': [(3, 4, 5)],
        'axis': [None, 0, 1, -1, (0, 1), (0, 2), (-1, -2), [0, 1]],
        'keepdims': [True, False],
        'overwrite_input': [True, False]
    })
)
@testing.gpu
class TestNanMedian:

    zero_density = 0.25

    def _make_array(self, dtype):
        dtype = numpy.dtype(dtype)
        if dtype.char in 'efdFD':
            r_dtype = dtype.char.lower()
            a = testing.shaped_random(self.shape, numpy, dtype=r_dtype,
                                      scale=1)
            if dtype.char in 'FD':
                ai = a
                aj = testing.shaped_random(self.shape, numpy, dtype=r_dtype,
                                           scale=1)
                ai[ai < math.sqrt(self.zero_density)] = 0
                aj[aj < math.sqrt(self.zero_density)] = 0
                a = ai + 1j * aj
            else:
                a[a < self.zero_density] = 0
            a = a / a
        else:
            a = testing.shaped_random(self.shape, numpy, dtype=dtype)
        return a

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nanmedian(self, xp, dtype):
        a = xp.array(self._make_array(dtype))
        out = xp.nanmedian(a, self.axis, keepdims=self.keepdims,
                           overwrite_input=self.overwrite_input)
        return xp.ascontiguousarray(out)


@testing.gpu
class TestAverage:

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_average_all(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return xp.average(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_average_axis(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.average(a, axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_average_weights(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        w = testing.shaped_arange((2, 3), xp, dtype)
        return xp.average(a, weights=w)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    @pytest.mark.parametrize(
        'axis,weights', [(1, False), (None, True), (1, True)])
    def test_returned(self, xp, dtype, axis, weights):
        a = testing.shaped_arange((2, 3), numpy, dtype)
        if weights:
            w = testing.shaped_arange((2, 3), numpy, dtype)
        else:
            w = None
        return xp.average(a, axis=axis, weights=w, returned=True)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol={'default': 5e-7})
    @pytest.mark.parametrize('returned', [True, False])
    @testing.with_requires('numpy>=1.23.1')
    def test_average_keepdims_axis1(self, xp, dtype, returned):
        a = testing.shaped_random((2, 3), xp, dtype)
        w = testing.shaped_random((2, 3), xp, dtype)
        return xp.average(
            a, axis=1, weights=w, returned=returned, keepdims=True)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol={'default': 1e-7, numpy.float16: 1e-3})
    @pytest.mark.parametrize('returned', [True, False])
    @testing.with_requires('numpy>=1.23.1')
    def test_average_keepdims_noaxis(self, xp, dtype, returned):
        a = testing.shaped_random((2, 3), xp, dtype)
        w = testing.shaped_random((2, 3), xp, dtype)
        return xp.average(a, weights=w, returned=returned, keepdims=True)


@testing.gpu
class TestMeanVar:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_mean_all(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return a.mean()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_external_mean_all(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return xp.mean(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_mean_axis(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return a.mean(axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_external_mean_axis(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.mean(a, axis=1)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_mean_all_float64_dtype(self, xp, dtype):
        a = xp.full((2, 3, 4), 123456789, dtype=dtype)
        return xp.mean(a, dtype=numpy.float64)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_mean_all_int64_dtype(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.mean(a, dtype=numpy.int64)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_mean_all_complex_dtype(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.mean(a, dtype=numpy.complex64)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_var_all(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return a.var()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_external_var_all(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return xp.var(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_var_all_ddof(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return a.var(ddof=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_external_var_all_ddof(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return xp.var(a, ddof=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_var_axis(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return a.var(axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_external_var_axis(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.var(a, axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_var_axis_ddof(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return a.var(axis=1, ddof=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_external_var_axis_ddof(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.var(a, axis=1, ddof=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_std_all(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return a.std()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_external_std_all(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return xp.std(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_std_all_ddof(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return a.std(ddof=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_external_std_all_ddof(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return xp.std(a, ddof=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_std_axis(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return a.std(axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_external_std_axis(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.std(a, axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_std_axis_ddof(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return a.std(axis=1, ddof=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_external_std_axis_ddof(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.std(a, axis=1, ddof=1)


@testing.parameterize(
    *testing.product({
        'shape': [(3, 4), (30, 40, 50)],
        'axis': [None, 0, 1],
        'keepdims': [True, False]
    })
)
@testing.gpu
class TestNanMean:

    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_nanmean_without_nan(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        return xp.nanmean(a, axis=self.axis, keepdims=self.keepdims)

    @ignore_runtime_warnings
    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_nanmean_with_nan_float(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)

        if a.dtype.kind not in 'biu':
            a[1, :] = xp.nan
            a[:, 3] = xp.nan

        return xp.nanmean(a, axis=self.axis, keepdims=self.keepdims)


@testing.gpu
class TestNanMeanAdditional:

    @ignore_runtime_warnings
    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_nanmean_out(self, xp, dtype):
        a = testing.shaped_random((10, 20, 30), xp, dtype)
        z = xp.zeros((20, 30), dtype=dtype)

        if a.dtype.kind not in 'biu':
            a[1, :] = xp.nan
            a[:, 3] = xp.nan

        xp.nanmean(a, axis=0, out=z)
        return z

    @testing.slow
    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_nanmean_huge(self, xp, dtype):
        a = testing.shaped_random((1024, 512), xp, dtype)

        if a.dtype.kind not in 'biu':
            a[:512, :256] = xp.nan

        return xp.nanmean(a, axis=1)

    @testing.numpy_cupy_allclose(rtol=1e-4)
    def test_nanmean_float16(self, xp):
        a = testing.shaped_arange((2, 3), xp, numpy.float16)
        a[0][0] = xp.nan
        return xp.nanmean(a)

    @ignore_runtime_warnings
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_nanmean_all_nan(self, xp):
        a = xp.zeros((3, 4))
        a[:] = xp.nan
        return xp.nanmean(a)


@testing.parameterize(
    *testing.product({
        'shape': [(3, 4), (4, 3, 5)],
        'axis': [None, 0, 1],
        'keepdims': [True, False],
        'ddof': [0, 1]
    }))
class TestNanVarStd:

    @ignore_runtime_warnings
    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_nanvar(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype=dtype)
        if a.dtype.kind not in 'biu':
            a[0, :] = xp.nan
        return xp.nanvar(
            a, axis=self.axis, ddof=self.ddof, keepdims=self.keepdims)

    @ignore_runtime_warnings
    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_nanstd(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype=dtype)
        if a.dtype.kind not in 'biu':
            a[0, :] = xp.nan
        return xp.nanstd(
            a, axis=self.axis, ddof=self.ddof, keepdims=self.keepdims)


class TestNanVarStdAdditional:

    @ignore_runtime_warnings
    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_nanvar_out(self, xp, dtype):
        a = testing.shaped_random((10, 20, 30), xp, dtype)
        z = xp.zeros((20, 30))

        if a.dtype.kind not in 'biu':
            a[1, :] = xp.nan
            a[:, 3] = xp.nan

        xp.nanvar(a, axis=0, out=z)
        return z

    @testing.slow
    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_nanvar_huge(self, xp, dtype):
        a = testing.shaped_random((1024, 512), xp, dtype)

        if a.dtype.kind not in 'biu':
            a[:512, :256] = xp.nan

        return xp.nanvar(a, axis=1)

    @testing.numpy_cupy_allclose(rtol=1e-4)
    def test_nanvar_float16(self, xp):
        a = testing.shaped_arange((4, 5), xp, numpy.float16)
        a[0][0] = xp.nan
        return xp.nanvar(a, axis=0)

    @ignore_runtime_warnings
    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_nanstd_out(self, xp, dtype):
        a = testing.shaped_random((10, 20, 30), xp, dtype)
        z = xp.zeros((20, 30))

        if a.dtype.kind not in 'biu':
            a[1, :] = xp.nan
            a[:, 3] = xp.nan

        xp.nanstd(a, axis=0, out=z)
        return z

    @testing.slow
    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_nanstd_huge(self, xp, dtype):
        a = testing.shaped_random((1024, 512), xp, dtype)

        if a.dtype.kind not in 'biu':
            a[:512, :256] = xp.nan

        return xp.nanstd(a, axis=1)

    @testing.numpy_cupy_allclose(rtol=1e-4)
    def test_nanstd_float16(self, xp):
        a = testing.shaped_arange((4, 5), xp, numpy.float16)
        a[0][0] = xp.nan
        return xp.nanstd(a, axis=1)


@testing.parameterize(*testing.product({
    'params': [
        ((), None),
        ((0,), None),
        ((0, 0), None),
        ((0, 0), 1),
        ((0, 0, 0), None),
        ((0, 0, 0), (0, 2)),
    ],
    'func': ['mean', 'std', 'var'],
}))
@testing.gpu
class TestProductZeroLength:

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_external_mean_zero_len(self, xp, dtype):
        shape, axis = self.params
        a = testing.shaped_arange(shape, xp, dtype)
        f = getattr(xp, self.func)
        return f(a, axis=axis)

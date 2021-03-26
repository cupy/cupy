import functools
import math
import sys
import unittest

import numpy
import pytest

import cupy
from cupy import testing


def skip_int_equality_before_numpy_1_20(names=('dtype',)):
    """Require numpy/numpy#16841 or skip the equality check."""
    def decorator(wrapped):
        if numpy.lib.NumpyVersion(numpy.__version__) >= '1.20.0':
            return wrapped

        @functools.wraps(wrapped)
        def wrapper(self, *args, **kwargs):
            xp = kwargs['xp']
            dtypes = [kwargs[name] for name in names]
            ret = wrapped(self, *args, **kwargs)
            if any(numpy.issubdtype(dtype, numpy.integer) for dtype in dtypes):
                ret = xp.zeros_like(ret)
            return ret

        return wrapper

    return decorator


@testing.gpu
class TestRanges(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_arange(self, xp, dtype):
        return xp.arange(10, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_arange2(self, xp, dtype):
        return xp.arange(5, 10, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_arange3(self, xp, dtype):
        return xp.arange(1, 11, 2, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_arange4(self, xp, dtype):
        return xp.arange(20, 2, -3, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_arange5(self, xp, dtype):
        return xp.arange(0, 100, None, dtype=dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_arange6(self, xp, dtype):
        return xp.arange(0, 2, dtype=dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_arange7(self, xp, dtype):
        return xp.arange(10, 11, dtype=dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_arange8(self, xp, dtype):
        return xp.arange(10, 8, -1, dtype=dtype)

    def test_arange9(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.arange(10, dtype=xp.bool_)

    @testing.numpy_cupy_array_equal()
    def test_arange_no_dtype_int(self, xp):
        return xp.arange(1, 11, 2)

    @testing.numpy_cupy_array_equal()
    def test_arange_no_dtype_float(self, xp):
        return xp.arange(1.0, 11.0, 2.0)

    @testing.numpy_cupy_array_equal()
    def test_arange_negative_size(self, xp):
        return xp.arange(3, 1)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_linspace(self, xp, dtype):
        return xp.linspace(0, 10, 5, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_linspace2(self, xp, dtype):
        return xp.linspace(10, 0, 5, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    @skip_int_equality_before_numpy_1_20()
    def test_linspace3(self, xp, dtype):
        if xp.dtype(dtype).kind == 'u':
            pytest.skip()
        return xp.linspace(-10, 8, 9, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_linspace_zero_num(self, xp, dtype):
        return xp.linspace(0, 10, 0, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_linspace_zero_num_no_endopoint_with_retstep(self, xp, dtype):
        x, step = xp.linspace(0, 10, 0, dtype=dtype, endpoint=False,
                              retstep=True)
        assert math.isnan(step)
        return x

    @testing.with_requires('numpy>=1.18')
    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_linspace_one_num_no_endopoint_with_retstep(self, xp, dtype):
        start, stop = 3, 7
        x, step = xp.linspace(start, stop, 1, dtype=dtype, endpoint=False,
                              retstep=True)
        assert step == stop - start
        return x

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_linspace_one_num(self, xp, dtype):
        return xp.linspace(0, 2, 1, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_linspace_no_endpoint(self, xp, dtype):
        return xp.linspace(0, 10, 5, dtype=dtype, endpoint=False)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_linspace_with_retstep(self, xp, dtype):
        x, step = xp.linspace(0, 10, 5, dtype=dtype, retstep=True)
        assert step == 2.5
        return x

    @testing.numpy_cupy_allclose()
    def test_linspace_no_dtype_int(self, xp):
        return xp.linspace(0, 10)

    @testing.numpy_cupy_allclose()
    def test_linspace_no_dtype_float(self, xp):
        return xp.linspace(0.0, 10.0)

    @testing.numpy_cupy_allclose()
    def test_linspace_float_args_with_int_dtype(self, xp):
        return xp.linspace(0.1, 9.1, 11, dtype=int)

    def test_linspace_neg_num(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.linspace(0, 10, -1)

    @testing.numpy_cupy_allclose()
    def test_linspace_float_overflow(self, xp):
        return xp.linspace(0., sys.float_info.max / 5, 10, dtype=float)

    @testing.numpy_cupy_array_equal()
    def test_linspace_float_underflow(self, xp):
        # find minimum subnormal number
        x = sys.float_info.min
        while x / 2 > 0:
            x /= 2
        return xp.linspace(0., x, 10, dtype=float)

    @testing.with_requires('numpy>=1.16')
    @testing.for_all_dtypes_combination(names=('dtype_range', 'dtype_out'),
                                        no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_linspace_array_start_stop(self, xp, dtype_range, dtype_out):
        start = xp.array([0, 120], dtype=dtype_range)
        stop = xp.array([100, 0], dtype=dtype_range)
        return xp.linspace(start, stop, num=50, dtype=dtype_out)

    @testing.with_requires('numpy>=1.16')
    @testing.for_all_dtypes_combination(names=('dtype_range', 'dtype_out'),
                                        no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    @skip_int_equality_before_numpy_1_20(names=('dtype_range', 'dtype_out'))
    def test_linspace_mixed_start_stop(self, xp, dtype_range, dtype_out):
        start = 0.0
        if xp.dtype(dtype_range).kind in 'u':
            stop = xp.array([100, 16], dtype=dtype_range)
        else:
            stop = xp.array([100, -100], dtype=dtype_range)
        return xp.linspace(start, stop, num=50, dtype=dtype_out)

    @testing.with_requires('numpy>=1.16')
    @testing.for_all_dtypes_combination(names=('dtype_range', 'dtype_out'),
                                        no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    @skip_int_equality_before_numpy_1_20(names=('dtype_range', 'dtype_out'))
    def test_linspace_mixed_start_stop2(self, xp, dtype_range, dtype_out):
        if xp.dtype(dtype_range).kind in 'u':
            start = xp.array([160, 120], dtype=dtype_range)
        else:
            start = xp.array([-120, 120], dtype=dtype_range)
        stop = 0
        return xp.linspace(start, stop, num=50, dtype=dtype_out)

    @testing.with_requires('numpy>=1.16')
    @testing.for_all_dtypes_combination(names=('dtype_range', 'dtype_out'),
                                        no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_linspace_array_start_stop_axis1(self, xp, dtype_range, dtype_out):
        start = xp.array([0, 120], dtype=dtype_range)
        stop = xp.array([100, 0], dtype=dtype_range)
        return xp.linspace(start, stop, num=50, dtype=dtype_out, axis=1)

    @testing.with_requires('numpy>=1.16')
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_linspace_complex_start_stop(self, xp, dtype):
        start = xp.array([0, 120], dtype=dtype)
        stop = xp.array([100, 0], dtype=dtype)
        return xp.linspace(start, stop, num=50, dtype=dtype)

    @testing.with_requires('numpy>=1.16')
    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_linspace_start_stop_list(self, xp, dtype):
        start = [0, 0]
        stop = [100, 16]
        return xp.linspace(start, stop, num=50, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_logspace(self, xp, dtype):
        return xp.logspace(0, 2, 5, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_logspace2(self, xp, dtype):
        return xp.logspace(2, 0, 5, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_logspace_zero_num(self, xp, dtype):
        return xp.logspace(0, 2, 0, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_logspace_one_num(self, xp, dtype):
        return xp.logspace(0, 2, 1, dtype=dtype)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_logspace_no_endpoint(self, xp, dtype):
        return xp.logspace(0, 2, 5, dtype=dtype, endpoint=False)

    @testing.numpy_cupy_allclose()
    def test_logspace_no_dtype_int(self, xp):
        return xp.logspace(0, 2)

    @testing.numpy_cupy_allclose()
    def test_logspace_no_dtype_float(self, xp):
        return xp.logspace(0.0, 2.0)

    @testing.numpy_cupy_allclose()
    def test_logspace_float_args_with_int_dtype(self, xp):
        return xp.logspace(0.1, 2.1, 11, dtype=int)

    def test_logspace_neg_num(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.logspace(0, 10, -1)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_logspace_base(self, xp, dtype):
        return xp.logspace(0, 2, 5, base=2.0, dtype=dtype)


@testing.parameterize(
    *testing.product({
        'indexing': ['xy', 'ij'],
        'sparse': [False, True],
        'copy': [False, True],
    })
)
@testing.gpu
class TestMeshgrid(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_meshgrid0(self, dtype):
        out = cupy.meshgrid(indexing=self.indexing, sparse=self.sparse,
                            copy=self.copy)
        assert(out == [])

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_meshgrid1(self, xp, dtype):
        x = xp.arange(2).astype(dtype)
        return xp.meshgrid(x, indexing=self.indexing, sparse=self.sparse,
                           copy=self.copy)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_meshgrid2(self, xp, dtype):
        x = xp.arange(2).astype(dtype)
        y = xp.arange(3).astype(dtype)
        return xp.meshgrid(x, y, indexing=self.indexing, sparse=self.sparse,
                           copy=self.copy)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_meshgrid3(self, xp, dtype):
        x = xp.arange(2).astype(dtype)
        y = xp.arange(3).astype(dtype)
        z = xp.arange(4).astype(dtype)
        return xp.meshgrid(x, y, z, indexing=self.indexing, sparse=self.sparse,
                           copy=self.copy)


@testing.gpu
class TestMgrid(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_mgrid0(self, xp):
        return xp.mgrid[0:]

    @testing.numpy_cupy_array_equal()
    def test_mgrid1(self, xp):
        return xp.mgrid[-10:10]

    @testing.numpy_cupy_array_equal()
    def test_mgrid2(self, xp):
        return xp.mgrid[-10:10:10j]

    @testing.numpy_cupy_array_equal()
    def test_mgrid3(self, xp):
        x = xp.zeros(10)[:, None]
        y = xp.ones(10)[:, None]
        return xp.mgrid[x:y:10j]

    @testing.numpy_cupy_array_equal()
    def test_mgrid4(self, xp):
        # check len(keys) > 1
        return xp.mgrid[-10:10:10j, -10:10:10j]

    @testing.numpy_cupy_array_equal()
    def test_mgrid5(self, xp):
        # check len(keys) > 1
        x = xp.zeros(10)[:, None]
        y = xp.ones(10)[:, None]
        return xp.mgrid[x:y:10j, x:y:10j]


@testing.gpu
class TestOgrid(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_ogrid0(self, xp):
        return xp.ogrid[0:]

    @testing.numpy_cupy_array_equal()
    def test_ogrid1(self, xp):
        return xp.ogrid[-10:10]

    @testing.numpy_cupy_array_equal()
    def test_ogrid2(self, xp):
        return xp.ogrid[-10:10:10j]

    @testing.numpy_cupy_array_equal()
    def test_ogrid3(self, xp):
        x = xp.zeros(10)[:, None]
        y = xp.ones(10)[:, None]
        return xp.ogrid[x:y:10j]

    @testing.numpy_cupy_array_equal()
    def test_ogrid4(self, xp):
        # check len(keys) > 1
        return xp.ogrid[-10:10:10j, -10:10:10j]

    @testing.numpy_cupy_array_equal()
    def test_ogrid5(self, xp):
        # check len(keys) > 1
        x = xp.zeros(10)[:, None]
        y = xp.ones(10)[:, None]
        return xp.ogrid[x:y:10j, x:y:10j]

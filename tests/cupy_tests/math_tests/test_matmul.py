from __future__ import annotations

import operator
import unittest
import warnings

import numpy
import pytest

import cupy
from cupy._core import _routines_linalg as _linalg
from cupy import testing


@testing.parameterize(
    *testing.product({
        'shape_pair': [
            # dot test
            ((3, 2), (2, 4)),
            ((3, 0), (0, 4)),
            ((0, 2), (2, 4)),
            ((3, 2), (2, 0)),
            ((2,), (2, 4)),
            ((0,), (0, 4)),
            ((3, 2), (2,)),
            ((3, 0), (0,)),
            ((2,), (2,)),
            ((0,), (0,)),
            # matmul test
            ((5, 3, 2), (5, 2, 4)),
            ((0, 3, 2), (0, 2, 4)),
            ((5, 3, 2), (2, 4)),
            ((0, 3, 2), (2, 4)),
            ((3, 2), (5, 2, 4)),
            ((3, 2), (0, 2, 4)),
            ((5, 3, 2), (1, 2, 4)),
            ((0, 3, 2), (1, 2, 4)),
            ((1, 3, 2), (5, 2, 4)),
            ((1, 3, 2), (0, 2, 4)),
            ((5, 3, 2), (2,)),
            ((5, 3, 0), (0,)),
            ((2,), (5, 2, 4)),
            ((0,), (5, 0, 4)),
            ((2, 2, 3, 2), (2, 2, 2, 4)),
            ((5, 0, 3, 2), (5, 0, 2, 4)),
            ((6, 5, 3, 2), (2, 4)),
            ((5, 0, 3, 2), (2, 4)),
            ((3, 2), (6, 5, 2, 4)),
            ((3, 2), (5, 0, 2, 4)),
            ((1, 5, 3, 2), (6, 1, 2, 4)),
            ((1, 0, 3, 2), (6, 1, 2, 4)),
            ((6, 1, 3, 2), (1, 5, 2, 4)),
            ((6, 1, 3, 2), (1, 0, 2, 4)),
            ((6, 5, 3, 2), (2,)),
            ((6, 5, 3, 0), (0,)),
            ((2,), (6, 5, 2, 4)),
            ((0,), (6, 5, 0, 4)),
            ((1, 3, 3), (10, 1, 3, 1)),
        ],
    }))
class TestMatmul(unittest.TestCase):

    @testing.for_all_dtypes(name='dtype1')
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_operator_matmul(self, xp, dtype1, dtype2):
        x1 = testing.shaped_arange(self.shape_pair[0], xp, dtype1)
        x2 = testing.shaped_arange(self.shape_pair[1], xp, dtype2)
        return operator.matmul(x1, x2)

    @testing.for_all_dtypes(name='dtype1')
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_cupy_matmul(self, xp, dtype1, dtype2):
        x1 = testing.shaped_arange(self.shape_pair[0], xp, dtype1)
        x2 = testing.shaped_arange(self.shape_pair[1], xp, dtype2)
        return xp.matmul(x1, x2)


@testing.parameterize(
    *testing.product({
        'shape_pair': [
            # dot test
            ((2, 3), (3, 4), (2, 4)),
            # ((0,), (0,), (0,)),  # TODO: fix GUFunc bug?
            # matmul test
            ((5, 3, 2), (5, 2, 4), (5, 3, 4)),
            ((0, 3, 2), (0, 2, 4), (0, 3, 4)),
        ],
    }))
class TestMatmulOut(unittest.TestCase):

    @testing.for_all_dtypes(name='dtype1')
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_cupy_allclose(
        rtol=1e-3, atol=1e-3,  # required for uint8
        accept_error=TypeError)
    def test_cupy_matmul_noncontiguous(self, xp, dtype1, dtype2):
        x1 = testing.shaped_arange(self.shape_pair[0], xp, dtype1)
        x2 = testing.shaped_arange(self.shape_pair[1], xp, dtype2)
        out = xp.zeros(self.shape_pair[2], dtype1)[::-1]
        ret = xp.matmul(x1, x2, out=out)
        # TODO: Fix GUFunc bug
        # assert ret is out
        assert xp.allclose(ret, out)
        return ret

    @testing.for_all_dtypes(name='dtype1')
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_cupy_matmul_out_cast(self, xp, dtype1, dtype2):
        x1 = testing.shaped_arange(self.shape_pair[0], xp, dtype1)
        x2 = testing.shaped_arange(self.shape_pair[1], xp, dtype2)
        out = xp.zeros(self.shape_pair[2], bool)
        ret = xp.matmul(x1, x2, out=out, casting='unsafe')
        # TODO: Fix GUFunc bug
        # assert ret is out
        assert xp.allclose(ret, out)
        return ret


class TestMatmulOutOverlap:

    @pytest.mark.parametrize('shape', [
        (900, 900),
        (2, 600, 600),
    ])
    @testing.for_dtypes([numpy.int32, numpy.float64])
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5)
    def test_overlap_both(self, xp, dtype, shape):
        a = xp.ones(shape, dtype)
        return xp.matmul(a, a, out=a)


class TestMatmulStrides:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_relaxed_c_contiguous_input(self, xp, dtype):
        x1 = testing.shaped_arange((2, 2, 3), xp, dtype)[:, None, :, :]
        x2 = testing.shaped_arange((2, 1, 3, 1), xp, dtype)
        return x1 @ x2


@testing.parameterize(
    *testing.product({
        'shape_pair': [
            ((6, 5, 3, 2), (6, 5, 2, 4)),
            ((6, 5, 3, 2), (6, 1, 2, 4)),
            ((6, 5, 3, 2), (1, 5, 2, 4)),
            ((6, 5, 3, 2), (1, 1, 2, 4)),
            ((6, 1, 3, 2), (6, 5, 2, 4)),
            ((1, 5, 3, 2), (6, 5, 2, 4)),
            ((1, 1, 3, 2), (6, 5, 2, 4)),
            ((3, 2), (6, 5, 2, 4)),
            ((6, 5, 3, 2), (2, 4)),
            ((2,), (6, 5, 2, 4)),
            ((6, 5, 3, 2), (2,)),
        ],
    }))
class TestMatmulLarge(unittest.TestCase):

    # Avoid overflow
    skip_dtypes = {
        (numpy.int8, numpy.uint8),
        (numpy.int8, numpy.int16),
        (numpy.int8, numpy.float16),
        (numpy.uint8, numpy.uint8),
        (numpy.uint8, numpy.int16),
        (numpy.uint8, numpy.uint16),
        (numpy.int16, numpy.int16),
        (numpy.uint16, numpy.uint16),
    }

    @testing.for_all_dtypes(name='dtype1')
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_operator_matmul(self, xp, dtype1, dtype2):
        if ((dtype1, dtype2) in self.skip_dtypes or
                (dtype2, dtype1) in self.skip_dtypes):
            pytest.skip()
        x1 = testing.shaped_random(self.shape_pair[0], xp, dtype1)
        x2 = testing.shaped_random(self.shape_pair[1], xp, dtype2)
        return operator.matmul(x1, x2)

    @testing.for_all_dtypes(name='dtype1')
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_cupy_matmul(self, xp, dtype1, dtype2):
        if ((dtype1, dtype2) in self.skip_dtypes or
                (dtype2, dtype1) in self.skip_dtypes):
            pytest.skip()
        shape1, shape2 = self.shape_pair
        x1 = testing.shaped_random(shape1, xp, dtype1)
        x2 = testing.shaped_random(shape2, xp, dtype2)
        return xp.matmul(x1, x2)


@pytest.mark.parametrize('shape1,shape2', [
    ((256, 256, 3, 2), (256, 256, 2, 4)),
    ((256, 256, 3, 2), (2, 4)),
    ((3, 2), (256, 256, 2, 4))
])
class TestMatmulIntegralLargeBatch:

    @testing.for_int_dtypes(name='dtype')
    @testing.numpy_cupy_array_equal()
    def test_operator_matmul(self, xp, dtype, shape1, shape2):
        x1 = testing.shaped_random(shape1, xp, dtype)
        x2 = testing.shaped_random(shape2, xp, dtype)
        return operator.matmul(x1, x2)

    @testing.for_int_dtypes(name='dtype')
    @testing.numpy_cupy_array_equal()
    def test_cupy_matmul(self, xp, dtype, shape1, shape2):
        x1 = testing.shaped_random(shape1, xp, dtype)
        x2 = testing.shaped_random(shape2, xp, dtype)
        return xp.matmul(x1, x2)


class TestMatmulOverflow(unittest.TestCase):

    @testing.for_int_dtypes(name='dtype', no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_overflow(self, xp, dtype):
        value = numpy.iinfo(dtype).max
        a = xp.array([value - 10]).astype(dtype)
        b = xp.array([value - 10]).astype(dtype)
        return xp.matmul(a, b)


class _TestMatmulComputeTypes(unittest.TestCase):

    def setUp(self):
        self.old_compute_type = cupy._core.get_compute_type(self.dtype)
        cupy._core.set_compute_type(self.dtype, self.compute_type)

    def tearDown(self):
        cupy._core.set_compute_type(self.dtype, self.old_compute_type)

    def make_x1_x2(self, xp, shapes, dtypes):
        x1 = testing.shaped_random(shapes[0], xp, dtypes[0])
        x2 = testing.shaped_random(shapes[1], xp, dtypes[1])
        return x1, x2


@testing.parameterize(
    *testing.product({
        'compute_type': [
            _linalg.COMPUTE_TYPE_DEFAULT,
            _linalg.COMPUTE_TYPE_PEDANTIC,
        ],
        'shape_pair': [
            ((32, 64), (64, 96)),
            ((64, 96), (96, 32)),
            ((96, 32), (32, 64)),
        ],
    }))
class TestMatmulFp16ComputeTypes(_TestMatmulComputeTypes):
    dtype = numpy.float16

    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)
    def test_operator_matmul(self, xp):
        x1, x2 = self.make_x1_x2(xp, self.shape_pair, (self.dtype, self.dtype))
        return operator.matmul(x1, x2)

    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)
    def test_cupy_matmul(self, xp):
        x1, x2 = self.make_x1_x2(xp, self.shape_pair, (self.dtype, self.dtype))
        return xp.matmul(x1, x2)


@testing.parameterize(
    *testing.product({
        'compute_type': [
            _linalg.COMPUTE_TYPE_DEFAULT,
            _linalg.COMPUTE_TYPE_PEDANTIC,
            _linalg.COMPUTE_TYPE_TF32,
        ],
        'shape_pair': [
            ((100, 200), (200, 300)),
            ((200, 300), (300, 100)),
            ((300, 100), (100, 200)),
        ],
        'dtype_pair': [
            (numpy.float16, numpy.float32),
            (numpy.float32, numpy.float32),
            (numpy.float16, numpy.complex64),
            (numpy.float32, numpy.complex64),
            (numpy.complex64, numpy.complex64),
        ],
    }))
class TestMatmulFp32ComputeTypes(_TestMatmulComputeTypes):
    dtype = numpy.float32

    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)
    def test_operator_matmul(self, xp):
        x1, x2 = self.make_x1_x2(xp, self.shape_pair, self.dtype_pair)
        return operator.matmul(x1, x2)

    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)
    def test_cupy_matmul(self, xp):
        x1, x2 = self.make_x1_x2(xp, self.shape_pair, self.dtype_pair)
        return xp.matmul(x1, x2)


@testing.parameterize(
    *testing.product({
        'compute_type': [
            _linalg.COMPUTE_TYPE_DEFAULT,
            _linalg.COMPUTE_TYPE_PEDANTIC,
        ],
        'shape_pair': [
            ((100, 200), (200, 300)),
            ((200, 300), (300, 100)),
            ((300, 100), (100, 200)),
        ],
        'dtype_pair': [
            (numpy.float32, numpy.float64),
            (numpy.float64, numpy.float64),
            (numpy.float32, numpy.complex128),
            (numpy.float64, numpy.complex128),
            (numpy.complex64, numpy.complex128),
            (numpy.complex128, numpy.complex128),
        ],
    }))
class TestMatmulFp64ComputeTypes(_TestMatmulComputeTypes):
    dtype = numpy.float64

    @testing.numpy_cupy_allclose()
    def test_operator_matmul(self, xp):
        x1, x2 = self.make_x1_x2(xp, self.shape_pair, self.dtype_pair)
        return operator.matmul(x1, x2)

    @testing.numpy_cupy_allclose()
    def test_cupy_matmul(self, xp):
        x1, x2 = self.make_x1_x2(xp, self.shape_pair, self.dtype_pair)
        return xp.matmul(x1, x2)


@testing.parameterize(
    *testing.product({
        'shape_pair': [
            # k is a multiple of 4 (cuBLAS IMMA path)
            ((32, 64), (64, 96)),
            ((64, 32), (32, 96)),
            # k is NOT a multiple of 4 (alignment guard -> CUDA kernel path)
            ((32, 7), (7, 16)),
            ((16, 9), (9, 32)),
        ],
    }))
class TestMatmulInt8(unittest.TestCase):
    """int8 matmul correctness against NumPy across aligned and unaligned k."""

    def setUp(self):
        if cupy.cuda.runtime.is_hip:
            pytest.skip('int8 cublasGemmEx path is NVIDIA-only')
        if int(cupy.cuda.Device().compute_capability) < 61:
            pytest.skip(
                'CUBLAS_COMPUTE_32I requires compute capability >= 6.1')

    @testing.numpy_cupy_array_equal()
    def test_operator_matmul(self, xp):
        rng = numpy.random.default_rng(seed=42)
        m, k = self.shape_pair[0]
        _, n = self.shape_pair[1]
        x1 = xp.asarray(rng.integers(-10, 10, (m, k), dtype=numpy.int8))
        x2 = xp.asarray(rng.integers(-10, 10, (k, n), dtype=numpy.int8))
        return x1 @ x2

    @testing.numpy_cupy_array_equal()
    def test_transposed_inputs(self, xp):
        """Exercises _mat_to_cublas_contiguous with non-standard strides."""
        rng = numpy.random.default_rng(seed=0)
        m, k = self.shape_pair[0]
        _, n = self.shape_pair[1]
        x1 = xp.asarray(numpy.asfortranarray(
            rng.integers(-10, 10, (m, k), dtype=numpy.int8)))
        x2 = xp.asarray(numpy.asfortranarray(
            rng.integers(-10, 10, (k, n), dtype=numpy.int8)))
        return x1 @ x2

    def test_overflow_wraps_like_numpy(self):
        """int32->int8 cast wraps mod 256, matching NumPy matmul semantics."""
        a = cupy.full((4, 4), 100, dtype=numpy.int8)
        b = cupy.full((4, 4), 100, dtype=numpy.int8)
        np_a = cupy.asnumpy(a)
        np_b = cupy.asnumpy(b)
        cupy.testing.assert_array_equal(a @ b, np_a @ np_b)

    def test_fallback_emits_performance_warning(self):
        """When cuBLAS fails, PerformanceWarning is emitted before fallback."""
        from cupy import _util
        from cupy_backends.cuda.libs import cublas
        import unittest.mock as mock

        m, k = self.shape_pair[0]
        _, n = self.shape_pair[1]
        k_aligned = ((k + 3) // 4) * 4
        a = cupy.zeros((m, k_aligned), dtype=numpy.int8)
        b = cupy.zeros((k_aligned, n), dtype=numpy.int8)

        def _failing_gemmEx(*args, **kwargs):
            raise cublas.CUBLASError(13)  # CUBLAS_STATUS_NOT_SUPPORTED

        with mock.patch.object(cublas, 'gemmEx', side_effect=_failing_gemmEx):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                _ = a @ b
        perf_warnings = [
            w for w in caught
            if issubclass(w.category, _util.PerformanceWarning)
        ]
        assert len(perf_warnings) >= 1


class TestMatmulInt8SetComputeTypeError(unittest.TestCase):
    """set_compute_type rejects non-DEFAULT compute types for int dtypes."""

    def test_int8_rejects_fp16_compute_type(self):
        with pytest.raises(ValueError, match='integer dtypes'):
            cupy._core.set_compute_type(numpy.int8, _linalg.COMPUTE_TYPE_FP16)

    def test_int8_rejects_fp32_compute_type(self):
        with pytest.raises(ValueError, match='integer dtypes'):
            cupy._core.set_compute_type(numpy.int8, _linalg.COMPUTE_TYPE_FP32)

    def test_int8_accepts_default_compute_type(self):
        old = cupy._core.get_compute_type(numpy.int8)
        cupy._core.set_compute_type(numpy.int8, _linalg.COMPUTE_TYPE_DEFAULT)
        cupy._core.set_compute_type(numpy.int8, old)


@testing.parameterize(
    *testing.product({
        'shape_pair': [
            ((5, 3, 1), (3, 1, 4)),
            ((3, 2, 3), (3, 2, 4)),
            ((3, 2), ()),
            ((), (3, 2)),
            ((), ()),
            ((3, 2), (1,)),
            ((0, 2), (3, 0)),
            ((0, 1, 1), (2, 1, 1)),
        ],
    }))
class TestMatmulInvalidShape(unittest.TestCase):

    def test_invalid_shape(self):
        for xp in (numpy, cupy):
            shape1, shape2 = self.shape_pair
            x1 = testing.shaped_arange(shape1, xp, numpy.float32)
            x2 = testing.shaped_arange(shape2, xp, numpy.float32)
            with pytest.raises(ValueError):
                xp.matmul(x1, x2)


@testing.parameterize(
    *testing.product({
        'shapes_axes': [
            (((2, 5, 3, 2, 3, 4),  (3, 5, 1, 1, 1, 4), (5, 5, 2, 2, 3, 4)),
             [(1, 2), (0, 1), (0, 1)]),
            (((2, 5, 3, 2, 3, 4),  (2, 5, 3, 1, 4, 1), (3, 1, 2, 5, 3, 2)),
             [(-2, -1), (-2, -1), (0, 1)]),
            (((3, 2, 4, 4), (4, 4, 3, 2), (4, 4, 3, 3)),
             [(0, 1), (-1, -2), (-2, -1)]),
            (((3, 2, 4, 4), (2, 3, 4, 4), (4, 3, 3, 4)),
             [(0, 1), (0, 1), (1, 2)]),
        ],
    }))
class TestMatmulAxes(unittest.TestCase):

    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_cupy_matmul_axes(self, xp):
        x1 = testing.shaped_arange(self.shapes_axes[0][0], xp)
        x2 = testing.shaped_arange(self.shapes_axes[0][1], xp)
        return xp.matmul(x1, x2, axes=self.shapes_axes[1])

    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_cupy_matmul_axes_out(self, xp):
        x1 = testing.shaped_arange(self.shapes_axes[0][0], xp)
        x2 = testing.shaped_arange(self.shapes_axes[0][1], xp)
        out = xp.zeros(self.shapes_axes[0][2])
        xp.matmul(x1, x2, axes=self.shapes_axes[1], out=out)
        return out


class TestMatmulDispatch(unittest.TestCase):

    def test_matmul_dispatch(self):
        x1 = testing.shaped_arange((2, 10, 5), cupy)
        x2 = testing.shaped_arange((10, 2, 5), cupy)
        o_np = numpy.matmul(x1, x2, axes=[(0, 1), (0, 1), (0, 1)])
        assert isinstance(o_np, cupy.ndarray)
        o_cp = cupy.matmul(x1, x2, axes=[(0, 1), (0, 1), (0, 1)])
        testing.assert_allclose(o_np, o_cp)

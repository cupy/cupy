from __future__ import annotations

import operator
import unittest
from unittest import mock

import numpy
import pytest

import cupy
from cupy._core import _routines_linalg as _linalg
from cupy import testing
from cupy.cuda import runtime
from cupy_backends.cuda.libs import cublas


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


@testing.parameterize(
    *testing.product({
        'dtype_name': ['float16', 'bfloat16'],
        'compute_type': [
            _linalg.COMPUTE_TYPE_DEFAULT,
            _linalg.COMPUTE_TYPE_PEDANTIC,
        ],
        'shape_pair_batched': [
            (((2, 3, 64), (2, 64, 4)), False),
            (((2, 5, 3, 64), (2, 5, 64, 4)), False),
            (((2, 3, 64), (64, 4)), True),
            (((2, 5, 3, 64), (64, 4)), True),
            (((2, 1, 3, 64), (1, 5, 64, 4)), True),
        ],
        'noncontiguous': [False, True],
    }))
class TestMatmul16Bit(unittest.TestCase):

    def setUp(self):
        if runtime.is_hip:
            pytest.skip('CUDA-specific GEMM dispatch')

        min_capability = 80 if self.dtype_name == 'bfloat16' else 70
        if int(cupy.cuda.Device().compute_capability) < min_capability:
            pytest.skip('16-bit tensor cores are not available')

        if self.dtype_name == 'bfloat16':
            if (numpy.lib.NumpyVersion(numpy.__version__) < '2.1.2'
                    or cupy.cuda.get_local_runtime_version() < 12020):
                pytest.skip('bfloat16 is not supported')
            ml_dtypes = pytest.importorskip('ml_dtypes')
            self.dtype = numpy.dtype(ml_dtypes.bfloat16)
            self.cuda_dtype = runtime.CUDA_R_16BF
        else:
            self.dtype = numpy.dtype(numpy.float16)
            self.cuda_dtype = runtime.CUDA_R_16F

    @pytest.mark.thread_unsafe(reason="uses mock")
    def test_matmul(self):
        shape_pair, batched = self.shape_pair_batched
        a = testing.shaped_random(
            shape_pair[0], cupy, numpy.float32).astype(self.dtype)
        b = testing.shaped_random(
            shape_pair[1], cupy, numpy.float32).astype(self.dtype)

        if self.noncontiguous:
            a = a[..., ::-1]
            b = b[..., ::-1, :]

        # Compute the reference in float32 from the rounded inputs.
        expected = numpy.matmul(
            cupy.asnumpy(a.astype(numpy.float32)),
            cupy.asnumpy(b.astype(numpy.float32)),
        ).astype(self.dtype)

        out = cupy.empty(expected.shape, dtype=self.dtype)
        if self.noncontiguous:
            out = out[..., ::-1]

        name = 'gemmBatchedEx' if batched else 'gemmStridedBatchedEx'
        with mock.patch.object(
                cublas, name, wraps=getattr(cublas, name)) as gemm:
            result = cupy.matmul(a, b, out=out)

        assert result is out
        assert result.dtype == self.dtype
        assert gemm.call_count == 1

        args = gemm.call_args.args
        # Positions of Atype, Btype, and Ctype in each cuBLAS wrapper.
        dtype_indices = (8, 11, 15) if batched else (8, 12, 17)
        for index in dtype_indices:
            assert args[index] == self.cuda_dtype
        assert args[-2] == cublas.CUBLAS_COMPUTE_32F

        testing.assert_allclose(
            result.astype(numpy.float32),
            expected.astype(numpy.float32),
            rtol=1e-2, # bfloat16 error range
            atol=1e-3,
        )

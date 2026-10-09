from __future__ import annotations

import operator
import unittest
import warnings

import numpy
import pytest

import cupy
from cupy._core import _routines_linalg as _linalg
from cupy import testing
from cupy.cuda import runtime


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

    @pytest.mark.parametrize('side', ['left', 'right'])
    @testing.for_dtypes([numpy.int32, numpy.float64])
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5)
    def test_overlap_broadcast(self, xp, dtype, side):
        a = testing.shaped_arange((2, 3, 3), xp, dtype)
        if side == 'left':
            return xp.matmul(a[0], a, out=a)
        return xp.matmul(a, a[0], out=a)


class TestMatmulStrides:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_relaxed_c_contiguous_input(self, xp, dtype):
        x1 = testing.shaped_arange((2, 2, 3), xp, dtype)[:, None, :, :]
        x2 = testing.shaped_arange((2, 1, 3, 1), xp, dtype)
        return x1 @ x2

    @pytest.mark.parametrize('side', ['left', 'right'])
    @testing.for_dtypes([
        numpy.int32, numpy.float16, numpy.float32, numpy.complex64])
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)
    def test_vector_strided_batch_output(self, xp, dtype, side):
        vector = testing.shaped_arange((4,), xp, dtype)
        matrix_shape = (2, 5, 4, 3) if side == 'left' else (2, 5, 3, 4)
        matrix = testing.shaped_arange(matrix_shape, xp, dtype)
        out = xp.empty((2, 5, 5), dtype)[..., :3][::-1, ::-1]
        if side == 'left':
            return xp.matmul(vector, matrix, out=out)
        return xp.matmul(matrix, vector, out=out)

    @pytest.mark.parametrize('broadcast', [False, True])
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)
    def test_noncontiguous_matrices(self, xp, dtype, broadcast):
        a = testing.shaped_random((2, 5, 3, 8), xp, dtype)[..., ::-1]
        b_shape = (1, 5, 8, 3) if broadcast else (2, 5, 8, 3)
        b = testing.shaped_random(b_shape, xp, dtype)[..., ::-1, :]
        return xp.matmul(a, b)

    @pytest.mark.parametrize('broadcast', [False, True])
    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_batch_strided_integral(self, xp, dtype, broadcast):
        a = testing.shaped_random((2, 5, 3, 8), xp, dtype)[::-1, ::-1]
        b_shape = (1, 5, 8, 3) if broadcast else (2, 5, 8, 3)
        b = testing.shaped_random(b_shape, xp, dtype)
        out = xp.empty((2, 5, 3, 4), dtype)[..., :3][::-1, ::-1]
        return xp.matmul(a, b, out=out)

    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_padded_integral_batches(self, xp, dtype):
        a_storage = testing.shaped_random((2, 5, 32), xp, dtype)
        b_storage = testing.shaped_random((2, 5, 32), xp, dtype)
        a = a_storage[..., :24].reshape((2, 5, 3, 8))
        b = b_storage[..., :24].reshape((2, 5, 8, 3))
        out_storage = xp.empty((2, 5, 16), dtype)
        out = out_storage[..., :9].reshape((2, 5, 3, 3))
        return xp.matmul(a, b, out=out)


@pytest.mark.parametrize('dtype', [
    numpy.bool_, numpy.int32, numpy.float16, numpy.float32, numpy.complex64,
])
@pytest.mark.parametrize('transpose_a, transpose_b, transpose_out', [
    (a, b, c) for a in (False, True) for b in (False, True)
    for c in (False, True)
])
@pytest.mark.parametrize('broadcast', [False, True])
@pytest.mark.parametrize('layout', ['plain', 'reversed', 'permuted'])
class TestMatmulMatrixOrder:

    def test_matrix_order(
            self, dtype, transpose_a, transpose_b, transpose_out,
            broadcast, layout):
        def make_array(shape, transpose, output=False):
            if transpose:
                shape = shape[:-2] + (shape[-1], shape[-2])
            if layout == 'permuted':
                shape = (shape[1], shape[0]) + shape[2:]
            arr = (cupy.empty(shape, dtype) if output
                   else testing.shaped_random(shape, cupy, dtype))
            if layout == 'permuted':
                arr = arr.swapaxes(0, 1)
            elif layout == 'reversed':
                arr = arr[::-1, ::-1]
            return arr.swapaxes(-1, -2) if transpose else arr

        a_shape = (2, 1, 3, 4) if broadcast else (2, 5, 3, 4)
        b_shape = (1, 5, 4, 2) if broadcast else (2, 5, 4, 2)
        a = make_array(a_shape, transpose_a)
        b = make_array(b_shape, transpose_b)
        out = make_array((2, 5, 3, 2), transpose_out, output=True)
        expected = numpy.matmul(cupy.asnumpy(a), cupy.asnumpy(b))
        result = cupy.matmul(a, b, out=out)
        assert result is out
        testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('dtype', [
    numpy.bool_, numpy.float16, numpy.float32, numpy.complex128,
])
@pytest.mark.parametrize('layout', [
    'contiguous', 'reversed', 'permuted', 'empty_batch',
])
class TestMatPtrs:

    def test_matrix_starts(self, dtype, layout):
        shape = (2, 0, 3, 4) if layout == 'empty_batch' else (2, 5, 3, 4)
        arr = cupy.empty(shape, dtype)
        if layout == 'reversed':
            arr = arr[::-1, ::-1]
        elif layout == 'permuted':
            arr = arr.swapaxes(0, 1)
        expected = numpy.array([
            arr.data.ptr + sum(i * s for i, s in zip(index, arr.strides))
            for index in numpy.ndindex(arr.shape[:-2])
        ], dtype=numpy.uintp).reshape(arr.shape[:-2])
        ptrs = cupy._core._mat_ptrs(arr)
        assert ptrs.shape == expected.shape
        assert ptrs.dtype == numpy.dtype(numpy.uintp)
        testing.assert_array_equal(ptrs, expected)


class TestMatmulBroadcastBatchSteps:

    @pytest.mark.parametrize('side', ['left', 'right', 'both'])
    @pytest.mark.parametrize('batch_shape', [(2, 5), (2, 1, 5)])
    @pytest.mark.parametrize('padded', [False, True])
    @testing.for_dtypes([numpy.int32, numpy.float32, numpy.complex64])
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5)
    def test_zero_steps(self, xp, dtype, side, batch_shape, padded):
        def make_array(shape):
            if not padded:
                return testing.shaped_random(shape, xp, dtype)
            core_size = shape[-2] * shape[-1]
            storage = testing.shaped_random(
                shape[:-2] + (core_size + 8,), xp, dtype)
            return storage[..., :core_size].reshape(shape)

        a_shape = (3, 4) if side in ('left', 'both') else batch_shape + (3, 4)
        b_shape = (4, 2) if side in ('right', 'both') else batch_shape + (4, 2)
        a = xp.broadcast_to(make_array(a_shape), batch_shape + (3, 4))
        b = xp.broadcast_to(make_array(b_shape), batch_shape + (4, 2))
        out = None
        if padded:
            storage = xp.empty(batch_shape + (14,), dtype)
            out = storage[..., :6].reshape(batch_shape + (3, 2))
        result = xp.matmul(a, b, out=out)
        if out is not None:
            assert result is out
        return result


@pytest.mark.parametrize('dtype, k', [
    (numpy.float16, 2),  # 4-byte pointer alignment
    (numpy.float16, 3),  # 2-byte pointer alignment
    (numpy.float16, 8),  # 16-byte pointer alignment
    (numpy.float32, 3), (numpy.complex64, 3),
])
@pytest.mark.parametrize('m, n', [(3, 3), (1, 3), (3, 1)])
@pytest.mark.parametrize('layout', [
    'contiguous', 'padded', 'reversed', 'permuted', 'unaligned_batch',
    'column_strided', 'row_padded',
])
class TestMatmulBatchedOutput:

    def test_output_layout(self, dtype, k, m, n, layout):
        a = testing.shaped_random((2, 1, m, k), cupy, dtype)
        b = testing.shaped_random((1, 5, k, n), cupy, dtype)
        out_shape = (2, 5, m, n)
        if layout == 'contiguous':
            out = cupy.empty(out_shape, dtype)
        elif layout in ('padded', 'reversed', 'permuted', 'unaligned_batch'):
            batch_shape = (5, 2) if layout == 'permuted' else (2, 5)
            stride = m * n + 1 if layout == 'unaligned_batch' else 16
            storage = cupy.empty(batch_shape + (stride,), dtype)
            out = storage[..., :m * n].reshape(batch_shape + (m, n))
            if layout == 'reversed':
                out = out[::-1, ::-1]
            elif layout == 'permuted':
                out = out.swapaxes(0, 1)
        elif layout == 'column_strided':
            out = cupy.empty((2, 5, m, 2 * n), dtype)[..., ::2]
        else:
            out = cupy.empty((2, 5, m, n + 1), dtype)[..., :n]

        result = cupy.matmul(a, b, out=out)
        assert result is out
        assert result.dtype == dtype
        expected = numpy.matmul(
            cupy.asnumpy(a), cupy.asnumpy(b))
        testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('dtype', [
    numpy.int32, numpy.float16, numpy.float32, numpy.complex64,
])
@pytest.mark.parametrize('layout', [
    'padded', 'reversed', 'permuted', 'unaligned_batch',
])
@pytest.mark.parametrize('m, k, n', [(1, 1, 1), (1, 1, 2), (3, 8, 3)])
@pytest.mark.parametrize('reverse_b', [False, True])
@pytest.mark.parametrize('reverse_out', [False, True])
class TestMatmulBatchedInput:

    def test_batch_strides(
            self, dtype, layout, m, k, n, reverse_b, reverse_out):
        if layout in ('padded', 'unaligned_batch'):
            stride = m * k + (1 if layout == 'unaligned_batch' else 8)
            storage = testing.shaped_random((2, 1, 5, stride), cupy, dtype)
            a = storage[..., :m * k].reshape((2, 1, 5, m, k), copy=False)
        elif layout == 'reversed':
            a = testing.shaped_random(
                (2, 1, 5, m, k), cupy, dtype)[::-1, :, ::-1]
        else:
            a = testing.shaped_random(
                (5, 1, 2, m, k), cupy, dtype).swapaxes(0, 2)
        b = testing.shaped_random((2, 1, 5, k, n), cupy, dtype)
        out = None
        if reverse_b:
            b = b[::-1, :, ::-1]
        if reverse_out:
            out = cupy.empty((2, 1, 5, m, n), dtype)[::-1, :, ::-1]
        expected = numpy.matmul(cupy.asnumpy(a), cupy.asnumpy(b))
        result = cupy.matmul(a, b, out=out)
        if out is not None:
            assert result is out
        assert result.dtype == dtype
        if numpy.dtype(dtype).kind in 'biu':
            testing.assert_array_equal(result, expected)
        else:
            testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-3)


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
    ((3, 2), (256, 256, 2, 4)),
    ((256, 1, 3, 2), (1, 256, 2, 4)),
])
@pytest.mark.parametrize('reverse_batch', [False, True])
class TestMatmulIntegralLargeBatch:

    @testing.for_int_dtypes(name='dtype')
    @testing.numpy_cupy_array_equal()
    def test_operator_matmul(self, xp, dtype, shape1, shape2, reverse_batch):
        x1 = testing.shaped_random(shape1, xp, dtype)
        x2 = testing.shaped_random(shape2, xp, dtype)
        if reverse_batch:
            if x1.ndim > 2:
                x1 = x1[::-1, ::-1]
            if x2.ndim > 2:
                x2 = x2[::-1, ::-1]
        return operator.matmul(x1, x2)

    @testing.for_int_dtypes(name='dtype')
    @testing.numpy_cupy_array_equal()
    def test_cupy_matmul(self, xp, dtype, shape1, shape2, reverse_batch):
        x1 = testing.shaped_random(shape1, xp, dtype)
        x2 = testing.shaped_random(shape2, xp, dtype)
        if reverse_batch:
            if x1.ndim > 2:
                x1 = x1[::-1, ::-1]
            if x2.ndim > 2:
                x2 = x2[::-1, ::-1]
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

    @pytest.mark.thread_unsafe(
        reason="patches cublas.gemmEx via mock.patch.object")
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


class _Matmul16BitTestBase:

    @pytest.fixture
    def dtype(self, dtype_name):
        if dtype_name == 'bfloat16':
            if (runtime.is_hip
                    or cupy.cuda.get_local_runtime_version() < 12020
                    or numpy.lib.NumpyVersion(numpy.__version__) < '2.1.2'):
                pytest.skip('bfloat16 is not supported')
            ml_dtypes = pytest.importorskip('ml_dtypes')
            return numpy.dtype(ml_dtypes.bfloat16)
        return numpy.dtype(numpy.float16)

    @staticmethod
    def _misaligned_empty(shape, dtype):
        out = cupy.empty(int(numpy.prod(shape)) + 1, dtype=dtype)
        assert out.data.ptr % 16 == 0
        out = out[1:].reshape(shape)
        assert out.flags.c_contiguous
        assert out.data.ptr % 16 == 2
        return out


@pytest.mark.parametrize('dtype_name', ['float16', 'bfloat16'])
@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('k, batch_shape', [
    (4, (2, 5)), (0, (2, 5)), (4, (0,)),
])
@pytest.mark.parametrize('with_out', [False, True])
class TestMatmul16BitVector(_Matmul16BitTestBase):

    def test_vector(self, dtype, side, k, batch_shape, with_out):
        vector = testing.shaped_random((k,), cupy, numpy.float32).astype(dtype)
        matrix_shape = (k, 3) if side == 'left' else (3, k)
        matrix = testing.shaped_random(
            batch_shape + matrix_shape, cupy, numpy.float32).astype(dtype)
        a, b = (vector, matrix) if side == 'left' else (matrix, vector)
        expected = numpy.matmul(
            cupy.asnumpy(a.astype(numpy.float32)),
            cupy.asnumpy(b.astype(numpy.float32))).astype(dtype)
        out = cupy.empty(expected.shape, dtype) if with_out else None
        result = cupy.matmul(a, b, out=out)
        if with_out:
            assert result is out
        assert result.dtype == dtype
        testing.assert_allclose(
            result.astype(numpy.float32), expected.astype(numpy.float32),
            rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize('dtype_name', ['float16', 'bfloat16'])
@pytest.mark.parametrize('broadcast', [False, True])
@pytest.mark.parametrize('transpose_out', [False, True])
class TestMatmul16BitMatrixOrder(_Matmul16BitTestBase):

    def test_matrix_order(self, dtype, broadcast, transpose_out):
        a_shape = (2, 1, 8, 3) if broadcast else (2, 5, 8, 3)
        b_shape = (1, 5, 5, 8) if broadcast else (2, 5, 5, 8)
        a = testing.shaped_random(
            a_shape, cupy, numpy.float32).astype(dtype).swapaxes(-1, -2)
        b = testing.shaped_random(
            b_shape, cupy, numpy.float32).astype(dtype).swapaxes(-1, -2)
        storage = cupy.empty((2, 5, 16), dtype)
        matrix_shape = (5, 3) if transpose_out else (3, 5)
        out = storage[..., :15].reshape((2, 5) + matrix_shape, copy=False)
        if transpose_out:
            out = out.swapaxes(-1, -2)
        expected = numpy.matmul(
            cupy.asnumpy(a.astype(numpy.float32)),
            cupy.asnumpy(b.astype(numpy.float32))).astype(dtype)
        result = cupy.matmul(a, b, out=out)
        assert result is out
        testing.assert_allclose(
            result.astype(numpy.float32), expected.astype(numpy.float32),
            rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize('dtype_name', ['float16', 'bfloat16'])
@pytest.mark.parametrize('shape_pair', [
    ((2, 3, 64), (2, 64, 4)),
    ((2, 5, 3, 64), (64, 4)),
    ((2, 1, 3, 8), (1, 5, 8, 3)),
    ((2, 1, 3, 2), (1, 5, 2, 3)),
    ((2, 1, 3, 3), (1, 5, 3, 3)),
])
@pytest.mark.parametrize(
    'layout', ['contiguous', 'noncontiguous', 'misaligned'])
class TestMatmul16Bit(_Matmul16BitTestBase):

    @pytest.mark.parametrize('with_out', [True, False])
    def test_matmul(self, dtype, shape_pair, layout, with_out):
        a = testing.shaped_random(
            shape_pair[0], cupy, numpy.float32).astype(dtype)
        b = testing.shaped_random(
            shape_pair[1], cupy, numpy.float32).astype(dtype)

        if layout == 'misaligned':
            # Contiguous views retain this two-byte offset through
            # ascontiguousarray, including when the contraction size is 64.
            a_offset = self._misaligned_empty(a.shape, dtype)
            b_offset = self._misaligned_empty(b.shape, dtype)
            a_offset[...] = a
            b_offset[...] = b
            a, b = a_offset, b_offset
        elif layout == 'noncontiguous':
            a = a[..., ::-1]
            b = b[..., ::-1, :]

        # Compute the reference in float32 from the rounded inputs.
        expected = numpy.matmul(
            cupy.asnumpy(a.astype(numpy.float32)),
            cupy.asnumpy(b.astype(numpy.float32)),
        ).astype(dtype)

        if not with_out:
            out = None
        elif layout == 'misaligned':
            out = self._misaligned_empty(expected.shape, dtype)
        else:
            out = cupy.empty(expected.shape, dtype=dtype)
        if with_out and layout == 'noncontiguous':
            out = out[..., ::-1]

        result = cupy.matmul(a, b, out=out)
        if with_out:
            assert result is out
        assert result.dtype == dtype

        testing.assert_allclose(
            result.astype(numpy.float32),
            expected.astype(numpy.float32),
            rtol=1e-2,  # bfloat16 error range
        )


@pytest.mark.parametrize('dtype_name', ['float16', 'bfloat16'])
@pytest.mark.parametrize('batch_shape', [(), (2, 3)])
@pytest.mark.parametrize('k', [2, 3, 8])
@pytest.mark.parametrize('misaligned', ['a', 'b', 'out', 'all'])
class TestMatmul16BitBaseAlignment(_Matmul16BitTestBase):

    def test_base_alignment(self, dtype, batch_shape, k, misaligned):
        a = testing.shaped_random(
            batch_shape + (3, k), cupy, numpy.float32).astype(dtype)
        b = testing.shaped_random(
            batch_shape + (k, 5), cupy, numpy.float32).astype(dtype)
        out = cupy.empty(batch_shape + (3, 5), dtype)
        operands = {'a': a, 'b': b, 'out': out}
        for key in operands:
            if misaligned in (key, 'all'):
                source = operands[key]
                operands[key] = self._misaligned_empty(source.shape, dtype)
                if key != 'out':
                    operands[key][...] = source
        a, b, out = (operands[key] for key in ('a', 'b', 'out'))
        expected = numpy.matmul(
            cupy.asnumpy(a.astype(numpy.float32)),
            cupy.asnumpy(b.astype(numpy.float32))).astype(dtype)

        result = cupy.matmul(a, b, out=out)
        assert result is out
        assert result.dtype == dtype
        testing.assert_allclose(
            result.astype(numpy.float32), expected.astype(numpy.float32),
            rtol=1e-2, atol=1e-3)

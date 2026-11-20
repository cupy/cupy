from __future__ import annotations

import unittest
import sys
import warnings

import numpy
import pytest

import cupy
from cupy import testing
from cupy.testing._helper import skip_if_after_baseline


@testing.parameterize(*testing.product({
    'shape': [
        ((2, 3, 4), (3, 4, 2)),
        ((1, 1), (1, 1)),
        ((1, 1), (1, 2)),
        ((1, 2), (2, 1)),
        ((2, 1), (1, 1)),
        ((1, 2), (2, 3)),
        ((2, 1), (1, 3)),
        ((2, 3), (3, 1)),
        ((2, 3), (3, 4)),
        ((0, 3), (3, 4)),
        ((2, 3), (3, 0)),
        ((0, 3), (3, 0)),
        ((3, 0), (0, 4)),
        ((2, 3, 0), (3, 0, 2)),
        ((0, 0), (0, 0)),
        ((3,), (3,)),
        ((2,), (2, 4)),
        ((4, 2), (2,)),
    ],
    'trans_a': [True, False],
    'trans_b': [True, False],
}))
class TestDot(unittest.TestCase):

    @testing.for_all_dtypes_combination(['dtype_a', 'dtype_b'])
    @testing.numpy_cupy_allclose()
    def test_dot(self, xp, dtype_a, dtype_b):
        shape_a, shape_b = self.shape
        if self.trans_a:
            a = testing.shaped_arange(shape_a[::-1], xp, dtype_a).T
        else:
            a = testing.shaped_arange(shape_a, xp, dtype_a)
        if self.trans_b:
            b = testing.shaped_arange(shape_b[::-1], xp, dtype_b).T
        else:
            b = testing.shaped_arange(shape_b, xp, dtype_b)
        return xp.dot(a, b)

    @testing.for_float_dtypes(name='dtype_a')
    @testing.for_float_dtypes(name='dtype_b')
    @testing.for_float_dtypes(name='dtype_c')
    @testing.numpy_cupy_allclose(accept_error=ValueError)
    def test_dot_with_out(self, xp, dtype_a, dtype_b, dtype_c):
        shape_a, shape_b = self.shape
        if self.trans_a:
            a = testing.shaped_arange(shape_a[::-1], xp, dtype_a).T
        else:
            a = testing.shaped_arange(shape_a, xp, dtype_a)
        if self.trans_b:
            b = testing.shaped_arange(shape_b[::-1], xp, dtype_b).T
        else:
            b = testing.shaped_arange(shape_b, xp, dtype_b)
        if a.ndim == 0 or b.ndim == 0:
            shape_c = shape_a + shape_b
        else:
            shape_c = shape_a[:-1] + shape_b[:-2] + shape_b[-1:]
        c = xp.empty(shape_c, dtype=dtype_c)
        out = xp.dot(a, b, out=c)
        assert out is c
        return c


@testing.parameterize(*testing.product({
    'params': [
        #  Test for 0 dimension
        ((3, ), (3, ), -1, -1, -1),
        #  Test for basic cases
        ((1, 3), (1, 3), 1, -1, -1),
        #  Test for higher dimensions
        ((2, 4, 5, 3), (2, 4, 5, 3), -1, -1, 0),
    ],
}))
class TestCrossProduct(unittest.TestCase):

    @testing.for_all_dtypes_combination(['dtype_a', 'dtype_b'])
    @testing.numpy_cupy_allclose()
    def test_cross(self, xp, dtype_a, dtype_b):
        if dtype_a == dtype_b == numpy.bool_:
            # cross does not support bool-bool inputs.
            return xp.array(True)
        shape_a, shape_b, axisa, axisb, axisc = self.params
        a = testing.shaped_arange(shape_a, xp, dtype_a)
        b = testing.shaped_arange(shape_b, xp, dtype_b)
        return xp.cross(a, b, axisa, axisb, axisc)


# XXX: cross with 2D vectors is deprecated in NumPy 2.0, also CuPy 1.14
@testing.parameterize(*testing.product({
    'params': [
        #  Test for basic cases
        ((1, 2), (1, 2), -1, -1, 1),
        ((1, 2), (1, 3), -1, -1, 1),
        ((2, 2), (1, 3), -1, -1, 0),
        ((3, 3), (1, 2), 0, -1, -1),
        ((0, 3), (0, 3), -1, -1, -1),
        #  Test for higher dimensions
        ((2, 0, 3), (2, 0, 3), 0, 0, 0),
        ((2, 4, 5, 2), (2, 4, 5, 2), 0, 0, -1),
    ],
}))
class TestCrossProductDeprecated(unittest.TestCase):
    @testing.for_all_dtypes_combination(['dtype_a', 'dtype_b'])
    @testing.numpy_cupy_allclose()
    @skip_if_after_baseline(numpy="2.5", reason="deprecation finalized.")
    def test_cross(self, xp, dtype_a, dtype_b):
        if dtype_a == dtype_b == numpy.bool_:
            # cross does not support bool-bool inputs.
            return xp.array(True)
        shape_a, shape_b, axisa, axisb, axisc = self.params
        a = testing.shaped_arange(shape_a, xp, dtype_a)
        b = testing.shaped_arange(shape_b, xp, dtype_b)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            res = xp.cross(a, b, axisa, axisb, axisc)
        return res


@testing.parameterize(*testing.product({
    'params': [
        #  Test for 0 dimension
        ((3, ), (3, ), -1,),
        #  Test for basic cases
        ((1, 3), (1, 3), 1,),
        #  Test for higher dimensions
        ((2, 4, 5, 3), (2, 4, 5, 3), -1),
    ],
}))
class TestLinalgCrossProduct(unittest.TestCase):

    @testing.with_requires('numpy>=2.0')
    @testing.for_all_dtypes_combination(['dtype_a', 'dtype_b'])
    @testing.numpy_cupy_allclose()
    def test_cross(self, xp, dtype_a, dtype_b):
        if dtype_a == dtype_b == numpy.bool_:
            # cross does not support bool-bool inputs.
            return xp.array(True)
        shape_a, shape_b, axis = self.params
        a = testing.shaped_arange(shape_a, xp, dtype_a)
        b = testing.shaped_arange(shape_b, xp, dtype_b)
        return xp.linalg.cross(a, b, axis=axis)


@testing.parameterize(*testing.product({
    'shape': [
        ((), ()),
        ((), (2, 4)),
        ((4, 2), ()),
    ],
    'trans_a': [True, False],
    'trans_b': [True, False],
}))
class TestDotFor0Dim(unittest.TestCase):

    @testing.for_all_dtypes_combination(['dtype_a', 'dtype_b'])
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_dot(self, xp, dtype_a, dtype_b):
        shape_a, shape_b = self.shape
        if self.trans_a:
            a = testing.shaped_arange(shape_a[::-1], xp, dtype_a).T
        else:
            a = testing.shaped_arange(shape_a, xp, dtype_a)
        if self.trans_b:
            b = testing.shaped_arange(shape_b[::-1], xp, dtype_b).T
        else:
            b = testing.shaped_arange(shape_b, xp, dtype_b)
        return xp.dot(a, b)


class TestProduct:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_dot_vec1(self, xp, dtype):
        a = testing.shaped_arange((2,), xp, dtype)
        b = testing.shaped_arange((2,), xp, dtype)
        return xp.dot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_dot_vec2(self, xp, dtype):
        a = testing.shaped_arange((2,), xp, dtype)
        b = testing.shaped_arange((2, 1), xp, dtype)
        return xp.dot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_dot_vec3(self, xp, dtype):
        a = testing.shaped_arange((1, 2), xp, dtype)
        b = testing.shaped_arange((2,), xp, dtype)
        return xp.dot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_transposed_dot(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype).transpose(1, 0, 2)
        b = testing.shaped_arange((2, 3, 4), xp, dtype).transpose(0, 2, 1)
        return xp.dot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_transposed_dot_with_out(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype).transpose(1, 0, 2)
        b = testing.shaped_arange((4, 2, 3), xp, dtype).transpose(2, 0, 1)
        c = xp.ndarray((3, 2, 3, 2), dtype=dtype)
        xp.dot(a, b, out=c)
        return c

    @testing.for_all_dtypes()
    def test_transposed_dot_with_out_f_contiguous(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 3, 4), xp, dtype).transpose(1, 0, 2)
            b = testing.shaped_arange((4, 2, 3), xp, dtype).transpose(2, 0, 1)
            c = xp.ndarray((3, 2, 3, 2), dtype=dtype, order='F')
            with pytest.raises(ValueError):
                # Only C-contiguous array is acceptable
                xp.dot(a, b, out=c)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_dot_with_single_elem_array1(self, xp, dtype):
        a = testing.shaped_arange((3, 1), xp, dtype)
        b = xp.array([[2]], dtype=dtype)
        return xp.dot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_dot_with_single_elem_array2(self, xp, dtype):
        a = xp.array([[2]], dtype=dtype)
        b = testing.shaped_arange((1, 3), xp, dtype)
        return xp.dot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_vdot(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        b = testing.shaped_reverse_arange((5,), xp, dtype)
        return xp.vdot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_reversed_vdot(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)[::-1]
        b = testing.shaped_reverse_arange((5,), xp, dtype)[::-1]
        return xp.vdot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_multidim_vdot(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_arange((2, 2, 2, 3), xp, dtype)
        return xp.vdot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_transposed_multidim_vdot(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype).transpose(2, 0, 1)
        b = testing.shaped_arange(
            (2, 2, 2, 3), xp, dtype).transpose(1, 3, 0, 2)
        return xp.vdot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_inner(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        b = testing.shaped_reverse_arange((5,), xp, dtype)
        return xp.inner(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_reversed_inner(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)[::-1]
        b = testing.shaped_reverse_arange((5,), xp, dtype)[::-1]
        return xp.inner(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_multidim_inner(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_arange((3, 2, 4), xp, dtype)
        return xp.inner(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_transposed_higher_order_inner(self, xp, dtype):
        a = testing.shaped_arange((2, 4, 3), xp, dtype).transpose(2, 0, 1)
        b = testing.shaped_arange((4, 2, 3), xp, dtype).transpose(1, 2, 0)
        return xp.inner(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_outer(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        b = testing.shaped_arange((4,), xp, dtype)
        return xp.outer(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_reversed_outer(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        b = testing.shaped_arange((4,), xp, dtype)
        return xp.outer(a[::-1], b[::-1])

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_multidim_outer(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_arange((4, 5), xp, dtype)
        return xp.outer(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_tensordot(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_arange((3, 4, 5), xp, dtype)
        return xp.tensordot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_transposed_tensordot(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype).transpose(1, 0, 2)
        b = testing.shaped_arange((4, 3, 2), xp, dtype).transpose(2, 0, 1)
        return xp.tensordot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_tensordot_with_int_axes(self, xp, dtype):
        if dtype in (numpy.uint8, numpy.int8, numpy.uint16, numpy.int16):
            a = testing.shaped_arange((1, 2, 3), xp, dtype)
            b = testing.shaped_arange((2, 3, 1), xp, dtype)
            return xp.tensordot(a, b, axes=2)
        else:
            a = testing.shaped_arange((2, 3, 4, 5), xp, dtype)
            b = testing.shaped_arange((3, 4, 5, 2), xp, dtype)
            return xp.tensordot(a, b, axes=3)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_transposed_tensordot_with_int_axes(self, xp, dtype):
        if dtype in (numpy.uint8, numpy.int8, numpy.uint16, numpy.int16):
            # Avoid overflow
            a = testing.shaped_arange(
                (1, 2, 3), xp, dtype).transpose(2, 0, 1)
            b = testing.shaped_arange(
                (3, 2, 1), xp, dtype).transpose(2, 1, 0)
            return xp.tensordot(a, b, axes=2)
        else:
            a = testing.shaped_arange(
                (2, 3, 4, 5), xp, dtype).transpose(2, 0, 3, 1)
            b = testing.shaped_arange(
                (5, 4, 3, 2), xp, dtype).transpose(3, 0, 2, 1)
            return xp.tensordot(a, b, axes=3)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_tensordot_with_list_axes(self, xp, dtype):
        if dtype in (numpy.uint8, numpy.int8, numpy.uint16, numpy.int16):
            # Avoid overflow
            a = testing.shaped_arange((1, 2, 3), xp, dtype)
            b = testing.shaped_arange((3, 1, 2), xp, dtype)
            return xp.tensordot(a, b, axes=([2, 1], [0, 2]))
        else:
            a = testing.shaped_arange((2, 3, 4, 5), xp, dtype)
            b = testing.shaped_arange((3, 5, 4, 2), xp, dtype)
            return xp.tensordot(a, b, axes=([3, 2, 1], [1, 2, 0]))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_transposed_tensordot_with_list_axes(self, xp, dtype):
        if dtype in (numpy.uint8, numpy.int8, numpy.uint16, numpy.int16):
            # Avoid overflow
            a = testing.shaped_arange(
                (1, 2, 3), xp, dtype).transpose(2, 0, 1)
            b = testing.shaped_arange(
                (2, 3, 1), xp, dtype).transpose(0, 2, 1)
            return xp.tensordot(a, b, axes=([2, 0], [0, 2]))
        else:
            a = testing.shaped_arange(
                (2, 3, 4, 5), xp, dtype).transpose(2, 0, 3, 1)
            b = testing.shaped_arange(
                (3, 5, 4, 2), xp, dtype).transpose(3, 0, 2, 1)
            return xp.tensordot(a, b, axes=([2, 0, 3], [3, 2, 1]))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_tensordot_zero_dim(self, xp, dtype):
        a = xp.array(2, dtype=dtype)
        b = testing.shaped_arange((3, 4, 2), xp, dtype)
        return xp.tensordot(a, b, axes=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_kron(self, xp, dtype):
        a = testing.shaped_arange((4,), xp, dtype)
        b = testing.shaped_arange((5,), xp, dtype)
        return xp.kron(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_reversed_kron(self, xp, dtype):
        a = testing.shaped_arange((4,), xp, dtype)
        b = testing.shaped_arange((5,), xp, dtype)
        return xp.kron(a[::-1], b[::-1])

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_multidim_kron(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_arange((4, 2, 3), xp, dtype)
        return xp.kron(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_zerodim_kron(self, xp, dtype):
        a = xp.array(2, dtype=dtype)
        b = testing.shaped_arange((4, 5), xp, dtype)
        return xp.kron(a, b)

    @pytest.mark.parametrize(
        "a, b", [
            (2, 3.0),
            (2, [[0, -1j / 2], [1j / 2, 0]]),
            ([[0, -1j / 2], [1j / 2, 0]], 2)
        ]
    )
    @testing.numpy_cupy_allclose()
    def test_kron_accepts_numbers_as_arguments(self, a, b, xp):
        args = [xp.array(arg) if isinstance(arg, list)
                else arg for arg in [a, b]]
        return xp.kron(*args)

    @pytest.mark.parametrize(
        "shape_a, shape_b", [
            # 2-D, empty in `a`
            ((1, 0), (2, 2)),
            ((3, 0), (2, 4)),
            ((0, 3), (2, 4)),
            ((0, 0), (2, 4)),
            # 2-D, empty in `b`
            ((2, 4), (3, 0)),
            ((2, 4), (0, 0)),
            # 1-D
            ((0,), (4,)),
            ((4,), (0,)),
            ((0,), (0,)),
            # >2-D with an empty dim
            ((2, 0, 3), (1, 4, 2)),
            ((1, 2, 0), (3, 4, 5)),
            # mixed ndim, empty operand smaller-rank
            ((0,), (3, 4)),
            ((4,), (3, 0)),
            ((3, 0), (2,)),
        ]
    )
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_kron_empty(self, xp, dtype, shape_a, shape_b):
        a = xp.empty(shape_a, dtype=dtype)
        b = xp.empty(shape_b, dtype=dtype)
        return xp.kron(a, b)

    @pytest.mark.parametrize(
        "shape_b", [(0,), (0, 5), (3, 0), (2, 0, 4)],
    )
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_kron_zerodim_with_empty(self, xp, dtype, shape_b):
        # 0-D scalar array × empty array — exercises the early
        # `cupy.multiply` branch rather than the empty-result short-circuit.
        a = xp.array(2, dtype=dtype)
        b = xp.empty(shape_b, dtype=dtype)
        return xp.kron(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_kron_empty_fortran_order(self, xp, dtype):
        a = xp.empty((3, 0), dtype=dtype, order='F')
        b = xp.empty((2, 4), dtype=dtype, order='F')
        return xp.kron(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_kron_empty_via_slice(self, xp, dtype):
        # Empty array produced by zero-length slicing of a non-empty buffer.
        a = xp.zeros((3, 4), dtype=dtype)[:0]
        b = xp.zeros((1, 2), dtype=dtype)
        return xp.kron(a, b)


@testing.slow
@pytest.mark.thread_unsafe(reason='Allocation too large.')
def test_integer_tensordot_large_indexing():
    # Dimensions fit int32, but the integer kernel's M * K does not.
    # This needs just over 4 GiB; uint8 also avoids the int8 cuBLAS path.
    # At this size the old bounds wrap to a small positive value, keeping
    # its reads inside the allocation instead of causing invalid accesses.
    k, m = 2**16 + 1, 2**16
    a = b = out = None
    try:
        try:
            a = cupy.zeros((2, k), dtype=cupy.uint8)
            b = cupy.zeros((k, m), dtype=cupy.uint8)
        except MemoryError:
            pytest.skip('out of memory in test.')

        # Select the second row, which is zero. The overflowing bound clamps
        # its reads to the last element of the first row, which is one.
        # A single contribution avoids masking the error via uint8 wraparound.
        a[:, 1] = 1
        b[0, :] = 1
        out = cupy.tensordot(a, b, axes=1)
        expected = numpy.zeros((2, m), dtype=numpy.uint8)
        testing.assert_array_equal(out, expected)
    finally:
        del out, b, a
        cupy.get_default_memory_pool().free_all_blocks()


class TestInt8Tensordot:
    """Smoke test for cupy.tensordot with int8 dtype via tensordot_core."""

    def setup_method(self):
        if cupy.cuda.runtime.is_hip:
            pytest.skip('int8 cublasGemmEx path is NVIDIA-only')
        if int(cupy.cuda.Device().compute_capability) < 61:
            pytest.skip(
                'CUBLAS_COMPUTE_32I requires compute capability >= 6.1')

    @testing.numpy_cupy_array_equal()
    def test_int8_tensordot_aligned(self, xp):
        """k=16 is IMMA-aligned; exercises the cuBLAS Tensor Core path."""
        rng = numpy.random.default_rng(seed=7)
        a = xp.asarray(rng.integers(-5, 5, (8, 16), dtype=numpy.int8))
        b = xp.asarray(rng.integers(-5, 5, (16, 8), dtype=numpy.int8))
        return xp.tensordot(a, b, axes=1)

    @testing.numpy_cupy_array_equal()
    def test_int8_tensordot_unaligned(self, xp):
        """k=7 is not IMMA-aligned; exercises the alignment-guard fallback."""
        rng = numpy.random.default_rng(seed=13)
        a = xp.asarray(rng.integers(-5, 5, (8, 7), dtype=numpy.int8))
        b = xp.asarray(rng.integers(-5, 5, (7, 8), dtype=numpy.int8))
        return xp.tensordot(a, b, axes=1)

    @testing.numpy_cupy_array_equal()
    def test_int8_inner_multidim(self, xp):
        """ret_shape (2, 3, 3, 2) differs from the 2-D (6, 6) GEMM output."""
        a = testing.shaped_arange((2, 3, 4), xp, numpy.int8)
        b = testing.shaped_arange((3, 2, 4), xp, numpy.int8)
        return xp.inner(a, b)

    @testing.numpy_cupy_array_equal()
    def test_int8_tensordot_multidim(self, xp):
        rng = numpy.random.default_rng(seed=21)
        a = xp.asarray(rng.integers(-5, 5, (2, 3, 8), dtype=numpy.int8))
        b = xp.asarray(rng.integers(-5, 5, (8, 4, 5), dtype=numpy.int8))
        return xp.tensordot(a, b, axes=1)


@testing.parameterize(*testing.product({
    'params': [
        ((0, 0), 2),
        ((0, 0), (1, 0)),
        ((0, 0, 0), 2),
        ((0, 0, 0), 3),
        ((0, 0, 0), ([2, 1], [0, 2])),
        ((0, 0, 0), ([0, 2, 1], [1, 2, 0])),
    ],
}))
class TestProductZeroLength(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_tensordot_zero_length(self, xp, dtype):
        shape, axes = self.params
        a = testing.shaped_arange(shape, xp, dtype)
        return xp.tensordot(a, a, axes=axes)


class TestMatrixPower(unittest.TestCase):
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_matrix_power_0(self, xp, dtype):
        a = testing.shaped_arange((3, 3), xp, dtype)
        return xp.linalg.matrix_power(a, 0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_matrix_power_1(self, xp, dtype):
        a = testing.shaped_arange((3, 3), xp, dtype)
        return xp.linalg.matrix_power(a, 1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_matrix_power_2(self, xp, dtype):
        a = testing.shaped_arange((3, 3), xp, dtype)
        return xp.linalg.matrix_power(a, 2)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_matrix_power_3(self, xp, dtype):
        a = testing.shaped_arange((3, 3), xp, dtype)
        return xp.linalg.matrix_power(a, 3)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_matrix_power_inv1(self, xp, dtype):
        a = testing.shaped_arange((3, 3), xp, dtype)
        a = a * a % 30
        return xp.linalg.matrix_power(a, -1)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_matrix_power_inv2(self, xp, dtype):
        a = testing.shaped_arange((3, 3), xp, dtype)
        a = a * a % 30
        return xp.linalg.matrix_power(a, -2)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-4)
    def test_matrix_power_inv3(self, xp, dtype):
        a = testing.shaped_arange((3, 3), xp, dtype)
        a = a * a % 30
        return xp.linalg.matrix_power(a, -3)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_matrix_power_of_two(self, xp, dtype):
        a = xp.eye(23, k=17, dtype=dtype) + xp.eye(23, k=-6, dtype=dtype)
        return xp.linalg.matrix_power(a, 1 << 50)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_matrix_power_large(self, xp, dtype):
        a = xp.eye(23, k=17, dtype=dtype) + xp.eye(23, k=-6, dtype=dtype)
        return xp.linalg.matrix_power(a, 123456789123456789)

    @pytest.mark.skipif(sys.platform == "win32",
                        reason="python int overflows C long")
    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose()
    def test_matrix_power_invlarge(self, xp, dtype):
        # TODO (ev-br): np 2.0: check if it's fixed in numpy 2 (broken on 1.26)
        a = xp.eye(23, k=17, dtype=dtype) + xp.eye(23, k=-6, dtype=dtype)
        return xp.linalg.matrix_power(a, -987654321987654321)


@pytest.mark.parametrize('shape', [
    (2, 3, 3),
    (3, 0, 0),
])
@pytest.mark.parametrize('n', [0, 5, -7])
class TestMatrixPowerBatched:

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=5e-5)
    def test_matrix_power_batched(self, xp, dtype, shape, n):
        a = testing.shaped_arange(shape, xp, dtype)
        a += xp.identity(shape[-1], dtype)
        return xp.linalg.matrix_power(a, n)


@pytest.mark.parametrize('shapes', [
    ((3, 4), (4, 5)),
    ((1, 1), (1, 1)),
    ((5, 5), (5, 5)),
    ((1, 7), (7, 1)),
])
class TestLinalgMatmul2D:

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-3, rtol=1e-3)
    def test_matmul_2d(self, xp, dtype, shapes):
        shape_a, shape_b = shapes
        a = testing.shaped_random(shape_a, xp, dtype)
        b = testing.shaped_random(shape_b, xp, dtype)
        return xp.linalg.matmul(a, b)


class TestLinalgTensordot:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_default_axes(self, xp, dtype):
        x1 = testing.shaped_arange((2, 3, 4), xp, dtype)
        x2 = testing.shaped_arange((3, 4, 5), xp, dtype)
        return xp.linalg.tensordot(x1, x2)

    @pytest.mark.parametrize('axes', [
        0,
        1,
        ([1, 2], [0, 1]),
        ([-1, -2], [-2, -3]),
    ])
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_axes(self, xp, dtype, axes):
        x1 = testing.shaped_arange((2, 3, 3), xp, dtype)
        x2 = testing.shaped_arange((3, 3, 5), xp, dtype)
        return xp.linalg.tensordot(x1, x2, axes=axes)

    def test_is_cupy_tensordot(self):
        # `cupy.linalg.tensordot` is just the Array API compatible location
        # for `cupy.tensordot`, so the two are the same object.
        assert cupy.linalg.tensordot is cupy.tensordot


class TestLinalgMatrixTranspose:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_matrix_transpose(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return xp.linalg.matrix_transpose(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal(accept_error=ValueError)
    def test_matrix_transpose_error(self, xp, dtype):
        a = testing.shaped_arange((10,), xp, dtype)
        return xp.linalg.matrix_transpose(a)


@pytest.mark.parametrize('shapes', [
    ((3,), (3,)),
    ((0,), (0,)),
    ((5, 3), (5, 3)),
    ((2, 1, 3), (4, 3)),
    ((0, 3), (3,)),
    ((3, 0), (3, 0)),
])
class TestVecdotShapes:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)
    def test_vecdot(self, xp, dtype, shapes):
        shape_a, shape_b = shapes
        # Distinct operands so a missing conjugation of x1 changes the
        # complex results.
        a = testing.shaped_random(shape_a, xp, dtype, seed=0)
        b = testing.shaped_random(shape_b, xp, dtype, seed=1)
        return xp.vecdot(a, b)


class TestVecdot:

    @testing.for_all_dtypes(name='dtype1')
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)
    def test_vecdot_dtype_combination(self, xp, dtype1, dtype2):
        a = testing.shaped_random((4, 3), xp, dtype1, seed=0)
        b = testing.shaped_random((4, 3), xp, dtype2, seed=1)
        return xp.vecdot(a, b)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)
    def test_vecdot_axis(self, xp, dtype):
        a = testing.shaped_random((3, 4), xp, dtype, seed=0)
        b = testing.shaped_random((3, 4), xp, dtype, seed=1)
        return xp.vecdot(a, b, axis=0)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)
    def test_vecdot_keepdims(self, xp, dtype):
        a = testing.shaped_random((3, 4), xp, dtype, seed=0)
        b = testing.shaped_random((3, 4), xp, dtype, seed=1)
        return xp.vecdot(a, b, keepdims=True)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)
    def test_vecdot_out(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype, seed=0)
        b = testing.shaped_random((2, 3), xp, dtype, seed=1)
        out = xp.empty((2,), dtype=dtype)
        result = xp.vecdot(a, b, out=out)
        assert result is out
        return out

    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_vecdot_out_complex_to_real(self, xp):
        # Complex inputs with a real out must accumulate in complex and
        # cast at the end (used to fail kernel compilation).
        a = testing.shaped_random((2, 3), xp, numpy.complex128, seed=0)
        b = testing.shaped_random((2, 3), xp, numpy.complex128, seed=1)
        out = xp.empty((2,), dtype=numpy.float64)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', numpy.exceptions.ComplexWarning)
            result = xp.vecdot(a, b, out=out, casting='unsafe')
        assert result is out
        return out

    @testing.for_dtypes('dD')
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_vecdot_strided(self, xp, dtype):
        # Non-contiguous 1-D inputs must not take the cuBLAS dot(c) path,
        # which assumes unit stride and would be silently wrong.
        a = testing.shaped_random((8,), xp, dtype, seed=0)[::2]
        b = testing.shaped_random((8,), xp, dtype, seed=1)[::2]
        return xp.vecdot(a, b)

    @testing.numpy_cupy_allclose(accept_error=ValueError)
    def test_vecdot_core_dim_mismatch(self, xp):
        return xp.vecdot(xp.ones((3,)), xp.ones((4,)))

    @testing.numpy_cupy_allclose(accept_error=ValueError)
    def test_vecdot_zero_dim(self, xp):
        return xp.vecdot(xp.asarray(3.0), xp.ones((4,)))


class TestLinalgVecdot:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)
    def test_vecdot(self, xp, dtype):
        a = testing.shaped_random((5, 3), xp, dtype, seed=0)
        b = testing.shaped_random((3,), xp, dtype, seed=1)
        return xp.linalg.vecdot(a, b)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)
    def test_vecdot_axis(self, xp, dtype):
        a = testing.shaped_random((3, 4), xp, dtype, seed=0)
        b = testing.shaped_random((3, 4), xp, dtype, seed=1)
        return xp.linalg.vecdot(a, b, axis=0)

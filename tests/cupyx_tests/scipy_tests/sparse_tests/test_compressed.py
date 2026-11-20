from __future__ import annotations

import unittest

import numpy
import pytest

import cupy
from cupy import testing
from cupyx.scipy import sparse


@testing.parameterize(*testing.product({
    'format': ['csr_matrix', 'csc_matrix', 'csr_array', 'csc_array'],
    'dtype': ['float32', 'float64', 'complex64', 'complex128'],
    'index_dtype': ['int32', 'int64'],
}))
class TestCompressedSum(unittest.TestCase):

    def make_matrix(self):
        # Include empty, short and long segments, unsorted duplicates, and
        # explicit zeros. The result must not require canonicalizing A.
        lengths = numpy.array([0, 1, 7, 32, 33, 65, 129, 1025, 0])
        ptr = numpy.concatenate(([0], numpy.cumsum(lengths)))
        rng = numpy.random.default_rng(42)
        data = rng.standard_normal(ptr[-1]).astype(self.dtype)
        if data.dtype.kind == 'c':
            data += 1j * rng.standard_normal(data.size).astype(self.dtype)
        data[::7] = 0
        indices = rng.integers(0, 2000, size=data.size)
        expected = numpy.array([
            data[start:end].sum() for start, end in zip(ptr[:-1], ptr[1:])])
        shape = (len(lengths), 2000)
        if self.format.startswith('csc'):
            shape = shape[::-1]
        a = getattr(sparse, self.format)(
            (cupy.asarray(data), cupy.asarray(indices), cupy.asarray(ptr)),
            shape=shape)
        # Constructors may downcast small index values. Exercise both kernel
        # specializations explicitly, including non-contiguous index storage.
        a.indices = a.indices.astype(self.index_dtype)
        a.indptr = a.indptr.astype(self.index_dtype)
        if self.format.endswith('matrix'):
            expected = expected.reshape(
                (1, -1) if self.format.startswith('csc') else (-1, 1))
        return a, expected

    def assert_result(self, actual, expected):
        tolerance = 2e-5 if self.dtype in ('float32', 'complex64') else 1e-12
        if numpy.iscomplexobj(expected):
            # Compare components separately: NumPy requires exact equality
            # of complex values containing infinity, even for a finite part.
            testing.assert_allclose(actual.real, expected.real,
                                    rtol=tolerance, atol=tolerance)
            testing.assert_allclose(actual.imag, expected.imag,
                                    rtol=tolerance, atol=tolerance)
        else:
            testing.assert_allclose(actual, expected, rtol=tolerance,
                                    atol=tolerance)
        assert actual.shape == expected.shape

    def test_sum(self):
        a, expected = self.make_matrix()
        axis = 0 if self.format.startswith('csc') else 1
        original = [x.copy() for x in (a.data, a.indices, a.indptr)]
        for selected_axis in (axis, axis - 2, (axis,), (axis - 2,)):
            result = a.sum(axis=selected_axis)
            assert result.dtype == a.dtype
            self.assert_result(result, expected)
        for actual, before in zip((a.data, a.indices, a.indptr), original):
            testing.assert_array_equal(actual, before)

    def test_strided_buffers(self):
        axis = 0 if self.format.startswith('csc') else 1
        for stride in (2, -1, -2):
            a, expected = self.make_matrix()
            for name in ('data', 'indices', 'indptr'):
                values = getattr(a, name)
                storage = cupy.empty(values.size * abs(stride), values.dtype)
                view = storage[::stride]
                view[...] = values
                setattr(a, name, view)
            self.assert_result(a.sum(axis=axis), expected)

    def test_dtype_and_out(self):
        a, expected = self.make_matrix()
        axis = 0 if self.format.startswith('csc') else 1
        dtype = 'complex128' if a.dtype.kind == 'c' else 'float64'
        result = a.sum(axis=axis, dtype=dtype)
        assert result.dtype == numpy.dtype(dtype)
        self.assert_result(result, expected)

        storage = cupy.full(expected.size * 2, -99, dtype=dtype)
        out = storage[::2].reshape(expected.shape)
        strides = out.strides
        assert a.sum(axis=axis, dtype=dtype, out=out) is out
        assert out.strides == strides
        self.assert_result(out, expected)
        testing.assert_array_equal(storage[1::2], -99)
        with pytest.raises(ValueError):
            a.sum(axis=axis, out=cupy.empty(expected.size + 1))

    def test_empty(self):
        axis = 0 if self.format.startswith('csc') else 1
        for shape in ((0, 0), (0, 5), (5, 0), (3, 5)):
            a = getattr(sparse, self.format)(shape, dtype=self.dtype)
            expected = numpy.zeros(shape, dtype=self.dtype).sum(
                axis=axis, keepdims=self.format.endswith('matrix'))
            self.assert_result(a.sum(axis=axis), expected)

    def test_long_segments(self):
        data = testing.shaped_random((5, 8193), cupy, self.dtype)
        data[0] = 0
        a = getattr(sparse, self.format)(
            data.T if self.format.startswith('csc') else data)
        a.indices = a.indices.astype(self.index_dtype)
        a.indptr = a.indptr.astype(self.index_dtype)
        axis = 0 if self.format.startswith('csc') else 1
        expected = data.get().sum(axis=1)
        if self.format.endswith('matrix'):
            expected = expected.reshape(
                (1, -1) if self.format.startswith('csc') else (-1, 1))
        self.assert_result(a.sum(axis=axis), expected)

    def test_tile_boundaries(self):
        lengths = numpy.array([0, 4095, 4096, 4097, 8191, 8192, 8193, 0, 1])
        ptr = numpy.concatenate(([0], numpy.cumsum(lengths)))
        values = (numpy.arange(ptr[-1]) % 11 - 5).astype(self.dtype)
        if values.dtype.kind == 'c':
            values += 0.5j * values
        expected = numpy.array([
            values[start:end].sum()
            for start, end in zip(ptr[:-1], ptr[1:])])
        shape = (len(lengths), 9000)
        if self.format.startswith('csc'):
            shape = shape[::-1]
        stream = cupy.cuda.Stream(non_blocking=True)
        with stream:
            a = getattr(sparse, self.format)(
                (cupy.asarray(values), cupy.arange(values.size) % 9000,
                 cupy.asarray(ptr)), shape=shape)
            a.indices = a.indices.astype(self.index_dtype)
            a.indptr = a.indptr.astype(self.index_dtype)
            for name in ('data', 'indptr'):
                original = getattr(a, name)
                storage = cupy.empty(original.size * 2, original.dtype)
                view = storage[::-2]
                view[...] = original
                setattr(a, name, view)
            axis = 0 if self.format.startswith('csc') else 1
            result = a.sum(axis=axis)
        stream.synchronize()
        testing.assert_array_equal(result.ravel(), expected)

    def test_nonfinite(self):
        a, _ = self.make_matrix()
        # Keep non-finite values in separate nonempty segments.
        a.data[0] = numpy.nan
        a.data[1] = numpy.inf
        a.data[8] = -numpy.inf
        data = a.data.get()
        ptr = a.indptr.get()
        expected = numpy.array([
            data[start:end].sum() for start, end in zip(ptr[:-1], ptr[1:])])
        if self.format.endswith('matrix'):
            expected = expected.reshape(
                (1, -1) if self.format.startswith('csc') else (-1, 1))
        axis = 0 if self.format.startswith('csc') else 1
        self.assert_result(a.sum(axis=axis), expected)

    def test_large_minor_dimension(self):
        # The workspace must depend on the stored data and output, rather
        # than allocating a dense ones vector along the reduced dimension.
        if self.index_dtype != 'int64':
            pytest.skip('requires 64-bit indices')
        width = 2**31 + 1
        shape = (3, width)
        if self.format.startswith('csc'):
            shape = shape[::-1]
        a = getattr(sparse, self.format)(
            (cupy.array([2, 3, 4], dtype=self.dtype),
             cupy.array([0, width - 1, 1], dtype=self.index_dtype),
             cupy.array([0, 2, 2, 3], dtype=self.index_dtype)), shape=shape)
        axis = 0 if self.format.startswith('csc') else 1
        pool = cupy.cuda.MemoryPool()
        pool.set_limit(size=1024 * 1024)
        with cupy.cuda.using_allocator(pool.malloc):
            result = a.sum(axis=axis)
        testing.assert_array_equal(result.ravel(), [5, 0, 4])

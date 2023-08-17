import warnings

import numpy
import pytest

import cupy
from cupy import testing
import cupyx.distributed._array


@pytest.fixture
def mem_pool():
    try:
        old_pool = cupy.get_default_memory_pool()
        pool = cupy.cuda.memory.MemoryPool()
        cupy.cuda.memory.set_allocator(pool.malloc)
        yield pool
    finally:
        pool.set_limit(size=0)
        pool.free_all_blocks()
        cupy.cuda.memory.set_allocator(old_pool.malloc)


@testing.multi_gpu(4)
class TestDistributedArray:
    shape_dim2 = (8, 8)
    mapping_dim2 = {
        0: (slice(4), slice(4)),
        1: (slice(4), slice(4, None)),
        2: (slice(4, None), slice(None, None, 2)),
        3: (slice(4, None), slice(1, None, 2))}
    mapping_dim2_2 = {
        0: (slice(None, None, 2), slice(None, None, 2)),
        1: (slice(None, None, 2), slice(1, None, 4)),
        2: (slice(None, None, 2), slice(3, None, 4)),
        3: slice(1, None, 2)}

    shape_dim3 = (4, 4, 4)
    mapping_dim3 = {
        0: slice(2),
        1: (slice(2, None), slice(None, 3)),
        2: (slice(2, None), 3, slice(None, 3)),
        3: (slice(2, None), 3, 3)
    }
    mapping_dim3_2 = {
        0: (slice(1, None, 2), 3),
        1: (slice(1, None, 2), slice(3), 3),
        2: (slice(1, None, 2), slice(3), slice(None, 3)),
        3: slice(None, None, 2),
    }

    @pytest.mark.parametrize(
            'shape, mapping',
            [(shape_dim2, mapping_dim2), (shape_dim3, mapping_dim3)])
    def test_array_creation_from_numpy(self, mem_pool, shape, mapping):
        array = numpy.arange(64, dtype='q').reshape(shape)
        assert mem_pool.used_bytes() == 0
        da = cupyx.distributed._array.distributed_array(array, mapping)
        assert da.device.id == -1
        # Ensure no memory allocation other than the chunks
        assert da.data.ptr == 0
        assert da.shape == shape
        assert mem_pool.used_bytes() == array.nbytes
        for dev, idx in mapping.items():
            assert da._chunks[dev].ndim == array.ndim
            testing.assert_array_equal(da._chunks[dev].squeeze(), array[idx])
            assert da._chunks[dev].device.id == dev

    @pytest.mark.parametrize(
            'shape, mapping',
            [(shape_dim2, mapping_dim2), (shape_dim3, mapping_dim3)])
    def test_array_creation_from_cupy(self, mem_pool, shape, mapping):
        array = cupy.arange(64, dtype='q').reshape(shape)
        assert mem_pool.used_bytes() == array.nbytes
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', cupy._util.PerformanceWarning)
            da = cupyx.distributed._array.distributed_array(array, mapping)
        assert da.device.id == -1
        # Ensure no memory allocation other than chunks & original array
        assert da.data.ptr == 0
        assert da.shape == shape
        assert mem_pool.used_bytes() == 2 * array.nbytes
        for dev, idx in mapping.items():
            assert da._chunks[dev].ndim == array.ndim
            testing.assert_array_equal(da._chunks[dev].squeeze(), array[idx])
            assert da._chunks[dev].device.id == dev

    @pytest.mark.parametrize(
            'shape, mapping',
            [(shape_dim2, mapping_dim2), (shape_dim3, mapping_dim3)])
    def test_array_creation(self, mem_pool, shape, mapping):
        array = numpy.arange(64, dtype='q').reshape(shape)
        assert mem_pool.used_bytes() == 0
        da = cupyx.distributed._array.distributed_array(array.tolist(), mapping)
        assert da.device.id == -1
        # Ensure no memory allocation other than the chunks
        assert da.data.ptr == 0
        assert da.shape == shape
        assert mem_pool.used_bytes() == array.nbytes
        for dev, idx in mapping.items():
            assert da._chunks[dev].ndim == array.ndim
            testing.assert_array_equal(da._chunks[dev].squeeze(), array[idx])
            assert da._chunks[dev].device.id == dev

    @pytest.mark.parametrize(
            'shape, mapping',
            [(shape_dim2, mapping_dim2), (shape_dim3, mapping_dim3)])
    def test_ufuncs(self, shape, mapping):
        np_a = numpy.arange(64).reshape(shape)
        np_b = numpy.arange(64).reshape(shape) * 2
        np_r = numpy.sin(np_a * np_b)
        d_a = cupyx.distributed._array.distributed_array(np_a, mapping)
        d_b = cupyx.distributed._array.distributed_array(np_b, mapping)
        d_r = cupy.sin(d_a * d_b)
        testing.assert_array_almost_equal(d_r.asnumpy(), np_r)

    @pytest.mark.parametrize(
            'shape, mapping',
            [(shape_dim2, mapping_dim2), (shape_dim3, mapping_dim3)])
    def test_elementwise_kernel(self, shape, mapping):
        custom_kernel = cupy.ElementwiseKernel(
            'float32 x, float32 y',
            'float32 z',
            'z = (x - y) * (x - y)',
            'custom')
        np_a = numpy.arange(64).reshape(shape).astype(numpy.float32)
        np_b = (numpy.arange(64).reshape(shape) * 2.0).astype(numpy.float32)
        np_r = (np_a - np_b) * (np_a - np_b)
        d_a = cupyx.distributed._array.distributed_array(np_a, mapping)
        d_b = cupyx.distributed._array.distributed_array(np_b, mapping)
        d_r = custom_kernel(d_a, d_b)
        testing.assert_array_almost_equal(d_r.asnumpy(), np_r)

    @pytest.mark.parametrize(
            'shape, mapping_a, mapping_b',
            [(shape_dim2, mapping_dim2, mapping_dim2_2),
             (shape_dim3, mapping_dim3, mapping_dim3_2)])
    def test_incompatible_chunk_shapes(self, shape, mapping_a, mapping_b):
        np_a = numpy.arange(64).reshape(shape)
        np_b = numpy.arange(64).reshape(shape) * 2
        d_a = cupyx.distributed._array.distributed_array(np_a, mapping_a)
        d_b = cupyx.distributed._array.distributed_array(np_b, mapping_b)
        with pytest.raises(RuntimeError, match=r'chunk sizes'):
            cupy.sin(d_a * d_b)

    @pytest.mark.parametrize(
            'shape, mapping_a, mapping_b',
            [(shape_dim2, mapping_dim2, mapping_dim2_2),
             (shape_dim3, mapping_dim3, mapping_dim3_2)])
    def test_incompatible_chunk_shapes_resharded(
            self, shape, mapping_a, mapping_b):
        np_a = numpy.arange(64).reshape(shape)
        np_b = numpy.arange(64).reshape(shape) * 2
        np_r = numpy.sin(np_a * np_b)
        d_a = cupyx.distributed._array.distributed_array(np_a, mapping_a)
        d_b = cupyx.distributed._array.distributed_array(np_b, mapping_b)
        d_r = cupy.sin(d_a * d_b.reshard(mapping_a))
        testing.assert_array_almost_equal(d_r.asnumpy(), np_r)

    @pytest.mark.parametrize(
            'shape, mapping',
            [(shape_dim2, mapping_dim2), (shape_dim3, mapping_dim3)])
    def test_incompatible_operand(self, shape, mapping):
        np_a = numpy.arange(64).reshape(shape)
        cp_b = cupy.arange(64).reshape(shape)
        d_a = cupyx.distributed._array.distributed_array(np_a, mapping)
        with pytest.raises(RuntimeError, match=r'Mix `cupy.ndarray'):
            cupy.sin(d_a * cp_b)

    def test_extgcd(self):
        iteration = 300
        max_value = 100

        import random
        import math
        for _ in range(iteration):
            a = random.randint(1, max_value)
            b = random.randint(1, max_value)
            g, x = cupyx.distributed._array._extgcd(a, b)
            assert g == math.gcd(a, b)
            assert (g - a * x) % b == 0

    def test_slice_intersection(self):
        iteration = 300
        max_value = 100

        import random
        for _ in range(iteration):
            a_start = random.randint(0, max_value - 1)
            b_start = random.randint(0, max_value - 1)
            a_stop = random.randint(a_start + 1, max_value)
            b_stop = random.randint(b_start + 1, max_value)
            a_step = random.randint(1, max_value)
            b_step = random.randint(1, max_value)
            a = slice(a_start, a_stop, a_step)
            b = slice(b_start, b_stop, b_step)

            def indices(s0: slice, s1: slice = slice(None)) -> set[int]:
                """Return indices for the elements of array[s0][s1]."""
                all_indices = list(range(max_value))
                return set(all_indices[s0][s1])

            c = cupyx.distributed._array._slice_intersection(
                a, b, max_value)
            if c is None:
                assert not (indices(a) & indices(b))
            else:
                assert indices(c) == indices(a) & indices(b)
                p = cupyx.distributed._array._subslice_index(a, c, max_value)
                assert indices(c) == indices(a, p)

    @pytest.mark.parametrize(
            'shape, mapping_a, mapping_b',
            [(shape_dim2, mapping_dim2, mapping_dim2_2),
             (shape_dim3, mapping_dim3, mapping_dim3_2)])
    def test_reshard(self, mem_pool, shape, mapping_a, mapping_b):
        array = numpy.arange(64, dtype='q').reshape(shape)
        assert mem_pool.used_bytes() == 0
        da = cupyx.distributed._array.distributed_array(array, mapping_a)
        assert mem_pool.used_bytes() == array.nbytes
        db = da.reshard(mapping_b)
        assert mem_pool.used_bytes() == 2 * array.nbytes
        testing.assert_array_equal(da.asnumpy(), db.asnumpy())
        for dev, idx in mapping_b.items():
            assert db._chunks[dev].ndim == array.ndim
            testing.assert_array_equal(db._chunks[dev].squeeze(), array[idx])
            assert db._chunks[dev].device.id == dev

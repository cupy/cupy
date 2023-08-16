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
    def test_array_creation_from_numpy(self, mem_pool):
        array = numpy.arange(64, dtype='q').reshape(8, 8)
        assert mem_pool.used_bytes() == 0
        device_mapping = {
            0: (slice(0, 4), slice(0, 4)),
            1: (slice(0, 4), slice(4, 8)),
            2: (slice(4, 8), slice(0, 8, 2)),
            3: (slice(4, 8), slice(1, 8, 2))}
        da = cupyx.distributed._array.distributed_array(array, device_mapping)
        assert da.device.id == -1
        # Ensure no memory allocation other than the chunks
        assert da.data.ptr == 0
        assert da.shape == (8, 8)
        assert mem_pool.used_bytes() == array.nbytes
        testing.assert_array_equal(da._chunks[0], array[:4, :4])
        testing.assert_array_equal(da._chunks[1], array[:4, 4:])
        testing.assert_array_equal(da._chunks[2], array[4:, ::2])
        testing.assert_array_equal(da._chunks[3], array[4:, 1::2])
        for dev in device_mapping:
            assert da._chunks[dev].device.id == dev

    def test_array_creation_from_cupy(self, mem_pool):
        array = cupy.arange(64, dtype='q').reshape(8, 8)
        assert mem_pool.used_bytes() == array.nbytes
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', cupy._util.PerformanceWarning)
            device_mapping = {
                0: (slice(0, 4), slice(0, 4)),
                1: (slice(0, 4), slice(4, 8)),
                2: (slice(4, 8), slice(0, 8, 2)),
                3: (slice(4, 8), slice(1, 8, 2))}
            da = cupyx.distributed._array.distributed_array(array, device_mapping)
        assert da.device.id == -1
        # Ensure no memory allocation other than chunks & original array
        assert da.data.ptr == 0
        assert da.shape == (8, 8)
        assert mem_pool.used_bytes() == 2 * array.nbytes
        testing.assert_array_equal(da._chunks[0], array[:4, :4])
        testing.assert_array_equal(da._chunks[1], array[:4, 4:])
        testing.assert_array_equal(da._chunks[2], array[4:, ::2])
        testing.assert_array_equal(da._chunks[3], array[4:, 1::2])
        for dev in device_mapping:
            assert da._chunks[dev].device.id == dev

    def test_array_creation(self, mem_pool):
        array = numpy.arange(64, dtype='q').reshape(8, 8)
        assert mem_pool.used_bytes() == 0
        device_mapping = {
            0: (slice(0, 4), slice(0, 4)),
            1: (slice(0, 4), slice(4, 8)),
            2: (slice(4, 8), slice(0, 8, 2)),
            3: (slice(4, 8), slice(1, 8, 2))}
        da = cupyx.distributed._array.distributed_array(
                array.tolist(), device_mapping)
        assert da.device.id == -1
        # Ensure no memory allocation other than the chunks
        assert da.data.ptr == 0
        assert da.shape == (8, 8)
        assert mem_pool.used_bytes() == array.nbytes
        testing.assert_array_equal(da._chunks[0], array[:4, :4])
        testing.assert_array_equal(da._chunks[1], array[:4, 4:])
        testing.assert_array_equal(da._chunks[2], array[4:, ::2])
        testing.assert_array_equal(da._chunks[3], array[4:, 1::2])
        for dev in device_mapping:
            assert da._chunks[dev].device.id == dev

    @pytest.mark.parametrize('device_mapping', [
        {0: (slice(0, 4),), 1: (slice(4, 8),)},
        {0: (slice(None), slice(0, 4)), 1: (slice(None), slice(4, 8))}])
    def test_ufuncs(self, device_mapping):
        np_a = numpy.arange(64).reshape(8, 8)
        np_b = numpy.arange(64).reshape(8, 8) * 2
        np_r = numpy.sin(np_a * np_b)
        d_a = cupyx.distributed._array.distributed_array(np_a, device_mapping)
        d_b = cupyx.distributed._array.distributed_array(np_b, device_mapping)
        d_r = cupy.sin(d_a * d_b)
        testing.assert_array_almost_equal(d_r.asnumpy(), np_r)

    @pytest.mark.parametrize('device_mapping', [
        {0: (slice(0, 4),), 1: (slice(4, 8),)},
        {0: (slice(None), slice(0, 4)), 1: (slice(None), slice(4, 8))}])
    def test_elementwise_kernel(self, device_mapping):
        custom_kernel = cupy.ElementwiseKernel(
            'float32 x, float32 y',
            'float32 z',
            'z = (x - y) * (x - y)',
            'custom')
        np_a = numpy.arange(64).reshape(8, 8).astype(numpy.float32)
        np_b = (numpy.arange(64).reshape(8, 8) * 2.0).astype(numpy.float32)
        np_r = (np_a - np_b) * (np_a - np_b)
        d_a = cupyx.distributed._array.distributed_array(np_a, device_mapping)
        d_b = cupyx.distributed._array.distributed_array(np_b, device_mapping)
        d_r = custom_kernel(d_a, d_b)
        testing.assert_array_almost_equal(d_r.asnumpy(), np_r)

    def test_incompatible_chunk_shapes(self):
        np_a = numpy.arange(64).reshape(8, 8)
        np_b = numpy.arange(64).reshape(8, 8) * 2
        mapping_a = {0: (slice(0, 4),), 1: (slice(4, 8),)}
        mapping_b = {0: (slice(None), slice(0, 4)), 1: (slice(None), slice(4, 8))}
        d_a = cupyx.distributed._array.distributed_array(np_a, mapping_a)
        d_b = cupyx.distributed._array.distributed_array(np_b, mapping_b)
        with pytest.raises(RuntimeError, match=r'chunk sizes'):
            cupy.sin(d_a * d_b)

    def test_incompatible_chunk_shapes_resharded(self):
        np_a = numpy.arange(64).reshape(8, 8)
        np_b = numpy.arange(64).reshape(8, 8) * 2
        np_r = numpy.sin(np_a * np_b)
        mapping_a = {0: (slice(0, 4),), 1: (slice(4, 8),)}
        mapping_b = {0: (slice(None), slice(0, 4)), 1: (slice(None), slice(4, 8))}
        d_a = cupyx.distributed._array.distributed_array(np_a, mapping_a)
        d_b = cupyx.distributed._array.distributed_array(np_b, mapping_b)
        d_r = cupy.sin(d_a * d_b.reshard(mapping_a))
        testing.assert_array_almost_equal(d_r.asnumpy(), np_r)

    def test_incompatible_operand(self):
        np_a = numpy.arange(64).reshape(8, 8)
        cp_b = cupy.arange(64).reshape(8, 8)
        mapping_a = {0: slice(0, 4), 1: slice(4, 8)}
        d_a = cupyx.distributed._array.distributed_array(np_a, mapping_a)
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

    def test_reshard(self, mem_pool):
        array = numpy.arange(64, dtype='q').reshape(8, 8)
        assert mem_pool.used_bytes() == 0
        device_mapping = {
            0: (slice(0, 4), slice(0, 4)),
            1: (slice(0, 4), slice(4, 8)),
            2: (slice(4, 8), slice(0, 8, 2)),
            3: (slice(4, 8), slice(1, 8, 2))}
        da = cupyx.distributed._array.distributed_array(array, device_mapping)
        assert mem_pool.used_bytes() == array.nbytes
        new_device_mapping = {
            0: (slice(0, 8, 2), slice(0, 8, 2)),
            1: (slice(0, 8, 2), slice(1, 8, 4)),
            2: (slice(0, 8, 2), slice(3, 8, 4)),
            3: (slice(1, 8, 2),)}
        db = da.reshard(new_device_mapping)
        assert mem_pool.used_bytes() == 2 * array.nbytes
        testing.assert_array_equal(da.asnumpy(), db.asnumpy())
        testing.assert_array_equal(db._chunks[0], array[::2, ::2])
        testing.assert_array_equal(db._chunks[1], array[::2, 1::4])
        testing.assert_array_equal(db._chunks[2], array[::2, 3::4])
        testing.assert_array_equal(db._chunks[3], array[1::2])
        for dev in device_mapping:
            assert db._chunks[dev].device.id == dev

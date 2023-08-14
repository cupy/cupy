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
            2: (slice(4, 8), slice(0, 4)),
            3: (slice(4, 8), slice(4, 8))}
        da = cupyx.distributed._array.distributed_array(array, device_mapping)
        assert da.device.id == -1
        # Ensure no memory allocation other than the chunks
        assert da.data.ptr == 0
        assert da.shape == (8, 8)
        assert mem_pool.used_bytes() == array.nbytes
        testing.assert_array_equal(da._chunks[0], array[:4, :4])
        testing.assert_array_equal(da._chunks[1], array[:4, 4:])
        testing.assert_array_equal(da._chunks[2], array[4:, :4])
        testing.assert_array_equal(da._chunks[3], array[4:, 4:])
        for dev in range(4):
            assert da._chunks[i] == i

    def test_array_creation_from_cupy(self, mem_pool):
        array = cupy.arange(64, dtype='q').reshape(8, 8)
        assert mem_pool.used_bytes() == array.nbytes
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', cupy._util.PerformanceWarning)
            device_mapping = {
                0: (slice(0, 4), slice(0, 4)),
                1: (slice(0, 4), slice(4, 8)),
                2: (slice(4, 8), slice(0, 4)),
                3: (slice(4, 8), slice(4, 8))}
            da = cupyx.distributed._array.distributed_array(array, device_mapping)
        assert da.device.id == -1
        # Ensure no memory allocation other than chunks & original array
        assert da.data.ptr == 0
        assert da.shape == (8, 8)
        assert mem_pool.used_bytes() == 2 * array.nbytes
        testing.assert_array_equal(da._chunks[0], array[:4, :4])
        testing.assert_array_equal(da._chunks[1], array[:4, 4:])
        testing.assert_array_equal(da._chunks[2], array[4:, :4])
        testing.assert_array_equal(da._chunks[3], array[4:, 4:])
        for dev in range(4):
            assert da._chunks[i] == i

    def test_array_creation(self, mem_pool):
        array = numpy.arange(64, dtype='q').reshape(8, 8)
        assert mem_pool.used_bytes() == 0
        device_mapping = {
            0: (slice(0, 4), slice(0, 4)),
            1: (slice(0, 4), slice(4, 8)),
            2: (slice(4, 8), slice(0, 4)),
            3: (slice(4, 8), slice(4, 8))}
        da = cupyx.distributed._array.distributed_array(
                array.tolist(), device_mapping)
        assert da.device.id == -1
        # Ensure no memory allocation other than the chunks
        assert da.data.ptr == 0
        assert da.shape == (8, 8)
        assert mem_pool.used_bytes() == array.nbytes
        testing.assert_array_equal(da._chunks[0], array[:4, :4])
        testing.assert_array_equal(da._chunks[1], array[:4, 4:])
        testing.assert_array_equal(da._chunks[2], array[4:, :4])
        testing.assert_array_equal(da._chunks[3], array[4:, 4:])
        for dev in range(4):
            assert da._chunks[i] == i

    @pytest.mark.parametrize('device_mapping', [
        {0: slice(0, 4), 1: slice(4, 8)},
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
        {0: slice(0, 4), 1: slice(4, 8)},
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
        mapping_a = {0: slice(0, 4), 1: slice(4, 8)}
        mapping_b = {0: (slice(None), slice(0, 4)), 1: (slice(None), slice(4, 8))}
        d_a = cupyx.distributed._array.distributed_array(np_a, mapping_a)
        d_b = cupyx.distributed._array.distributed_array(np_b, mapping_b)
        with pytest.raises(RuntimeError, match=r'chunk sizes'):
            cupy.sin(d_a * d_b)

    def test_incompatible_operand(self):
        np_a = numpy.arange(64).reshape(8, 8)
        cp_b = cupy.arange(64).reshape(8, 8)
        mapping_a = {0: slice(0, 4), 1: slice(4, 8)}
        d_a = cupyx.distributed._array.distributed_array(np_a, mapping_a)
        with pytest.raises(RuntimeError, match=r'Mix `cupy.ndarray'):
            cupy.sin(d_a * cp_b)

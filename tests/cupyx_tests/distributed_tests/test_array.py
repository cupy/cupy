import warnings

import numpy
import pytest

import cupy
from cupy.cuda import nccl
from cupy import testing
import cupyx.distributed._array as _array


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


def make_comms():
    if not nccl.available:
        return None
    comms_list = nccl.NcclCommunicator.initAll(4)
    return {dev: comm for dev, comm in zip(range(4), comms_list)}


comms = make_comms()


@testing.multi_gpu(4)
class TestDistributedArray:
    shape_dim2 = (8, 8)
    mapping_dim2 = {
        0: (slice(5), slice(5)),
        1: (slice(5), slice(3, None)),
        2: (slice(3, None), slice(None, None, 2)),
        3: (slice(3, None), slice(1, None, 2))}
    mapping_dim2_2 = {
        0: (slice(None, None, 2), slice(None, None, 2)),
        1: (slice(None, None, 2), slice(1, 5, 2)),
        2: (slice(None, None, 2), slice(3, None, 2)),
        3: slice(1, None, 2)}

    shape_dim3 = (4, 4, 4)
    mapping_dim3 = {
        0: slice(2),
        1: (slice(2, None), slice(None, 3)),
        2: (slice(2, None), 3),
        3: (slice(2, None), slice(None), 3)
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
    @pytest.mark.parametrize('mode', ['replica', 'sum', 'max'])
    def test_array_creation_from_numpy(self, mem_pool, shape, mapping, mode):
        array = numpy.arange(64, dtype='q').reshape(shape)
        assert mem_pool.used_bytes() == 0
        da = _array.distributed_array(array, mapping, mode, comms)
        assert da.device.id == -1
        # Ensure no memory allocation other than the chunks
        assert da.data.ptr == 0
        assert da.shape == shape
        assert mem_pool.used_bytes() == array.nbytes
        for dev, idx in mapping.items():
            assert da._chunks[dev].data.device.id == dev
            assert da._chunks[dev].data.ndim == array.ndim
            if mode == 'replica':
                testing.assert_array_equal(da._chunks[dev].data.squeeze(), array[idx])

    @pytest.mark.parametrize(
            'shape, mapping',
            [(shape_dim2, mapping_dim2), (shape_dim3, mapping_dim3)])
    @pytest.mark.parametrize('mode', ['replica', 'sum'])
    def test_array_creation_from_cupy(self, mem_pool, shape, mapping, mode):
        array = cupy.arange(64, dtype='q').reshape(shape)
        assert mem_pool.used_bytes() == array.nbytes
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', cupy._util.PerformanceWarning)
            da = _array.distributed_array(
                array, mapping, mode, comms)
        assert da.device.id == -1
        # Ensure no memory allocation other than chunks & original array
        assert da.data.ptr == 0
        assert da.shape == shape
        assert mem_pool.used_bytes() == 2 * array.nbytes
        for dev, idx in mapping.items():
            assert da._chunks[dev].data.device.id == dev
            assert da._chunks[dev].data.ndim == array.ndim
            if mode == 'replica':
                testing.assert_array_equal(da._chunks[dev].data.squeeze(), array[idx])

    @pytest.mark.parametrize(
            'shape, mapping',
            [(shape_dim2, mapping_dim2), (shape_dim3, mapping_dim3)])
    @pytest.mark.parametrize('mode', ['replica', 'sum'])
    def test_array_creation(self, mem_pool, shape, mapping, mode):
        array = numpy.arange(64, dtype='q').reshape(shape)
        assert mem_pool.used_bytes() == 0
        da = _array.distributed_array(
            array.tolist(), mapping, mode, comms)
        assert da.device.id == -1
        # Ensure no memory allocation other than the chunks
        assert da.data.ptr == 0
        assert da.shape == shape
        assert mem_pool.used_bytes() == array.nbytes
        for dev, idx in mapping.items():
            assert da._chunks[dev].data.device.id == dev
            assert da._chunks[dev].data.ndim == array.ndim
            if mode == 'replica':
                testing.assert_array_equal(
                    da._chunks[dev].data.squeeze(), array[idx])

    @pytest.mark.parametrize(
            'shape, mapping',
            [(shape_dim2, mapping_dim2), (shape_dim3, mapping_dim3)])
    def test_change_mode(self, shape, mapping):
        np_a = numpy.zeros(shape)
        cp_chunks = {}
        for dev, idx in mapping.items():
            np_a[idx] += 1 << dev
            with cupy.cuda.Device(dev):
                cp_chunks[dev] = _array._ControlledData(
                    cupy.full_like(np_a[idx], 1 << dev))
            for dev, idx in mapping.items():
                new_idx = _array._convert_chunk_idx_to_slices(
                    shape, idx)
                mapping[dev] = new_idx

        d_a = _array._DistributedArray(
            shape, np_a.dtype, cp_chunks, mapping, _array._MODES['sum'], comms)
        d_a = d_a.to_replica_mode()
        testing.assert_array_equal(d_a.asnumpy(), np_a)
        for dev, idx in mapping.items():
            assert d_a._chunks[dev].data.device.id == dev
            testing.assert_array_equal(d_a._chunks[dev].data, np_a[idx])

    @pytest.mark.parametrize(
            'shape, mapping',
            [(shape_dim2, mapping_dim2), (shape_dim3, mapping_dim3)])
    @pytest.mark.parametrize('mode_a', ['replica', 'sum'])
    @pytest.mark.parametrize('mode_b', ['replica', 'sum'])
    def test_ufuncs(self, shape, mapping, mode_a, mode_b):
        np_a = numpy.arange(64).reshape(shape)
        np_b = numpy.arange(64).reshape(shape) * 2
        np_r = numpy.cos(np_a * np_b) # We do not choose sin because sin(0) == 0
        d_a = _array.distributed_array(np_a, mapping, mode_a, comms)
        d_b = _array.distributed_array(np_b, mapping, mode_b, comms)
        d_r = cupy.cos(d_a * d_b)
        testing.assert_array_almost_equal(d_r.asnumpy(), np_r)

    @pytest.mark.parametrize(
            'shape, mapping',
            [(shape_dim2, mapping_dim2), (shape_dim3, mapping_dim3)])
    @pytest.mark.parametrize('mode_a', ['replica', 'sum'])
    @pytest.mark.parametrize('mode_b', ['replica', 'sum'])
    def test_elementwise_kernel(self, shape, mapping, mode_a, mode_b):
        custom_kernel = cupy.ElementwiseKernel(
            'float32 x, float32 y',
            'float32 z',
            'z = (x - y) * (x - y)',
            'custom')
        np_a = numpy.arange(64).reshape(shape).astype(numpy.float32)
        np_b = (numpy.arange(64).reshape(shape) * 2.0).astype(numpy.float32)
        np_r = (np_a - np_b) * (np_a - np_b)
        d_a = _array.distributed_array(np_a, mapping, mode_a, comms)
        d_b = _array.distributed_array(np_b, mapping, mode_b, comms)
        d_r = custom_kernel(d_a, d_b)
        testing.assert_array_almost_equal(d_r.asnumpy(), np_r)

    @pytest.mark.parametrize(
            'shape, mapping_a, mapping_b',
            [(shape_dim2, mapping_dim2, mapping_dim2_2),
             (shape_dim3, mapping_dim3, mapping_dim3_2)])
    @pytest.mark.parametrize('mode', ['replica', 'sum'])
    def test_incompatible_chunk_shapes(self, shape, mapping_a, mapping_b, mode):
        np_a = numpy.arange(64).reshape(shape)
        np_b = numpy.arange(64).reshape(shape) * 2
        d_a = _array.distributed_array(np_a, mapping_a, mode, comms)
        d_b = _array.distributed_array(np_b, mapping_b, mode, comms)
        with pytest.raises(RuntimeError, match=r'different chunk sizes'):
            cupy.cos(d_a * d_b)

    @pytest.mark.parametrize(
            'shape, mapping',
            [(shape_dim2, mapping_dim2), (shape_dim3, mapping_dim3)])
    @pytest.mark.parametrize('mode', ['replica', 'sum'])
    def test_incompatible_operand(self, shape, mapping, mode):
        np_a = numpy.arange(64).reshape(shape)
        cp_b = cupy.arange(64).reshape(shape)
        d_a = _array.distributed_array(np_a, mapping, mode, comms)
        with pytest.raises(RuntimeError, match=r'Mix `cupy.ndarray'):
            cupy.cos(d_a * cp_b)

    def test_extgcd(self):
        iteration = 300
        max_value = 100

        import random
        import math
        for _ in range(iteration):
            a = random.randint(1, max_value)
            b = random.randint(1, max_value)
            g, x = _array._extgcd(a, b)
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

            c = _array._slice_intersection(
                a, b, max_value)
            if c is None:
                assert not (indices(a) & indices(b))
            else:
                assert indices(c) == indices(a) & indices(b)
                p = _array._index_for_subslice(
                        a, c, max_value)
                assert indices(c) == indices(a, p)

    @pytest.mark.parametrize(
            'shape, mapping_a, mapping_b',
            [(shape_dim2, mapping_dim2, mapping_dim2_2),
             (shape_dim3, mapping_dim3, mapping_dim3_2)])
    @pytest.mark.parametrize('mode', ['replica', 'sum'])
    def test_reshard(self, mem_pool, shape, mapping_a, mapping_b, mode):
        array = numpy.arange(64, dtype='q').reshape(shape)
        assert mem_pool.used_bytes() == 0
        da = _array.distributed_array(array, mapping_a, mode, comms)
        assert mem_pool.used_bytes() == array.nbytes
        db = da.reshard(mapping_b)
        testing.assert_array_equal(da.asnumpy(), db.asnumpy())
        for dev, idx in mapping_b.items():
            assert db._chunks[dev].data.device.id == dev
            assert db._chunks[dev].data.ndim == array.ndim
            if mode == 'replica':
                testing.assert_array_equal(
                    db._chunks[dev].data.squeeze(), array[idx])

    @pytest.mark.parametrize(
            'shape, mapping_a, mapping_b',
            [(shape_dim2, mapping_dim2, mapping_dim2_2),
             (shape_dim3, mapping_dim3, mapping_dim3_2)])
    @pytest.mark.parametrize('mode', ['replica', 'sum'])
    def test_incompatible_chunk_shapes_resharded(
            self, shape, mapping_a, mapping_b, mode):
        np_a = numpy.arange(64).reshape(shape)
        np_b = numpy.arange(64).reshape(shape) * 2
        np_r = numpy.cos(np_a + np_b)
        d_a = _array.distributed_array(np_a, mapping_a, mode, comms)
        d_b = _array.distributed_array(np_b, mapping_b, mode, comms)
        d_c = d_a + d_b.reshard(mapping_a)
        d_r = cupy.cos(d_c.reshard(mapping_b))
        testing.assert_array_almost_equal(d_r.asnumpy(), np_r)

    @pytest.mark.parametrize(
            'shape, mapping',
            [(shape_dim2, mapping_dim2), (shape_dim3, mapping_dim3)])
    @pytest.mark.parametrize('mode', ['replica', 'sum'])
    @pytest.mark.parametrize('dtype', ['int64', 'float64'])
    def test_max_reduction(self, shape, mapping, mode, dtype):
        np_a = numpy.arange(64, dtype=dtype).reshape(shape)
        d_a = _array.distributed_array(np_a, mapping, mode, comms)
        for axis in range(np_a.ndim):
            np_b = np_a.max(axis=axis)
            d_b = d_a.max(axis=axis)
            testing.assert_array_equal(d_b.asnumpy(), np_b)

    @pytest.mark.parametrize(
            'shape, mapping',
            [(shape_dim2, mapping_dim2), (shape_dim3, mapping_dim3)])
    @pytest.mark.parametrize('mode', ['replica', 'sum'])
    @pytest.mark.parametrize('dtype', ['int64', 'float64'])
    def test_min_reduction(self, shape, mapping, mode, dtype):
        np_a = numpy.arange(64, dtype=dtype).reshape(shape)
        d_a = _array.distributed_array(np_a, mapping, mode, comms)
        for axis in range(np_a.ndim):
            np_b = np_a.min(axis=axis)
            d_b = d_a.min(axis=axis)
            testing.assert_array_equal(d_b.asnumpy(), np_b)

    @pytest.mark.parametrize('shape, mapping', [(shape_dim3, mapping_dim3)])
    @pytest.mark.parametrize('mode', ['replica', 'sum', 'prod'])
    def test_sum_reduction(self, shape, mapping, mode):
        np_a = numpy.arange(64).reshape(shape)
        d_a = _array.distributed_array(np_a, mapping, mode, comms)
        for axis in range(np_a.ndim):
            np_b = np_a.sum(axis=axis)
            d_b = d_a.sum(axis=axis)
            assert d_b._mode is _array._MODES['sum']
            testing.assert_array_equal(d_b.asnumpy(), np_b)

    @pytest.mark.parametrize('shape, mapping', [(shape_dim3, mapping_dim3)])
    @pytest.mark.parametrize('mode', ['replica', 'sum'])
    def test_prod_reduction(self, shape, mapping, mode):
        np_a = numpy.arange(64, dtype=cupy.float64).reshape(shape)
        d_a = _array.distributed_array(np_a, mapping, mode, comms)
        for axis in range(np_a.ndim):
            np_b = np_a.prod(axis=axis)
            d_b = d_a.prod(axis=axis)
            testing.assert_array_equal(d_b.asnumpy(), np_b)

    @pytest.mark.parametrize('shape, mapping', [(shape_dim3, mapping_dim3)])
    def test_unsupported_reduction(self, shape, mapping):
        np_a = numpy.arange(64).reshape(shape)
        d_a = _array.distributed_array(np_a, mapping, 'replica', comms)
        with pytest.raises(RuntimeError, match=r'Unsupported .* cupy_argmax'):
            d_a.argmax(axis=0)

    @pytest.mark.parametrize(
            'shape, mapping_a, mapping_b',
            [(shape_dim2, mapping_dim2, mapping_dim2_2),
             (shape_dim3, mapping_dim3, mapping_dim3_2)])
    def test_reshard_max(self, shape, mapping_a, mapping_b):
        np_a = numpy.arange(64).reshape(shape)
        np_b = np_a.max(axis=0)
        d_a = _array.distributed_array(np_a, mapping_a)
        d_b = d_a.reshard(mapping_b).max(axis=0)
        testing.assert_array_equal(np_b, d_b.asnumpy())

    @pytest.mark.parametrize(
            'shape, mapping_a, mapping_b',
            [(shape_dim2, mapping_dim2, mapping_dim2_2),
             (shape_dim3, mapping_dim3, mapping_dim3_2)])
    def test_mul_max_mul(self, shape, mapping_a, mapping_b):
        rng = numpy.random.default_rng(seed=42)
        np_a = rng.integers(0, 1 << 3, shape)
        np_b = rng.integers(0, 1 << 3, shape)
        np_c = rng.integers(0, 1 << 3, shape[1:])
        np_c2 = (np_a * np_b).max(axis=0)
        np_d = (np_a * np_b).max(axis=0) * np_c
        mapping_c = {dev: idx[1:] for dev, idx in mapping_a.items()}
        d_a = _array.distributed_array(np_a, mapping_a)
        d_b = _array.distributed_array(np_b, mapping_b)
        d_c = _array.distributed_array(np_c, mapping_c)
        print(np_a, np_b, (np_a * np_b), np_c2, np_c, sep='\n')
        d_c2 = (d_a.reshard(mapping_b) * d_b).max(axis=0)
        # print(vars(d_c2))
        d_d = d_c2.reshard(mapping_c) * d_c
        # print(vars(d_d))
        testing.assert_array_equal(np_c2, d_c2.asnumpy())
        testing.assert_array_equal(np_d, d_d.asnumpy())

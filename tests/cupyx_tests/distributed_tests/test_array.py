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
        pool.set_limit(16 << 20)
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


size = 262144


shape_dim2 = (512, 512)
index_map_dim2 = {
    0: [(slice(300), slice(300)),
        (slice(300), slice(200, None))],
    1: [(slice(200, None), slice(None, None, 2)),
        (slice(200, None), slice(1, None, 4)),
        (slice(200, None), slice(3, None, 4))],
    3: [(slice(200, None, 3), slice(None, None, 3))],
}
index_map_dim2_2 = {
    0: [(slice(None, None, 2), slice(None, None, 2))],
    2: [(slice(None, None, 2), slice(1, 200, 2)),
        (slice(None, None, 2), slice(101, 300, 2))],
    3: [(slice(None, None, 2), slice(201, None, 2)),
        slice(1, None, 2)],
}

shape_dim3 = (64, 64, 64)
index_map_dim3 = {
    0: [slice(32)],
    1: [(slice(32, None), slice(None, 63))],
    2: [(slice(32, None), 63)],
    3: [(slice(32, None), slice(None), 42)],
}
index_map_dim3_2 = {
    1: [(slice(1, None, 2), 0),
        (slice(1, None, 2), slice(1, None), 63)],
    2: [(slice(1, None, 2), slice(1, None), slice(None, 63))],
    3: [slice(None, None, 2)],
}

index_map_only_1 = {
    1: [(slice(None),)],
}


@testing.multi_gpu(4)
class TestDistributedArray:
    @pytest.mark.parametrize(
            'shape, index_map',
            [(shape_dim2, index_map_dim2), (shape_dim3, index_map_dim3)])
    @pytest.mark.parametrize('mode', ['replica', 'sum', 'max'])
    def test_array_creation_from_numpy(self, mem_pool, shape, index_map, mode):
        array = numpy.arange(size, dtype='q').reshape(shape)
        # assert mem_pool.used_bytes() == 0
        da = _array.distributed_array(array, index_map, mode, comms)
        assert da.device.id == -1
        # Ensure no memory allocation other than the chunks
        assert da.data.ptr == 0
        assert da.shape == shape
        # assert mem_pool.used_bytes() == array.nbytes
        for dev, idxs in index_map.items():
            for i, idx in enumerate(idxs):
                assert da._get_chunk_data(dev, i).device.id == dev
                assert da._get_chunk_data(dev, i).ndim == array.ndim
                if mode == 'replica':
                    testing.assert_array_equal(
                        da._get_chunk_data(dev, i).ravel(), array[idx].ravel(),
                        strict=True)

    @pytest.mark.parametrize(
            'shape, index_map',
            [(shape_dim2, index_map_dim2), (shape_dim3, index_map_dim3)])
    @pytest.mark.parametrize('mode', ['replica', 'sum', 'max'])
    def test_array_creation_from_cupy(self, mem_pool, shape, index_map, mode):
        array = cupy.arange(size, dtype='q').reshape(shape)
        # assert mem_pool.used_bytes() == array.nbytes
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', cupy._util.PerformanceWarning)
            da = _array.distributed_array(
                array, index_map, mode, comms)
        assert da.device.id == -1
        # Ensure no memory allocation other than chunks & original array
        assert da.data.ptr == 0
        assert da.shape == shape
        # assert mem_pool.used_bytes() == 2 * array.nbytes
        for dev, idxs in index_map.items():
            for i, idx in enumerate(idxs):
                assert da._get_chunk_data(dev, i).device.id == dev
                assert da._get_chunk_data(dev, i).ndim == array.ndim
                if mode == 'replica':
                    testing.assert_array_equal(
                        da._get_chunk_data(dev, i).ravel(), array[idx].ravel(),
                        strict=True)

    @pytest.mark.parametrize(
            'shape, index_map',
            [(shape_dim2, index_map_dim2), (shape_dim3, index_map_dim3)])
    @pytest.mark.parametrize('mode', ['replica', 'sum', 'max'])
    def test_array_creation(self, mem_pool, shape, index_map, mode):
        array = numpy.arange(size, dtype='q').reshape(shape)
        # assert mem_pool.used_bytes() == 0
        da = _array.distributed_array(
            array.tolist(), index_map, mode, comms)
        assert da.device.id == -1
        # Ensure no memory allocation other than the chunks
        assert da.data.ptr == 0
        assert da.shape == shape
        # assert mem_pool.used_bytes() == array.nbytes
        for dev, idxs in index_map.items():
            for i, idx in enumerate(idxs):
                assert da._get_chunk_data(dev, i).device.id == dev
                assert da._get_chunk_data(dev, i).ndim == array.ndim
                if mode == 'replica':
                    testing.assert_array_equal(
                        da._get_chunk_data(dev, i).ravel(), array[idx].ravel(),
                        strict=True)

    @pytest.mark.parametrize(
            'shape, index_map',
            [(shape_dim2, index_map_dim2), (shape_dim2, index_map_dim2_2),
             (shape_dim3, index_map_dim3), (shape_dim3, index_map_dim3_2)])
    def test_array_creation_comms(self, mem_pool, shape, index_map):
        array = numpy.arange(size, dtype='q').reshape(shape)
        # assert mem_pool.used_bytes() == 0
        da = _array.distributed_array(array, index_map)
        assert da._comms.keys() == index_map.keys()
        da = _array.distributed_array(array, index_map, devices=set(range(4)))
        assert da._comms.keys() == set(range(4))
        da = _array.distributed_array(
            array, index_map, comms=_array._create_communicators(range(4)))
        assert da._comms.keys() == set(range(4))
        with pytest.raises(RuntimeError, match='Only one of devices and comms'):
            da = _array.distributed_array(
                array, index_map, devices=set(), comms={})

    @pytest.mark.parametrize(
            'shape, index_map',
            [(shape_dim2, index_map_dim2),
             (shape_dim3, index_map_dim3),
             (shape_dim3, index_map_only_1)])
    def test_change_to_replica_mode(self, shape, index_map):
        np_a = numpy.zeros(shape)
        chunks_map = {}
        for dev, idxs in index_map.items():
            chunks_map[dev] = []
            for idx in idxs:
                np_a[idx] += 1 << dev

                full_idx = _array._convert_chunk_idx_to_slices(shape, idx)
                with cupy.cuda.Device(dev):
                    chunk = _array._Chunk(
                        cupy.full_like(np_a[idx], 1 << dev),
                        cupy.cuda.Event(), full_idx)
                    chunks_map[dev].append(chunk)

        d_a = _array._DistributedArray(
            shape, np_a.dtype, chunks_map, _array._MODES['sum'], comms)
        d_b = d_a.to_replica_mode()
        assert d_b._mode is _array._REPLICA_MODE
        testing.assert_array_equal(d_b.asnumpy(), np_a, strict=True)
        testing.assert_array_equal(d_a.asnumpy(), np_a, strict=True)
        for dev, idxs in index_map.items():
            for i, idx in enumerate(idxs):
                assert d_b._get_chunk_data(dev, i).device.id == dev
                idx = _array._convert_chunk_idx_to_slices(shape, idx)
                testing.assert_array_equal(
                    d_b._get_chunk_data(dev, i), np_a[idx], strict=True)

    @pytest.mark.parametrize(
            'shape, index_map',
            [(shape_dim2, index_map_dim2),
             (shape_dim3, index_map_dim3),
             (shape_dim3, index_map_only_1)])
    @pytest.mark.parametrize('mode', ['max', 'sum'])
    def test_change_to_op_mode(self, shape, index_map, mode):
        np_a = numpy.arange(size).reshape(shape)
        d_a = _array.distributed_array(np_a, index_map, mode, comms)
        d_b = d_a.change_mode(mode)
        assert d_b.mode == mode
        testing.assert_array_equal(d_b.asnumpy(), np_a, strict=True)
        testing.assert_array_equal(d_a.asnumpy(), np_a, strict=True)

    @pytest.mark.parametrize(
            'shape, index_map',
            [(shape_dim2, index_map_dim2), (shape_dim3, index_map_dim3)])
    @pytest.mark.parametrize('mode_a', ['replica', 'sum'])
    @pytest.mark.parametrize('mode_b', ['replica', 'sum'])
    def test_ufuncs(self, shape, index_map, mode_a, mode_b):
        np_a = numpy.arange(size).reshape(shape)
        np_b = numpy.arange(size).reshape(shape) * 2
        np_r = numpy.cos(np_a * np_b) # We do not choose sin because sin(0) == 0
        d_a = _array.distributed_array(np_a, index_map, mode_a, comms)
        d_b = _array.distributed_array(np_b, index_map, mode_b, comms)
        d_r = cupy.cos(d_a * d_b)
        testing.assert_array_almost_equal(d_r.asnumpy(), np_r)

    @pytest.mark.parametrize(
            'shape, index_map',
            [(shape_dim2, index_map_dim2), (shape_dim3, index_map_dim3)])
    @pytest.mark.parametrize('mode_a', ['replica', 'sum'])
    @pytest.mark.parametrize('mode_b', ['replica', 'sum'])
    def test_elementwise_kernel(self, shape, index_map, mode_a, mode_b):
        custom_kernel = cupy.ElementwiseKernel(
            'float32 x, float32 y',
            'float32 z',
            'z = (x - y) * (x - y)',
            'custom')
        np_a = numpy.arange(size).reshape(shape).astype(numpy.float32)
        np_b = (numpy.arange(size).reshape(shape) * 2.0).astype(numpy.float32)
        np_r = (np_a - np_b) * (np_a - np_b)
        d_a = _array.distributed_array(np_a, index_map, mode_a, comms)
        d_b = _array.distributed_array(np_b, index_map, mode_b, comms)
        d_r = custom_kernel(d_a, d_b)
        testing.assert_array_almost_equal(d_r.asnumpy(), np_r)

    @pytest.mark.parametrize(
            'shape, mapping',
            [(shape_dim2, index_map_dim2), (shape_dim3, index_map_dim3)])
    @pytest.mark.parametrize('mode', ['replica', 'sum', 'max'])
    def test_incompatible_chunk_shapes(self, shape, mapping, mode):
        mapping_a = {}
        for dev, idxs in mapping.items():
            mapping_a.setdefault(dev % 2, []).extend(idxs)
        mapping_b = {0: mapping_a[1],
                     1: mapping_a[0]}

        np_a = numpy.arange(size).reshape(shape)
        np_b = numpy.arange(size).reshape(shape) * 2
        np_r = numpy.cos(np_a * np_b)
        d_a = _array.distributed_array(np_a, mapping_a, mode, comms)
        d_b = _array.distributed_array(np_b, mapping_b, mode, comms)
        with pytest.warns(cupy._util.PerformanceWarning, match=r'Peer access'):
            d_r = cupy.cos(d_a * d_b)
        testing.assert_array_almost_equal(d_r.asnumpy(), np_r)

    @pytest.mark.parametrize(
            'shape, index_map',
            [(shape_dim2, index_map_dim2), (shape_dim3, index_map_dim3)])
    @pytest.mark.parametrize('mode', ['replica', 'sum', 'max'])
    def test_incompatible_operand(self, shape, index_map, mode):
        np_a = numpy.arange(size).reshape(shape)
        cp_b = cupy.arange(size).reshape(shape)
        d_a = _array.distributed_array(np_a, index_map, mode, comms)
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
            [(shape_dim2, index_map_dim2, index_map_dim2_2),
             (shape_dim3, index_map_dim3, index_map_dim3_2),
             (shape_dim3, index_map_only_1, index_map_dim3)])
    @pytest.mark.parametrize('mode', ['replica', 'sum', 'max'])
    def test_reshard(self, mem_pool, shape, mapping_a, mapping_b, mode):
        np_a = numpy.arange(size, dtype='q').reshape(shape)
        # assert mem_pool.used_bytes() == 0
        # Initialize without comms
        d_a = _array.distributed_array(np_a, mapping_a, mode)
        # assert mem_pool.used_bytes() == np_a.nbytes
        d_b = d_a.reshard(mapping_b)
        testing.assert_array_equal(d_b.asnumpy(), np_a, strict=True)
        testing.assert_array_equal(d_a.asnumpy(), np_a, strict=True)
        assert d_b.mode == mode
        for dev, idxs in mapping_b.items():
            for i, idx in enumerate(idxs):
                assert d_b._get_chunk_data(dev, i).device.id == dev
                assert d_b._get_chunk_data(dev, i).ndim == np_a.ndim
                if mode == 'replica':
                    testing.assert_array_equal(
                        d_b._get_chunk_data(dev, i).ravel(), np_a[idx].ravel(),
                        strict=True)

    @pytest.mark.parametrize(
            'shape, mapping_a, mapping_b',
            [(shape_dim2, index_map_dim2, index_map_dim2_2),
             (shape_dim3, index_map_dim3, index_map_dim3_2)])
    @pytest.mark.parametrize('mode', ['replica', 'sum', 'max'])
    def test_incompatible_chunk_shapes_resharded(
            self, shape, mapping_a, mapping_b, mode):
        np_a = numpy.arange(size).reshape(shape)
        np_b = numpy.arange(size).reshape(shape) * 2
        np_r = numpy.cos(np_a + np_b)
        d_a = _array.distributed_array(np_a, mapping_a, mode, comms)
        d_b = _array.distributed_array(np_b, mapping_b, mode, comms)
        d_c = d_a + d_b.reshard(mapping_a)
        d_r = cupy.cos(d_c.reshard(mapping_b))
        testing.assert_array_almost_equal(d_r.asnumpy(), np_r)

    @pytest.mark.parametrize(
            'shape, index_map',
            [(shape_dim2, index_map_dim2),
             (shape_dim3, index_map_dim3),
             (shape_dim3, index_map_only_1)])
    @pytest.mark.parametrize('mode', ['replica', 'sum', 'max'])
    @pytest.mark.parametrize('dtype', ['int64', 'float64'])
    def test_max_reduction(self, shape, index_map, mode, dtype):
        np_a = numpy.arange(size, dtype=dtype).reshape(shape)
        d_a = _array.distributed_array(np_a, index_map, mode, comms)
        for axis in range(np_a.ndim):
            np_b = np_a.max(axis=axis)
            d_b = d_a.max(axis=axis)
            testing.assert_array_equal(d_b.asnumpy(), np_b, strict=True)
            testing.assert_array_equal(d_a.asnumpy(), np_a, strict=True)

    @pytest.mark.parametrize(
            'shape, index_map',
            [(shape_dim2, index_map_dim2),
             (shape_dim3, index_map_dim3),
             (shape_dim3, index_map_only_1)])
    @pytest.mark.parametrize('mode', ['replica', 'sum', 'max'])
    @pytest.mark.parametrize('dtype', ['int64', 'float64'])
    def test_min_reduction(self, shape, index_map, mode, dtype):
        np_a = numpy.arange(size, dtype=dtype).reshape(shape)
        d_a = _array.distributed_array(np_a, index_map, mode, comms)
        for axis in range(np_a.ndim):
            np_b = np_a.min(axis=axis)
            d_b = d_a.min(axis=axis)
            testing.assert_array_equal(d_b.asnumpy(), np_b, strict=True)
            testing.assert_array_equal(d_a.asnumpy(), np_a, strict=True)

    @pytest.mark.parametrize('shape, index_map',
                             [(shape_dim3, index_map_dim3),
                              (shape_dim3, index_map_only_1)])
    @pytest.mark.parametrize('mode', ['replica', 'sum', 'prod'])
    def test_sum_reduction(self, shape, index_map, mode):
        np_a = numpy.arange(size).reshape(shape)
        d_a = _array.distributed_array(np_a, index_map, mode, comms)
        for axis in range(np_a.ndim):
            np_b = np_a.sum(axis=axis)
            d_b = d_a.sum(axis=axis)
            assert d_b._mode is _array._MODES['sum']
            testing.assert_array_equal(d_b.asnumpy(), np_b, strict=True)
            testing.assert_array_equal(d_a.asnumpy(), np_a, strict=True)

    @pytest.mark.parametrize('shape, index_map',
                             [(shape_dim3, index_map_dim3),
                              (shape_dim3, index_map_only_1)])
    @pytest.mark.parametrize('mode', ['replica', 'sum', 'max'])
    def test_prod_reduction(self, shape, index_map, mode):
        np_a = numpy.random.default_rng().random(shape)
        d_a = _array.distributed_array(np_a, index_map, mode, comms)
        for axis in range(np_a.ndim):
            np_b = np_a.prod(axis=axis)
            d_b = d_a.prod(axis=axis)
            testing.assert_array_almost_equal(d_b.asnumpy(), np_b)
            testing.assert_array_almost_equal(d_a.asnumpy(), np_a)

    @pytest.mark.parametrize('shape, index_map', [(shape_dim3, index_map_dim3)])
    def test_unsupported_reduction(self, shape, index_map):
        np_a = numpy.arange(size).reshape(shape)
        d_a = _array.distributed_array(np_a, index_map, 'replica', comms)
        with pytest.raises(RuntimeError, match=r'Unsupported .* cupy_argmax'):
            d_a.argmax(axis=0)

    @pytest.mark.parametrize(
            'shape, mapping_a, mapping_b',
            [(shape_dim2, index_map_dim2, index_map_dim2_2),
             (shape_dim3, index_map_dim3, index_map_dim3_2),
             (shape_dim3, index_map_only_1, index_map_dim3)])
    def test_reshard_max(self, shape, mapping_a, mapping_b):
        np_a = numpy.arange(size).reshape(shape)
        np_b = np_a.max(axis=0)
        d_a = _array.distributed_array(np_a, mapping_a, comms=comms)
        d_b = d_a.reshard(mapping_b).max(axis=0)
        testing.assert_array_equal(np_b, d_b.asnumpy(), strict=True)
        testing.assert_array_equal(np_a, d_a.asnumpy(), strict=True)

    @pytest.mark.parametrize(
            'shape, mapping_a, mapping_b',
            [(shape_dim2, index_map_dim2, index_map_dim2_2),
             (shape_dim3, index_map_dim3, index_map_dim3_2),
             (shape_dim3, index_map_only_1, index_map_dim3)])
    def test_mul_max_mul(self, shape, mapping_a, mapping_b):
        rng = numpy.random.default_rng()
        np_a = rng.integers(0, 1 << 10, shape)
        np_b = rng.integers(0, 1 << 10, shape)
        np_c = rng.integers(0, 1 << 10, shape[1:])
        np_c2 = (np_a * np_b).max(axis=0)
        np_d = (np_a * np_b).max(axis=0) * np_c
        d_a = _array.distributed_array(np_a, mapping_a, comms=comms)
        d_b = _array.distributed_array(np_b, mapping_b, comms=comms)
        mapping_c = {dev: [idx[1:] for idx in idxs]
                     for dev, idxs in d_a.index_map.items()}
        d_c = _array.distributed_array(np_c, mapping_c, comms=comms)
        d_c2 = (d_a.reshard(mapping_b) * d_b).max(axis=0)
        d_d = d_c2.reshard(mapping_c) * d_c
        testing.assert_array_equal(np_d, d_d.asnumpy(), strict=True)
        testing.assert_array_equal(np_c2, d_c2.asnumpy(), strict=True)

    def test_random_reshard_change_mode(self):
        n_iter = 5
        n_ops = 4

        length = 2 ** 13
        size = length * length
        shape = (length, length)
        k = length // 10
        mapping_a = {
            0: slice(length // 15 * 5),
            1: slice(length // 15 * 5, length // 15 * 10),
            2: slice(length // 15 * 10, length // 15 * 13),
            3: slice(length // 15 * 13, None)}
        mapping_b = {
            0: slice(length // 15 * 5 + k),
            1: slice(length // 15 * 5 + k, length // 15 * 10 + k),
            2: slice(length // 15 * 10 + k, length // 15 * 13 + k),
            3: slice(length // 15 * 13 + k, None)}
        mapping_c = {0: slice(None)}

        mapping_a = {dev: _array._convert_chunk_idx_to_slices(shape, idx)
                     for dev, idx in mapping_a.items()}
        mapping_b = {dev: _array._convert_chunk_idx_to_slices(shape, idx)
                     for dev, idx in mapping_b.items()}
        mapping_c = {dev: _array._convert_chunk_idx_to_slices(shape, idx)
                     for dev, idx in mapping_c.items()}
        mappings = [mapping_a, mapping_b, mapping_c]

        ops = ['reshard', 'change_mode']
        modes = list(_array._MODES)

        rng = numpy.random.default_rng()
        for _ in range(n_iter):
            np_a = rng.integers(0, size, shape)
            d_a = _array.distributed_array(np_a, mappings[0], comms=comms)
            history = []
            maps = list(mappings)

            for _ in range(n_ops):
                history.append(d_a)
                op = rng.choice(ops)
                if op == 'reshard':
                    index_map = rng.choice(maps)
                    d_a = d_a.reshard(index_map)
                else:
                    mode = rng.choice(modes)
                    d_a = d_a.change_mode(mode)

            testing.assert_array_equal(np_a, d_a.asnumpy(), strict=True)
            d_b = history[rng.choice(len(history))]
            testing.assert_array_equal(np_a, d_b.asnumpy(), strict=True)

    def test_random_binary_operations(self):
        n_iter = 5
        n_ops = 4

        length = 10000
        size = length * length
        shape = (length, length)
        k = length // 10
        mapping_a = {
            0: slice(length // 15 * 5),
            1: slice(length // 15 * 5, length // 15 * 10),
            2: slice(length // 15 * 10, length // 15 * 13),
            3: slice(length // 15 * 13, None)}
        mapping_b = {
            0: slice(length // 15 * 5 + k),
            1: slice(length // 15 * 5 + k, length // 15 * 10 + k),
            2: slice(length // 15 * 10 + k, length // 15 * 13 + k),
            3: slice(length // 15 * 13 + k, None)}
        mapping_c = {0: slice(None)}

        mapping_a = {dev: _array._convert_chunk_idx_to_slices(shape, idx)
                     for dev, idx in mapping_a.items()}
        mapping_b = {dev: _array._convert_chunk_idx_to_slices(shape, idx)
                     for dev, idx in mapping_b.items()}
        mapping_c = {dev: _array._convert_chunk_idx_to_slices(shape, idx)
                     for dev, idx in mapping_c.items()}
        mappings = [mapping_a, mapping_b, mapping_c]

        ops = ['reshard', 'change_mode', 'element-wise', 'reduce']
        modes = list(_array._MODES)
        elementwise = ['add', 'multiply', 'maximum', 'minimum']
        reduce = ['sum', 'prod', 'max', 'min']

        rng = numpy.random.default_rng()
        for _ in range(n_iter):
            import random
            np_a = rng.integers(0, size, shape)
            np_b = rng.integers(0, size, shape)
            d_a = _array.distributed_array(np_a, mappings[0], comms=comms)
            d_b = _array.distributed_array(np_b, mappings[0], comms=comms)
            arrs = [(np_a, d_a), (np_b, d_b)]
            arrs_history = []
            maps = list(mappings)

            for _ in range(n_ops):
                arrs_history.append(list(arrs))
                op = rng.choice(ops)
                print(arrs[0][0].shape, arrs[0][1].shape)
                assert arrs[0][0].shape == arrs[0][1].shape
                print(op)
                # Cannot do rng.choice(arrs) here because numpy tries to convert
                # arrs to a ndarray
                arr_idx = rng.choice(len(arrs))
                np_arr, d_arr = arrs[arr_idx]
                if op == 'reshard':
                    index_map = rng.choice(maps)
                    arrs[arr_idx] = np_arr, d_arr.reshard(index_map)
                elif op == 'change_mode':
                    mode = rng.choice(modes)
                    arrs[arr_idx] = np_arr, d_arr.change_mode(mode)
                elif op == 'element-wise':
                    kernel = rng.choice(elementwise)
                    choice = rng.choice(len(arrs))
                    np_arr2, d_arr2 = arrs[choice]
                    np_arr_new = getattr(numpy, kernel)(np_arr, np_arr2)
                    if d_arr.index_map != d_arr2.index_map:
                        d_arr = d_arr.reshard(d_arr2.index_map)
                    d_arr_new = getattr(cupy, kernel)(d_arr, d_arr2)
                    arrs[arr_idx] = np_arr_new, d_arr_new
                else:
                    if np_arr.ndim == 0:
                        continue
                    kernel = rng.choice(reduce)
                    axis = rng.choice(np_arr.ndim)
                    for i in range(len(arrs)):
                        np_arr, d_arr = arrs[i]
                        np_arr_new = getattr(numpy, kernel)(np_arr, axis)
                        d_arr_new = getattr(cupy, kernel)(d_arr, axis)
                        arrs[i] = np_arr_new, d_arr_new
                    for i in range(len(maps)):
                        maps[i] = {dev: idx[:axis] + idx[axis+1:]
                                       for dev, idx in maps[i].items()}

            for i, arrs in enumerate(arrs_history):
                (np_a, d_a), (np_b, d_b) = arrs
                testing.assert_array_equal(np_a, d_a.asnumpy(), strict=True)
                testing.assert_array_equal(np_b, d_b.asnumpy(), strict=True)

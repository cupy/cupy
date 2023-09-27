import warnings

import numpy
import pytest

import cupy
from cupy import testing
import cupyx.distributed.array as darray
from cupyx.distributed.array import _chunk
from cupyx.distributed.array import _index_arith
from cupyx.distributed.array import _modes


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


size = 256


shape_dim2 = (16, 16)
index_map_dim2 = {
    0: [(slice(10), slice(10)),
        (slice(10), slice(6, None))],
    1: [(slice(6, None), slice(None, None, 2)),
        (slice(6, None), slice(1, None, 4)),
        (slice(6, None), slice(3, None, 4))],
    3: [(slice(6, None, 3), slice(None, None, 3))],
}
index_map_dim2_2 = {
    0: [(slice(None, None, 2), slice(None, None, 2))],
    2: [(slice(None, None, 2), slice(1, 6, 2)),
        (slice(None, None, 2), slice(3, 10, 2))],
    3: [(slice(None, None, 2), slice(5, None, 2)),
        slice(1, None, 2)],
}

shape_dim3 = (8, 8, 4)
index_map_dim3 = {
    0: [slice(4)],
    1: [(slice(4, None), slice(None, 7))],
    2: [(slice(4, None), 7)],
    3: [(slice(4, None), slice(None), 1)],
}
index_map_dim3_2 = {
    1: [(slice(1, None, 2), 0),
        (slice(1, None, 2), slice(1, None), 3)],
    2: [(slice(1, None, 2), slice(1, None), slice(None, 3))],
    3: [slice(None, None, 2)],
}

index_map_only_1 = {
    1: [(slice(None),)],
}


for dev in range(4):
    with cupy.cuda.Device(dev):
        stream = cupy.cuda.Stream()
        stream.__enter__()


@testing.multi_gpu(4)
class TestDistributedArray:
    @pytest.mark.parametrize(
        'shape, index_map',
        [(shape_dim2, index_map_dim2), (shape_dim3, index_map_dim3)])
    @pytest.mark.parametrize('mode', ['replica', 'sum', 'max'])
    def test_array_creation_from_numpy(self, mem_pool, shape, index_map, mode):
        array = numpy.arange(size, dtype='q').reshape(shape)
        # assert mem_pool.used_bytes() == 0
        da = darray.distributed_array(array, index_map, mode)
        assert da.device.id == -1
        # Ensure no memory allocation other than the chunks
        assert da.data.ptr == 0
        assert da.shape == shape
        # assert mem_pool.used_bytes() == array.nbytes
        for dev in index_map.keys():
            for chunk, idx in zip(da._chunks_map[dev], index_map[dev]):
                assert chunk.array.device.id == dev
                assert chunk.array.ndim == array.ndim
                if mode == 'replica':
                    idx = _index_arith._normalize_index(shape, idx)
                    testing.assert_array_equal(
                        chunk.array, array[idx], strict=True)

    @pytest.mark.parametrize(
        'shape, index_map',
        [(shape_dim2, index_map_dim2), (shape_dim3, index_map_dim3)])
    @pytest.mark.parametrize('mode', ['replica', 'sum', 'max'])
    def test_array_creation_from_cupy(self, mem_pool, shape, index_map, mode):
        array = cupy.arange(size, dtype='q').reshape(shape)
        # assert mem_pool.used_bytes() == array.nbytes
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', cupy._util.PerformanceWarning)
            da = darray.distributed_array(
                array, index_map, mode)
        assert da.device.id == -1
        # Ensure no memory allocation other than chunks & original array
        assert da.data.ptr == 0
        assert da.shape == shape
        # assert mem_pool.used_bytes() == 2 * array.nbytes
        for dev in index_map.keys():
            for chunk, idx in zip(da._chunks_map[dev], index_map[dev]):
                assert chunk.array.device.id == dev
                assert chunk.array.ndim == array.ndim
                if mode == 'replica':
                    idx = _index_arith._normalize_index(shape, idx)
                    testing.assert_array_equal(
                        chunk.array, array[idx], strict=True)

    @pytest.mark.parametrize(
        'shape, index_map',
        [(shape_dim2, index_map_dim2), (shape_dim3, index_map_dim3)])
    @pytest.mark.parametrize('mode', ['replica', 'sum', 'max'])
    def test_array_creation(self, mem_pool, shape, index_map, mode):
        array = numpy.arange(size, dtype='q').reshape(shape)
        # assert mem_pool.used_bytes() == 0
        da = darray.distributed_array(
            array.tolist(), index_map, mode)
        assert da.device.id == -1
        # Ensure no memory allocation other than the chunks
        assert da.data.ptr == 0
        assert da.shape == shape
        # assert mem_pool.used_bytes() == array.nbytes
        for dev in index_map.keys():
            for chunk, idx in zip(da._chunks_map[dev], index_map[dev]):
                assert chunk.array.device.id == dev
                assert chunk.array.ndim == array.ndim
                if mode == 'replica':
                    idx = _index_arith._normalize_index(shape, idx)
                    testing.assert_array_equal(
                        chunk.array, array[idx], strict=True)

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
                idx = _index_arith._normalize_index(shape, idx)
                np_a[idx] += 1 << dev
                with cupy.cuda.Device(dev):
                    chunk = _chunk._Chunk(
                        cupy.full_like(np_a[idx], 1 << dev),
                        cupy.cuda.Event(), idx)
                    chunks_map[dev].append(chunk)

        d_a = darray.DistributedArray(
            shape, np_a.dtype, chunks_map, _modes._MODES['sum'])
        d_b = d_a._to_replica_mode()
        assert d_b._mode is _modes._REPLICA_MODE
        testing.assert_array_equal(d_b.asnumpy(), np_a, strict=True)
        testing.assert_array_equal(d_a.asnumpy(), np_a, strict=True)
        for dev in index_map.keys():
            for chunk, idx in zip(d_b._chunks_map[dev], index_map[dev]):
                assert chunk.array.device.id == dev
                idx = _index_arith._normalize_index(shape, idx)
                testing.assert_array_equal(
                    chunk.array, np_a[idx], strict=True)

    @pytest.mark.parametrize(
        'shape, index_map',
        [(shape_dim2, index_map_dim2),
         (shape_dim3, index_map_dim3),
         (shape_dim3, index_map_only_1)])
    @pytest.mark.parametrize('mode', ['max', 'sum'])
    def test_change_to_op_mode(self, shape, index_map, mode):
        np_a = numpy.arange(size).reshape(shape)
        d_a = darray.distributed_array(np_a, index_map, mode)
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
        # We do not choose sin because sin(0) == 0
        np_r = numpy.cos(np_a * np_b)
        d_a = darray.distributed_array(np_a, index_map, mode_a)
        d_b = darray.distributed_array(np_b, index_map, mode_b)
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
        d_a = darray.distributed_array(np_a, index_map, mode_a)
        d_b = darray.distributed_array(np_b, index_map, mode_b)
        d_r = custom_kernel(d_a, d_b)
        testing.assert_array_almost_equal(d_r.asnumpy(), np_r)

    @pytest.mark.parametrize(
        'shape, mapping',
        [(shape_dim2, index_map_dim2), (shape_dim3, index_map_dim3)])
    @pytest.mark.parametrize('mode', ['replica', 'sum', 'max'])
    def test_incompatible_chunk_shapes(self, shape, mapping, mode):
        index_map_a = {}
        for dev, idxs in mapping.items():
            index_map_a.setdefault(dev % 2, []).extend(idxs)
        index_map_b = {0: index_map_a[1],
                       1: index_map_a[0]}

        np_a = numpy.arange(size).reshape(shape)
        np_b = numpy.arange(size).reshape(shape) * 2
        np_r = numpy.cos(np_a * np_b)
        d_a = darray.distributed_array(np_a, index_map_a, mode)
        d_b = darray.distributed_array(np_b, index_map_a, mode)
        with pytest.warns(cupy._util.PerformanceWarning, match=r'Peer access'):
            d_r = cupy.cos(d_a * d_b.reshard(index_map_b))
        testing.assert_array_almost_equal(d_r.asnumpy(), np_r)

    @pytest.mark.parametrize(
        'shape, index_map',
        [(shape_dim2, index_map_dim2), (shape_dim3, index_map_dim3)])
    @pytest.mark.parametrize('mode', ['replica', 'sum'])
    def test_elementwise_kernel_incompatible_chunk_shapes(
            self, shape, index_map, mode):
        index_map_a = {}
        for dev, idxs in index_map.items():
            index_map_a.setdefault(dev % 2, []).extend(idxs)
        index_map_b = {0: index_map_a[1],
                       1: index_map_a[0]}

        custom_kernel = cupy.ElementwiseKernel(
            'float32 x, float32 y',
            'float32 z',
            'z = (x - y) * (x - y)',
            'custom')
        np_a = numpy.arange(size).reshape(shape).astype(numpy.float32)
        np_b = (numpy.arange(size).reshape(shape) * 2.0).astype(numpy.float32)
        np_r = (np_a - np_b) * (np_a - np_b)
        d_a = darray.distributed_array(np_a, index_map_a, mode)
        d_b = darray.distributed_array(np_b, index_map_a, mode)
        with pytest.warns(cupy._util.PerformanceWarning, match=r'Peer access'):
            d_r = custom_kernel(d_a, d_b.reshard(index_map_b))
        testing.assert_array_almost_equal(d_r.asnumpy(), np_r)

    @pytest.mark.parametrize(
        'shape, index_map',
        [(shape_dim2, index_map_dim2), (shape_dim3, index_map_dim3)])
    @pytest.mark.parametrize('mode', ['replica', 'sum', 'max'])
    def test_incompatible_operand(self, shape, index_map, mode):
        np_a = numpy.arange(size).reshape(shape)
        cp_b = cupy.arange(size).reshape(shape)
        d_a = darray.distributed_array(np_a, index_map, mode)
        with pytest.raises(RuntimeError, match=r'Mixing.* dist.* non-dist'):
            cupy.cos(d_a * cp_b)

    @pytest.mark.parametrize(
        'shape, index_map_a, index_map_b',
        [(shape_dim2, index_map_dim2, index_map_dim2_2),
         (shape_dim3, index_map_dim3, index_map_dim3_2),
         (shape_dim3, index_map_only_1, index_map_dim3)])
    @pytest.mark.parametrize('mode', ['replica', 'sum', 'max'])
    def test_reshard(self, mem_pool, shape, index_map_a, index_map_b, mode):
        np_a = numpy.arange(size, dtype='q').reshape(shape)
        # assert mem_pool.used_bytes() == 0
        # Initialize without comms
        d_a = darray.distributed_array(np_a, index_map_a, mode)
        # assert mem_pool.used_bytes() == np_a.nbytes
        d_b = d_a.reshard(index_map_b)
        testing.assert_array_equal(d_b.asnumpy(), np_a, strict=True)
        testing.assert_array_equal(d_a.asnumpy(), np_a, strict=True)
        assert d_b.mode == mode
        for dev in index_map_b.keys():
            for chunk, idx in zip(d_b._chunks_map[dev], index_map_b[dev]):
                assert chunk.array.device.id == dev
                assert chunk.array.ndim == np_a.ndim
                if mode == 'replica':
                    idx = _index_arith._normalize_index(shape, idx)
                    testing.assert_array_equal(
                        chunk.array, np_a[idx], strict=True)

    @pytest.mark.parametrize(
        'shape, index_map_a, index_map_b',
        [(shape_dim2, index_map_dim2, index_map_dim2_2),
         (shape_dim3, index_map_dim3, index_map_dim3_2)])
    @pytest.mark.parametrize('mode', ['replica', 'sum', 'max'])
    def test_incompatible_chunk_shapes_resharded(
            self, shape, index_map_a, index_map_b, mode):
        np_a = numpy.arange(size).reshape(shape)
        np_b = numpy.arange(size).reshape(shape) * 2
        np_r = numpy.cos(np_a + np_b)
        d_a = darray.distributed_array(np_a, index_map_a, mode)
        d_b = darray.distributed_array(np_b, index_map_b, mode)
        d_c = d_a + d_b.reshard(index_map_a)
        d_r = cupy.cos(d_c.reshard(index_map_b))
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
        d_a = darray.distributed_array(np_a, index_map, mode)
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
        d_a = darray.distributed_array(np_a, index_map, mode)
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
        d_a = darray.distributed_array(np_a, index_map, mode)
        for axis in range(np_a.ndim):
            np_b = np_a.sum(axis=axis)
            d_b = d_a.sum(axis=axis)
            assert d_b._mode is _modes._MODES['sum']
            testing.assert_array_equal(d_b.asnumpy(), np_b, strict=True)
            testing.assert_array_equal(d_a.asnumpy(), np_a, strict=True)

    @pytest.mark.parametrize('shape, index_map',
                             [(shape_dim3, index_map_dim3),
                              (shape_dim3, index_map_only_1)])
    @pytest.mark.parametrize('mode', ['replica', 'sum', 'max'])
    def test_prod_reduction(self, shape, index_map, mode):
        np_a = numpy.random.default_rng().random(shape)
        d_a = darray.distributed_array(np_a, index_map, mode)
        for axis in range(np_a.ndim):
            np_b = np_a.prod(axis=axis)
            d_b = d_a.prod(axis=axis)
            testing.assert_array_almost_equal(d_b.asnumpy(), np_b)
            testing.assert_array_almost_equal(d_a.asnumpy(), np_a)

    @pytest.mark.parametrize('shape, index_map',
                             [(shape_dim3, index_map_dim3)])
    def test_unsupported_reduction(self, shape, index_map):
        np_a = numpy.arange(size).reshape(shape)
        d_a = darray.distributed_array(np_a, index_map, 'replica')
        with pytest.raises(RuntimeError, match=r'Unsupported .* cupy_argmax'):
            d_a.argmax(axis=0)

    @pytest.mark.parametrize(
        'shape, index_map_a, index_map_b',
        [(shape_dim2, index_map_dim2, index_map_dim2_2),
         (shape_dim3, index_map_dim3, index_map_dim3_2),
         (shape_dim3, index_map_only_1, index_map_dim3)])
    def test_reshard_max(self, shape, index_map_a, index_map_b):
        np_a = numpy.arange(size).reshape(shape)
        np_b = np_a.max(axis=0)
        d_a = darray.distributed_array(np_a, index_map_a)
        d_b = d_a.reshard(index_map_b).max(axis=0)
        testing.assert_array_equal(np_b, d_b.asnumpy(), strict=True)
        testing.assert_array_equal(np_a, d_a.asnumpy(), strict=True)

    @pytest.mark.parametrize(
        'shape, index_map_a, index_map_b',
        [(shape_dim2, index_map_dim2, index_map_dim2_2),
         (shape_dim3, index_map_dim3, index_map_dim3_2),
         (shape_dim3, index_map_only_1, index_map_dim3)])
    def test_mul_max_mul(self, shape, index_map_a, index_map_b):
        rng = numpy.random.default_rng()
        np_a = rng.integers(0, 1 << 10, shape)
        np_b = rng.integers(0, 1 << 10, shape)
        np_c = rng.integers(0, 1 << 10, shape[1:])
        np_c2 = (np_a * np_b).max(axis=0)
        np_d = (np_a * np_b).max(axis=0) * np_c
        d_a = darray.distributed_array(np_a, index_map_a)
        d_b = darray.distributed_array(np_b, index_map_b)
        mapping_c = {dev: [idx[1:] for idx in idxs]
                     for dev, idxs in d_a.index_map.items()}
        d_c = darray.distributed_array(np_c, mapping_c)
        d_c2 = (d_a.reshard(index_map_b) * d_b).max(axis=0)
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
        index_map_a = {
            0: slice(length // 15 * 5),
            1: slice(length // 15 * 5, length // 15 * 10),
            2: slice(length // 15 * 10, length // 15 * 13),
            3: slice(length // 15 * 13, None)}
        index_map_b = {
            0: slice(length // 15 * 5 + k),
            1: slice(length // 15 * 5 + k, length // 15 * 10 + k),
            2: slice(length // 15 * 10 + k, length // 15 * 13 + k),
            3: slice(length // 15 * 13 + k, None)}
        mapping_c = {0: slice(None)}

        index_map_a = {dev: _index_arith._normalize_index(shape, idx)
                       for dev, idx in index_map_a.items()}
        index_map_b = {dev: _index_arith._normalize_index(shape, idx)
                       for dev, idx in index_map_b.items()}
        mapping_c = {dev: _index_arith._normalize_index(shape, idx)
                     for dev, idx in mapping_c.items()}
        mappings = [index_map_a, index_map_b, mapping_c]

        ops = ['reshard', 'change_mode']
        modes = list(_modes._MODES)

        rng = numpy.random.default_rng()
        for _ in range(n_iter):
            np_a = rng.integers(0, size, shape)
            d_a = darray.distributed_array(np_a, mappings[0])
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
        index_map_a = {
            0: slice(length // 15 * 5),
            1: slice(length // 15 * 5, length // 15 * 10),
            2: slice(length // 15 * 10, length // 15 * 13),
            3: slice(length // 15 * 13, None)}
        index_map_b = {
            0: slice(length // 15 * 5 + k),
            1: slice(length // 15 * 5 + k, length // 15 * 10 + k),
            2: slice(length // 15 * 10 + k, length // 15 * 13 + k),
            3: slice(length // 15 * 13 + k, None)}
        mapping_c = {0: slice(None)}

        index_map_a = {dev: _index_arith._normalize_index(shape, idx)
                       for dev, idx in index_map_a.items()}
        index_map_b = {dev: _index_arith._normalize_index(shape, idx)
                       for dev, idx in index_map_b.items()}
        mapping_c = {dev: _index_arith._normalize_index(shape, idx)
                     for dev, idx in mapping_c.items()}
        mappings = [index_map_a, index_map_b, mapping_c]

        ops = ['reshard', 'change_mode', 'element-wise', 'reduce']
        modes = list(_modes._MODES)
        elementwise = ['add', 'multiply', 'maximum', 'minimum']
        reduce = ['sum', 'prod', 'max', 'min']

        rng = numpy.random.default_rng()
        for _ in range(n_iter):
            np_a = rng.integers(0, size, shape)
            np_b = rng.integers(0, size, shape)
            d_a = darray.distributed_array(np_a, mappings[0])
            d_b = darray.distributed_array(np_b, mappings[0])
            arrs = [(np_a, d_a), (np_b, d_b)]
            arrs_history = []
            maps = list(mappings)

            for _ in range(n_ops):
                arrs_history.append(list(arrs))
                op = rng.choice(ops)
                assert arrs[0][0].shape == arrs[0][1].shape
                # Cannot do rng.choice(arrs) here because numpy tries to
                # convert arrs to a ndarray
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

import pytest

import numpy
import cupy
from cupy import testing
from cupy.cuda import nccl
from cupyx.distributed import _array
import math


def make_comms():
    if not nccl.available:
        return None
    comms_list = nccl.NcclCommunicator.initAll(4)
    return {dev: comm for dev, comm in zip(range(4), comms_list)}


comms = make_comms()


shape_a = (100, 200)
size_a = math.prod(shape_a)
index_map_a = {
    0: [(slice(60), slice(110)),
        (slice(60, None), slice(110, None))],
    1: [(slice(60), slice(110, None)),
        (slice(60, None), slice(110, None))],
    2: [(slice(60), slice(110)),
        (slice(60, None), slice(110))],
    3: [(slice(60), slice(110, None)),
        (slice(60, None), slice(110))],
}
index_map_a_2 = {
    3: [(slice(60), slice(110)),
        (slice(60, None), slice(110, None))],
    2: [(slice(60), slice(110, None)),
        (slice(60, None), slice(110, None))],
    1: [(slice(60), slice(110)),
        (slice(60, None), slice(110))],
    0: [(slice(60), slice(110, None)),
        (slice(60, None), slice(110))],
}


shape_b = (200, 120)
size_b = math.prod(shape_b)
index_map_b = {
    0: [(slice(110), slice(70)),
        (slice(110, None), slice(70))],
    1: [(slice(110, None), slice(70)),
        (slice(110, None), slice(70, None))],
    2: [(slice(110), slice(70)),
        (slice(110), slice(70, None))],
    3: [(slice(110), slice(70, None)),
        (slice(110, None), slice(70, None))],
}
index_map_b_2 = {
    3: [(slice(110), slice(70)),
        (slice(110, None), slice(70))],
    2: [(slice(110, None), slice(70)),
        (slice(110, None), slice(70, None))],
    1: [(slice(110), slice(70)),
        (slice(110), slice(70, None))],
    0: [(slice(110), slice(70, None)),
        (slice(110, None), slice(70, None))],
}


@testing.multi_gpu(4)
class TestDistributedMatMul:
    def test_matmul(self):
        np_a = numpy.arange(size_a).reshape(shape_a)
        np_b = numpy.arange(size_b).reshape(shape_b)
        np_c = np_a @ np_b
        d_a = _array.distributed_array(np_a, index_map_a, comms=comms)
        d_b = _array.distributed_array(np_b, index_map_b, comms=comms)
        d_c = d_a @ d_b
        testing.assert_array_equal(d_c.asnumpy(), np_c)

    def test_matmul_incompatible(self):
        np_a = numpy.arange(size_a).reshape(shape_a)
        np_b = numpy.arange(size_b).reshape(shape_b)
        index_map_b_2 = index_map_b | {0: (slice(1100), slice(600))}
        d_a = _array.distributed_array(np_a, index_map_a, comms=comms)
        d_b = _array.distributed_array(np_b, index_map_b_2, comms=comms)
        with pytest.raises(RuntimeError, match=r'Inconsistent'):
            d_c = d_a @ d_b

    def test_matmul_hi_dim(self):
        shape_dim3_a = (2, 3) + shape_a
        shape_dim3_b = (2, 3) + shape_b

        size_dim3_a = 6 * size_a
        size_dim3_b = 6 * size_b

        np_a = numpy.arange(size_dim3_a).reshape(shape_dim3_a)
        np_b = numpy.arange(size_dim3_b).reshape(shape_dim3_b)

        np_c = np_a @ np_b

        index_map_dim3_a = {dev: [] for dev in index_map_a.keys()}
        for dev, idxs in index_map_a.items():
            index_map_dim3_a[dev] += [(0, slice(None)) + idx for idx in idxs]
        for dev, idxs in index_map_a_2.items():
            index_map_dim3_a[dev] += [(1, slice(None)) + idx for idx in idxs]

        index_map_dim3_b = {dev: [] for dev in index_map_b.keys()}
        for dev, idxs in index_map_b.items():
            index_map_dim3_b[dev] += [(0, slice(None)) + idx for idx in idxs]
        for dev, idxs in index_map_b_2.items():
            index_map_dim3_b[dev] += [(1, slice(None)) + idx for idx in idxs]

        d_a = _array.distributed_array(np_a, index_map_dim3_a, comms=comms)
        d_b = _array.distributed_array(np_b, index_map_dim3_b, comms=comms)

        d_c = d_a @ d_b

        testing.assert_array_equal(d_c.asnumpy(), np_c)
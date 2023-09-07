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


shape_a = (1000, 2000)

size_a = math.prod(shape_a)

mapping_a = {
    0: (slice(None), slice(1100)),
    1: (slice(None), slice(1100)),
    2: (slice(None), slice(1100, None)),
    3: (slice(None), slice(1100, None)),
}


shape_b = (2000, 1200)

size_b = math.prod(shape_b)

mapping_b = {
    0: (slice(1100), slice(700)),
    1: (slice(1100), slice(700, None)),
    2: (slice(1100, None), slice(700)),
    3: (slice(1100, None), slice(700, None)),
}


@testing.multi_gpu(4)
class TestDistributedMatMul:
    def test_matmul(self):
        np_a = numpy.arange(size_a).reshape(shape_a)
        np_b = numpy.arange(size_b).reshape(shape_b)
        np_c = np_a @ np_b
        d_a = _array.distributed_array(np_a, mapping_a, comms=comms)
        d_b = _array.distributed_array(np_b, mapping_b, comms=comms)
        d_c = numpy.matmul(d_a, d_b)
        testing.assert_array_equal(d_c.asnumpy(), np_c)

    def test_matmul_incompatible(self):
        np_a = numpy.arange(size_a).reshape(shape_a)
        np_b = numpy.arange(size_b).reshape(shape_b)
        mapping_b_2 = mapping_b | {0: (slice(1100), slice(600))}
        d_a = _array.distributed_array(np_a, mapping_a, comms=comms)
        d_b = _array.distributed_array(np_b, mapping_b_2, comms=comms)
        with pytest.raises(RuntimeError, match=r'Inconsistent'):
            d_c = numpy.matmul(d_a, d_b)

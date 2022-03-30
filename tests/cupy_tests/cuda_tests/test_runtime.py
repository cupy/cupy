import pickle
import unittest

import cupy
from cupy.cuda import runtime


class TestExceptionPicklable(unittest.TestCase):

    def test(self):
        e1 = runtime.CUDARuntimeError(1)
        e2 = pickle.loads(pickle.dumps(e1))
        assert e1.args == e2.args
        assert str(e1) == str(e2)


class TestMemPool:

    def test_mallocFromPoolAsync(self):
        pool = runtime.deviceGetMemPool(0)
        assert pool > 0
        s = cupy.cuda.Stream()
        ptr = runtime.mallocFromPoolAsync(128, pool, s.ptr)
        assert ptr > 0
        runtime.freeAsync(ptr, s.ptr)

    def test_create_destroy(self):
        props = runtime.MemPoolProps(
            runtime.cudaMemAllocationTypePinned,
            runtime.cudaMemHandleTypeNone,
            runtime.cudaMemLocationTypeDevice,
            0)  # on device 0
        pool = runtime.memPoolCreate(props)
        runtime.memPoolDestroy(pool)

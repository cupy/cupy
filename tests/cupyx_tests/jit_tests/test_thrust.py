import numpy

import cupy
from cupyx import jit

class TestThrust:
    def test_count_smem(self):
        @jit.rawkernel()
        def useCount_smem(x, y):
            tid = jit.threadIdx.x
            smem = jit.shared_memory(numpy.int32, 32)
            smem[tid] = x[tid]
            jit.syncthreads()
            y[tid] = jit.thrust.count(jit.thrust.device, smem, smem + 32, tid)

        size = cupy.uint32(32)
        x = cupy.arange(size, dtype=cupy.int32)
        y = cupy.zeros(size, dtype=cupy.int32)
        useCount_smem[1, 32](x, y)
        assert (y == cupy.ones(size, dtype=cupy.int32)).all()

    def test_count_CArray(self):
        @jit.rawkernel()
        def useCount_CArray(array3dim, array2dim):
            tid = jit.threadIdx.x
            myh = tid // 32
            myw = tid % 32
            mychannel = array3dim[myh, myw]
            array2dim[myh, myw] = jit.thrust.count(jit.thrust.device, mychannel.begin(), mychannel.end(), tid*c)

        h = 32
        w = 32
        c = 3
        array3dim = cupy.arange(h*w*c, dtype=cupy.int32)
        array3dim = cupy.resize(array3dim, (h, w, c))
        array2dim = cupy.zeros(h*w, dtype=cupy.int32)
        array2dim = cupy.resize(array2dim, (h, w))
        useCount_CArray[1, 1024](array3dim, array2dim)
        ans = cupy.ones(h*w, dtype=cupy.int32)
        ans = cupy.resize(ans, (h, w))
        assert (array2dim == ans).all()

    def test_count_transposed_CArray(self):
        @jit.rawkernel()
        def useCount_CArray_Uncontiguous(array3dim, array2dim):
            tid = jit.threadIdx.x
            myh = tid // 32
            myw = tid % 32
            mychannel = array3dim[myh, myw]
            array2dim[myh, myw] = jit.thrust.count(jit.thrust.device, mychannel.begin(), mychannel.end(), tid)

        h = 3
        w = 32
        c = 32
        array3dim = cupy.arange(h*w*c, dtype=cupy.int32)
        array3dim = cupy.resize(array3dim, (h, w, c))
        array3dim = array3dim.transpose(1, 2, 0)
        array2dim = cupy.zeros(w*c, dtype=cupy.int32)
        array2dim = cupy.resize(array2dim, (w, c))
        useCount_CArray_Uncontiguous[1, 1024](array3dim, array2dim)
        ans = cupy.ones(w*c, dtype=cupy.int32)
        ans = cupy.resize(ans, (w, c))
        assert (array2dim == ans).all()

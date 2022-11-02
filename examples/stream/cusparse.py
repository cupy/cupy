# nvprof --print-gpu-trace python examples/stream/cusparse.py
import cupy
import cupyx


def _make(xp, sp, dtype):
    data = xp.array([0, 1, 3, 2], dtype)
    indices = xp.array([0, 0, 2, 1], 'i')
    indptr = xp.array([0, 1, 2, 3, 4], 'i')
    # 0, 1, 0, 0
    # 0, 0, 0, 2
    # 0, 0, 3, 0
    return sp.csc_matrix((data, indices, indptr), shape=(3, 4))


x = _make(cupy, cupyx.scipy.sparse, float)
expected = cupyx.cusparse.cscsort(x)
cupy.cuda.Device().synchronize()

stream = cupy.cuda.stream.Stream()
with stream:
    y = cupyx.cusparse.cscsort(x)
stream.synchronize()
cupy.testing.assert_array_equal(y, expected)

stream = cupy.cuda.stream.Stream()
stream.use()
y = cupyx.cusparse.cscsort(x)
stream.synchronize()
cupy.testing.assert_array_equal(y, expected)

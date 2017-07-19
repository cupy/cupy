# nvprof --print-gpu-trace python examples/stream/cusparse.py
import cupy


def _make(xp, sp, dtype):
    data = xp.array([0, 1, 3, 2], dtype)
    indices = xp.array([0, 0, 2, 1], 'i')
    indptr = xp.array([0, 1, 2, 3, 4], 'i')
    # 0, 1, 0, 0
    # 0, 0, 0, 2
    # 0, 0, 3, 0
    return sp.csc_matrix((data, indices, indptr), shape=(3, 4))


x = _make(cupy, cupy.sparse, float)

with cupy.cuda.stream.Stream():
    y = cupy.cusparse.cscsort(x)

stream = cupy.cuda.stream.Stream()
stream.use()
y = cupy.cusparse.cscsort(x)

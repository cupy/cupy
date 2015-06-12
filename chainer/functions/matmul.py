import numpy
import six

from chainer import cuda
from chainer import function


def _mat_ptrs(a):
    """Create an array of pointers to matrices

    Args:
        a: batch of matrices in GPU
    Returns:
        GPU array of pointers to matrices
    """
    return cuda.to_gpu(numpy.arange(
        a.ptr, a.ptr + a.shape[0] * a.strides[0], a.strides[0],
        dtype=cuda.cublas.ctypes.c_void_p))


def _as_mat(x):
    return x.reshape((len(x), 1)) if len(x.shape) == 1 else x


def _as_batch_mat(x):
    return x.reshape((x.shape[0], x.shape[1], 1)) if len(x.shape) == 2 else x


def _as_trans_op(trans):
    return 't' if trans else 'n'


def _matmul_cpu(a, b, transa=False, transb=False, transout=False):
    if transout:
        # (A B)^T = B^T A^T
        a, b, transa, transb = b, a, not transb, not transa
    a = _as_mat(a)
    b = _as_mat(b)
    if transa:
        a = a.T
    if transb:
        b = b.T
    return numpy.dot(a, b)


def _matmul_gpu(a, b, transa=False, transb=False, transout=False, out=None):
    if transout:
        # (A B)^T = B^T A^T
        a, b, transa, transb = b, a, not transb, not transa
    a = _as_mat(a)
    b = _as_mat(b)
    with cuda.using_cumisc():
        return cuda.culinalg.dot(a, b,
                                 transa=_as_trans_op(transa),
                                 transb=_as_trans_op(transb),
                                 out=out)


def _batch_matmul_gpu(a, b, out, transa=False, transb=False, transout=False):
    if transout:
        # (A B)^T = B^T A^T
        a, b, transa, transb = b, a, not transb, not transa
    a = _as_batch_mat(a)
    b = _as_batch_mat(b)
    alpha = numpy.float32(1.0)
    beta = numpy.float32(0.0)
    l, m, k = a.shape
    if transa:
        m, k = k, m
    n = b.shape[1] if transb else b.shape[2]
    with cuda.using_cumisc():
        return cuda.cublas.cublasSgemmBatched(
            cuda.get_cublas_handle(),
            _as_trans_op(transb),
            _as_trans_op(transa),
            n, m, k, alpha,
            _mat_ptrs(b).gpudata, k if transb else n,
            _mat_ptrs(a).gpudata, m if transa else k,
            beta, _mat_ptrs(out).gpudata, n, l)


class MatMul(function.Function):
    def __init__(self, transa=False, transb=False):
        self.transa = transa
        self.transb = transb

    def forward_cpu(self, x):
        assert len(x[0].shape) == len(x[1].shape)
        return _matmul_cpu(x[0], x[1], transa=self.transa, transb=self.transb),

    def forward_gpu(self, x):
        assert len(x[0].shape) == len(x[1].shape)
        return _matmul_gpu(x[0], x[1], transa=self.transa, transb=self.transb),

    def backward_cpu(self, x, gy):
        gx0 = _matmul_cpu(
            gy[0], x[1], transb=not self.transb, transout=self.transa
            ).reshape(x[0].shape)
        gx1 = _matmul_cpu(
            x[0], gy[0], transa=not self.transa, transout=self.transb
            ).reshape(x[1].shape)
        return gx0, gx1

    def backward_gpu(self, x, gy):
        with cuda.using_cumisc():
            gx0 = _matmul_gpu(
                gy[0], x[1], transb=not self.transb, transout=self.transa
                ).reshape(x[0].shape)
            gx1 = _matmul_gpu(
                x[0], gy[0], transa=not self.transa, transout=self.transb
                ).reshape(x[1].shape)
            return gx0, gx1


def matmul(a, b, transa=False, transb=False):
    """Compute matrix multiplication of two arrays.

    Args:
        a, b: Variables of 2-D or 1-D arrays.
            A 2-D array with shape (N, M) is considered as a NxM matrix.
            A 1-D array with shape (N,) is considered as a Nx1 matrix.
        transa (bool): If true, transpose a.
        transb (bool): If true, transpose b.

    Returns:
        ~chainer.Variable: Matrix multiplication as a 2-D array
    """
    return MatMul(transa=transa, transb=transb)(a, b)


class BatchMatMul(function.Function):
    def __init__(self, transa=False, transb=False):
        self.transa = transa
        self.transb = transb

    def _output_shape(self, a, b):
        batch_size = a.shape[0]
        a_mat_shape = _as_mat(a[0]).shape
        b_mat_shape = _as_mat(b[0]).shape
        m = a_mat_shape[1] if self.transa else a_mat_shape[0]
        n = b_mat_shape[0] if self.transb else b_mat_shape[1]
        return (batch_size, m, n)

    def forward_cpu(self, x):
        a, b = x
        assert a.shape[0] == b.shape[0]
        batch_size = a.shape[0]
        shape = self._output_shape(a, b)
        ret = numpy.empty(shape)
        for i in six.moves.range(batch_size):
            ret[i] = _matmul_cpu(
                a[i], b[i], transa=self.transa, transb=self.transb)
        return ret,

    def backward_cpu(self, x, gy):
        a, b = x
        batch_size = a.shape[0]
        ga = numpy.empty(a.shape)
        gb = numpy.empty(b.shape)
        for i in six.moves.range(batch_size):
            ga[i] = _matmul_cpu(gy[0][i], b[i],
                                transb=not self.transb, transout=self.transa
                                ).reshape(a[0].shape)
            gb[i] = _matmul_cpu(a[i], gy[0][i],
                                transa=not self.transa, transout=self.transb
                                ).reshape(b[0].shape)
        return ga, gb

    def forward_gpu(self, x):
        a, b = x
        assert a.shape[0] == b.shape[0]
        shape = self._output_shape(a, b)
        ret = cuda.empty(shape)
        _batch_matmul_gpu(a, b,
                          transa=self.transa, transb=self.transb, out=ret)
        return ret,

    def backward_gpu(self, x, gy):
        a, b = x
        batch_size = a.shape[0]
        ga = cuda.empty((batch_size,) + _as_mat(a[0]).shape)
        gb = cuda.empty((batch_size,) + _as_mat(b[0]).shape)
        _batch_matmul_gpu(gy[0], b,
                          transb=not self.transb, transout=self.transa, out=ga)
        _batch_matmul_gpu(a, gy[0],
                          transa=not self.transa, transout=self.transb, out=gb)
        ga = ga.reshape(a.shape)
        gb = gb.reshape(b.shape)
        return ga, gb


def batch_matmul(a, b, transa=False, transb=False):
    """Compute matrix multiplication of two arrays in a batch manner.

    Args:
        a, b: Variables of 3-D or 2-D arrays.
            A 3-D array of shape (B, N, M) is considered as B NxM matrices.
            A 2-D array of shape (B, N,) is considered as B Nx1 matrices.
        transa (bool): If true, transpose each matrix in a.
        transb (bool): If true, transpose each matrix in b.

    Returns:
        ~chainer.Variable: Batch of matrix multiplications as a 3-D array.
    """
    return BatchMatMul(transa=transa, transb=transb)(a, b)

import ctypes

import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _mat_ptrs(a):
    """Creates an array of pointers to matrices

    Args:
        a: A batch of matrices on GPU
    Returns:
        GPU array of pointers to matrices
    """
    return cuda.to_gpu(numpy.arange(
        a.ptr, a.ptr + a.shape[0] * a.strides[0], a.strides[0],
        dtype=ctypes.c_void_p))


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
    return cuda.cublas.cublasSgemmBatched(
        cuda.get_cublas_handle(),
        _as_trans_op(transb),
        _as_trans_op(transa),
        n, m, k, alpha,
        _mat_ptrs(b).gpudata, k if transb else n,
        _mat_ptrs(a).gpudata, m if transa else k,
        beta, _mat_ptrs(out).gpudata, n, l)


def _check_ndim(in_type, lower=1, upper=2):
    type_check.expect(
        in_type.ndim >= lower,
        in_type.ndim <= upper
    )


def _convert_type(in_type, vector_ndim=1):
    if in_type.ndim.eval() == vector_ndim:
        in_type = type_check.Variable(
            type_check.TypeInfo(in_type.shape.eval() + (1,),
                                in_type.dtype),
            '%s(1-D array)' % in_type.name)
    else:
        in_type.name = '%s(2-D array)' % in_type.name
    return in_type


def _get_check_index(trans, right, row_idx=0, col_idx=1):
    if trans ^ right:
        return row_idx
    else:
        return col_idx


class MatMul(function.Function):
    def __init__(self, transa=False, transb=False):
        self.transa = transa
        self.transb = transb

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        a_type, b_type = in_types

        type_check.expect(
            a_type.dtype == numpy.float32,
            b_type.dtype == numpy.float32
        )

        _check_ndim(a_type)
        _check_ndim(b_type)

        a_type = _convert_type(a_type)
        b_type = _convert_type(b_type)
        a_idx = _get_check_index(self.transa, False)
        b_idx = _get_check_index(self.transb, True)
        type_check.expect(
            a_type.shape[a_idx] == b_type.shape[b_idx]
        )

    def forward_cpu(self, x):
        return _matmul_cpu(x[0], x[1], transa=self.transa, transb=self.transb),

    def forward_gpu(self, x):
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
        gx0 = _matmul_gpu(
            gy[0], x[1], transb=not self.transb, transout=self.transa
            ).reshape(x[0].shape)
        gx1 = _matmul_gpu(
            x[0], gy[0], transa=not self.transa, transout=self.transb
            ).reshape(x[1].shape)
        return gx0, gx1


def matmul(a, b, transa=False, transb=False):
    """Computes the matrix multiplication of two arrays.

    Args:
        a (Variable): The left operand of the matrix multiplication.
            A 1-D array of shape (N,) is considered as an Nx1 matrix.
            A 2-D array of shape (M, N) is considered as an MxN matrix.
        b (Variable): The right operand of the matrix multiplication.
            Its array is treated as a matrix in the same way as ``a``'s array.
        transa (bool): If true, transpose a.
        transb (bool): If true, transpose b.

    Returns:
        ~chainer.Variable: The result of the matrix multiplication as a 2-D
            array.
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

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        a_type, b_type = in_types

        type_check.expect(
            a_type.dtype == numpy.float32,
            b_type.dtype == numpy.float32
        )

        _check_ndim(a_type, lower=2, upper=3)
        _check_ndim(b_type, lower=2, upper=3)

        a_type = _convert_type(a_type, vector_ndim=2)
        b_type = _convert_type(b_type, vector_ndim=2)
        a_idx = _get_check_index(self.transa, False, row_idx=1, col_idx=2)
        b_idx = _get_check_index(self.transb, True, row_idx=1, col_idx=2)
        type_check.expect(
            a_type.shape[a_idx] == b_type.shape[b_idx]
        )

    def forward_cpu(self, x):
        a, b = x
        batch_size = a.shape[0]
        shape = self._output_shape(a, b)
        ret_dtype = numpy.find_common_type([a.dtype, b.dtype], [])
        ret = numpy.empty(shape, dtype=ret_dtype)
        for i in six.moves.range(batch_size):
            ret[i] = _matmul_cpu(
                a[i], b[i], transa=self.transa, transb=self.transb)
        return ret,

    def backward_cpu(self, x, gy):
        a, b = x
        batch_size = a.shape[0]
        ga = numpy.empty_like(a)
        gb = numpy.empty_like(b)
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
    """Computes the batch matrix multiplications of two sets of arrays.

    Args:
        a (Variable): The left operand of the batch matrix multiplications.
            A 2-D array of shape (B, N,) is considered as B Nx1 matrices.
            A 3-D array of shape (B, M, N) is considered as B MxN matrices.
        b (Variable): The right operand of the batch matrix multiplications.
            Its array is treated as matrices in the same way as ``a``'s array.
        transa (bool): If true, transpose each matrix in a.
        transb (bool): If true, transpose each matrix in b.

    Returns:
        ~chainer.Variable: The result of the batch matrix multiplications as a
            3-D array.
    """
    return BatchMatMul(transa=transa, transb=transb)(a, b)

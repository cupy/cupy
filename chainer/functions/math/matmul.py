import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import array
from chainer.utils import type_check


def _mat_ptrs(a):
    """Creates an array of pointers to matrices

    Args:
        a: A batch of matrices on GPU.
    Returns:
        GPU array of pointers to matrices.
    """
    if len(a) == 1:
        return cuda.cupy.full((1,), a.data.ptr, dtype=numpy.uintp)
    else:
        stride = a.strides[0]
        ptr = a.data.ptr
        return cuda.cupy.arange(ptr, ptr + stride * len(a), stride,
                                dtype=numpy.uintp)


def _as_batch_mat(x):
    return x.reshape(len(x), x.shape[1], -1)


def _get_batch_mat_shape(shape):
    s = 1
    for x in shape[2:]:
        s *= x
    return shape[:2] + (s,)


def _matmul(a, b, transa=False, transb=False, transout=False):
    a = array.as_mat(a)
    b = array.as_mat(b)
    if transa:
        a = a.T
    if transb:
        b = b.T
    if transout:
        # (A B)^T = B^T A^T
        a, b = b.T, a.T
    return a.dot(b)


def _get_ld(a):
    strides = a.strides[-2:]
    trans = numpy.argmin(strides)
    return trans, int(max(a.shape[trans - 2], max(strides) // a.itemsize))


def _batch_matmul_gpu(a, b, out, transa=False, transb=False, transout=False):
    a = _as_batch_mat(cuda.cupy.ascontiguousarray(a))
    b = _as_batch_mat(cuda.cupy.ascontiguousarray(b))
    trans_axis = (0, 2, 1)
    if transout:
        out = out.transpose(trans_axis)
    needtrans, _ = _get_ld(out)
    if needtrans == 1:
        # (A B)^T = B^T A^T
        a, b = b, a
        transa, transb = not transb, not transa
        out = out.transpose(trans_axis)
    if transa:
        a = a.transpose(trans_axis)
    if transb:
        b = b.transpose(trans_axis)

    transa, lda = _get_ld(a)
    transb, ldb = _get_ld(b)
    transout, ldout = _get_ld(out)
    la, n, ka = a.shape
    lb, kb, m = b.shape

    assert ka == kb
    assert transout == 0 or ldout == 1
    assert out.shape == (la, n, m)

    ap = _mat_ptrs(a)
    bp = _mat_ptrs(b)
    outp = _mat_ptrs(out)
    cuda.cublas.sgemmBatched(
        cuda.Device().cublas_handle,
        transa,
        transb,
        n, m, ka, 1.0,
        ap.data.ptr, lda,
        bp.data.ptr, ldb,
        0.0, outp.data.ptr, ldout, la)


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

    def forward(self, x):
        return _matmul(x[0], x[1], transa=self.transa, transb=self.transb),

    def backward(self, x, gy):
        gx0 = _matmul(
            gy[0], x[1], transb=not self.transb, transout=self.transa
            ).reshape(x[0].shape)
        gx1 = _matmul(
            x[0], gy[0], transa=not self.transa, transout=self.transb
            ).reshape(x[1].shape)
        return gx0, gx1


def matmul(a, b, transa=False, transb=False):
    """Computes the matrix multiplication of two arrays.

    Args:
        a (Variable): The left operand of the matrix multiplication.
            A 1-D array of shape ``(N,)`` is considered as an
            :math:`N \\times 1` matrix.
            A 2-D array of shape ``(M, N)`` is considered as an
            :math:`M \\times N` matrix.
        b (Variable): The right operand of the matrix multiplication.
            Its array is treated as a matrix in the same way as ``a``'s array.
        transa (bool): If ``True``, transpose a.
        transb (bool): If ``True``, transpose b.

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
        batch_size = len(a)
        m = _get_batch_mat_shape(a.shape)[2 if self.transa else 1]
        n = _get_batch_mat_shape(b.shape)[1 if self.transb else 2]
        return batch_size, m, n

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
            ret[i] = _matmul(
                a[i], b[i], transa=self.transa, transb=self.transb)
        return ret,

    def backward_cpu(self, x, gy):
        a, b = x
        ga = numpy.empty_like(a)
        gb = numpy.empty_like(b)
        a0shape = a[0].shape
        b0shape = b[0].shape
        for i in six.moves.range(len(a)):
            ga[i] = _matmul(
                gy[0][i], b[i], transb=not self.transb,
                transout=self.transa).reshape(a0shape)
            gb[i] = _matmul(
                a[i], gy[0][i], transa=not self.transa,
                transout=self.transb).reshape(b0shape)
        return ga, gb

    def forward_gpu(self, x):
        a, b = x
        shape = self._output_shape(a, b)
        ret = cuda.cupy.zeros(shape, dtype=a.dtype)
        _batch_matmul_gpu(
            a, b, transa=self.transa, transb=self.transb, out=ret)
        return ret,

    def backward_gpu(self, x, gy):
        a, b = x
        ga = cuda.cupy.empty(_get_batch_mat_shape(a.shape), a.dtype)
        gb = cuda.cupy.empty(_get_batch_mat_shape(b.shape), a.dtype)
        _batch_matmul_gpu(
            gy[0], b, transb=not self.transb, transout=self.transa, out=ga)
        _batch_matmul_gpu(
            a, gy[0], transa=not self.transa, transout=self.transb, out=gb)
        ga = ga.reshape(a.shape)
        gb = gb.reshape(b.shape)
        return ga, gb


def batch_matmul(a, b, transa=False, transb=False):
    """Computes the batch matrix multiplications of two sets of arrays.

    Args:
        a (Variable): The left operand of the batch matrix multiplications.
            A 2-D array of shape ``(B, N)`` is considered as B
            :math:`N \\times 1` matrices.
            A 3-D array of shape ``(B, M, N)`` is considered as B
            :math:`M \\times N` matrices.
        b (Variable): The right operand of the batch matrix multiplications.
            Its array is treated as matrices in the same way as ``a``'s array.
        transa (bool): If ``True``, transpose each matrix in a.
        transb (bool): If ``True``, transpose each matrix in b.

    Returns:
        ~chainer.Variable: The result of the batch matrix multiplications as a
            3-D array.
    """
    return BatchMatMul(transa=transa, transb=transb)(a, b)

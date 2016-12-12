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


def _get_ld(a):
    strides = a.strides[-2:]
    trans = numpy.argmin(strides)
    return trans, int(max(a.shape[trans - 2], max(strides) // a.itemsize))


def _get_batch_mat_shape(shape):
    s = 1
    for x in shape[2:]:
        s *= x
    return shape[:2] + (s,)


def _matmul(a, b, transa=False, transb=False, transout=False):
    a = array.as_mat(a)
    b = array.as_mat(b)
    if transout:
        # (A B)^T = B^T A^T
        transa, transb = not transb, not transa
        a, b = b, a
    if transa:
        a = a.T
    if transb:
        b = b.T
    return a.dot(b)


def _batch_matmul(a, b, transa=False, transb=False, transout=False):
    a = a.reshape(a.shape[:2] + (-1,))
    b = b.reshape(b.shape[:2] + (-1,))
    trans_axis = (0, 2, 1)
    if transout:
        transa, transb = not transb, not transa
        a, b = b, a
    if transa:
        a = a.transpose(trans_axis)
    if transb:
        b = b.transpose(trans_axis)
    xp = cuda.get_array_module(a)
    if xp is numpy:
        ret = numpy.empty(a.shape[:2] + b.shape[2:], dtype=a.dtype)
        for i in six.moves.range(len(a)):
            ret[i] = numpy.dot(a[i], b[i])
        return ret
    return xp.matmul(a, b)


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
            a_type.dtype.kind == 'f',
            a_type.dtype == b_type.dtype
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
        a, b = x
        return _matmul(a, b, transa=self.transa, transb=self.transb),

    def backward(self, x, gy):
        a, b = x
        ga = _matmul(
            gy[0], b, transb=not self.transb, transout=self.transa
        ).reshape(a.shape)
        gb = _matmul(
            a, gy[0], transa=not self.transa, transout=self.transb
        ).reshape(b.shape)
        return ga, gb


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
        transa (bool): If ``True``, transpose ``a``.
        transb (bool): If ``True``, transpose ``b``.

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

    def forward(self, x):
        a, b = x
        return _batch_matmul(a, b, self.transa, self.transb),

    def backward(self, x, gy):
        a, b = x
        ga = _batch_matmul(gy[0], b, transb=not self.transb,
                           transout=self.transa).reshape(a.shape)
        gb = _batch_matmul(a, gy[0], transa=not self.transa,
                           transout=self.transb).reshape(b.shape)
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
        transa (bool): If ``True``, transpose each matrix in ``a``.
        transb (bool): If ``True``, transpose each matrix in ``b``.

    Returns:
        ~chainer.Variable: The result of the batch matrix multiplications as a
            3-D array.
    """
    return BatchMatMul(transa=transa, transb=transb)(a, b)

import numpy.linalg

from chainer import cuda
from chainer import function
from chainer.functions.math import matmul
from chainer import utils
from chainer.utils import type_check


def _inv_gpu(b):
    # We do a batched LU decomposition on the GPU to compute the inverse
    # Change the shape of the array to be size=1 minibatch if necessary
    # Also copy the matrix as the elments will be modified in-place
    a = matmul._as_batch_mat(b).copy()
    n = a.shape[1]
    n_matrices = len(a)
    # Pivot array
    p = cuda.cupy.empty((n, n_matrices), dtype=numpy.int32)
    # Output array
    c = cuda.cupy.empty_like(a)
    # These arrays hold information on the execution success
    # or if the matrix was singular
    info = cuda.cupy.empty(n_matrices, dtype=numpy.int32)
    ap = matmul._mat_ptrs(a)
    cp = matmul._mat_ptrs(c)
    _, lda = matmul._get_ld(a)
    _, ldc = matmul._get_ld(c)
    handle = cuda.Device().cublas_handle
    cuda.cublas.sgetrfBatched(
        handle, n, ap.data.ptr, lda, p.data.ptr, info.data.ptr, n_matrices)
    cuda.cublas.sgetriBatched(
        handle, n, ap.data.ptr, lda, p.data.ptr, cp.data.ptr, ldc,
        info.data.ptr, n_matrices)
    return c, info


class Inv(function.Function):

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        a_type, = in_types
        type_check.expect(a_type.dtype == numpy.float32)
        # Only 2D array shapes allowed
        type_check.expect(a_type.ndim == 2)
        # Matrix inversion only allowed for square matrices
        type_check.expect(a_type.shape[0] == a_type.shape[1])

    def forward_cpu(self, x):
        self.invx = utils.force_array(numpy.linalg.inv(x[0]))
        return self.invx,

    def forward_gpu(self, x):
        shape = x[0].shape
        self.invx = _inv_gpu(x[0].reshape(1, *shape))[0].reshape(shape)
        return self.invx,

    def backward(self, x, gy):
        # Gradient is - x^-T (dx) x^-T
        x, = x
        xp = cuda.get_array_module(x)
        gx = xp.dot(xp.dot(-self.invx.T, gy[0]), self.invx.T)
        return gx,


class BatchInv(function.Function):

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        a_type, = in_types
        type_check.expect(a_type.dtype == numpy.float32)
        # Only a minibatch of 2D array shapes allowed
        type_check.expect(a_type.ndim == 3)
        # Matrix inversion only allowed for square matrices
        # so assert the last two dimensions are equal
        type_check.expect(a_type.shape[-1] == a_type.shape[-2])

    def forward_cpu(self, x):
        self.invx = utils.force_array(numpy.linalg.inv(x[0]))
        return self.invx,

    def forward_gpu(self, x):
        self.invx, _ = _inv_gpu(x[0])
        return self.invx,

    def backward(self, x, gy):
        # Unpack 1-length tuples
        gy, = gy
        # Gradient is - x^-T (dx) x^-T
        ret = matmul._batch_matmul(-self.invx, gy, transa=True)
        ret2 = matmul._batch_matmul(ret, self.invx, transb=True)
        return ret2,


def inv(a):
    """Computes the inverse of square matrix.

    Args:
        a (Variable): Input array to compute the inverse for. Shape of
            the array should be ``(n, n)`` where ``n`` is the dimensionality of
            a square matrix.

    Returns:
        ~chainer.Variable: Matrix inverse of ``a``.
    """
    return Inv()(a)


def batch_inv(a):
    """Computes the inverse of a batch of square matrices.

    Args:
        a (Variable): Input array to compute the inverse for. Shape of
            the array should be ``(m, n, n)`` where ``m`` is the number of
            matrices in the batch, and ``n`` is the dimensionality of a square
            matrix.

    Returns:
        ~chainer.Variable: Inverse of every matrix in the batch of matrices.
    """
    return BatchInv()(a)

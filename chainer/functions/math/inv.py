import numpy.linalg
import six

from chainer import cuda
from chainer import function
from chainer.functions.array.reshape import Reshape
from chainer.functions.math.matmul import _as_batch_mat
from chainer.functions.math.matmul import _batch_matmul_gpu
from chainer.functions.math.matmul import _check_ndim
from chainer.functions.math.matmul import _convert_type
from chainer.functions.math.matmul import _get_ld
from chainer.functions.math.matmul import _mat_ptrs
from chainer import utils


def _inv_gpu(b):
    # We do a batched LU decomposition on the GPU to compute
    # the inverse
    # Change the shape of the array to be size=1 minibatch if necessary
    # Also copy the matrix as the elments will be modified in-place
    a = _as_batch_mat(b).copy()
    n = int(a.shape[1])
    n_matrices = int(a.shape[0])
    # Pivot array
    p = cuda.cupy.zeros((n, n_matrices), dtype='int32')
    # Output array
    c = cuda.cupy.zeros_like(a)
    # These arrays hold information on the execution success
    # or if the matrix was singular
    info1 = cuda.cupy.zeros(n_matrices, dtype=numpy.intp)
    info2 = cuda.cupy.zeros(n_matrices, dtype=numpy.intp)
    ap = _mat_ptrs(a)
    cp = _mat_ptrs(c)
    _, lda = _get_ld(a)
    _, ldc = _get_ld(c)
    cuda.cublas.sgetrf(cuda.Device().cublas_handle, n, ap.data.ptr, lda,
                       p.data.ptr, info1.data.ptr, n_matrices)
    cuda.cublas.sgetri(cuda.Device().cublas_handle, n, ap.data.ptr, lda,
                       p.data.ptr, cp.data.ptr, ldc, info2.data.ptr,
                       n_matrices)
    return c


class BatchInv(function.Function):

    @property
    def label(self):
        return 'inv'

    def check_type_forward(self, in_types):
        utils.type_check.expect(in_types.size() == 1)
        a_type, = in_types
        a_type = _convert_type(a_type)
        utils.type_check.expect(a_type.dtype.kind == 'f')
        # Only a minibatch of 2D array shapes allowed
        _check_ndim(a_type, lower=3, upper=3)
        # Matrix inversion only allowed for square matrices
        # so assert the last two dimensions are equal
        utils.type_check.expect(a_type.shape[-1] == a_type.shape[-2])

    def forward_cpu(self, x):
        self.invx = utils.force_array(numpy.linalg.inv(x[0]))
        return self.invx,

    def forward_gpu(self, x):
        self.invx = _inv_gpu(x[0])
        return self.invx,

    def backward_cpu(self, x, gy):
        # Gradient is - x^-T (dx) x^-T
        x, = x
        gy, = gy
        batch_size = x.shape[0]
        gx = numpy.empty_like(x)
        for i in six.moves.range(batch_size):
            gx[i] = numpy.dot(-self.invx[i].T, gy[i])
            gx[i] = numpy.dot(gx[i], self.invx[i].T)
        return gx,

    def backward_gpu(self, x, gy):
        # Unpack 1-length tuples
        gy, = gy
        shape = gy.shape
        ret = cuda.zeros(shape)
        _batch_matmul_gpu(-self.invx, gy, out=ret, transa=True)
        ret2 = cuda.zeros(shape)
        _batch_matmul_gpu(ret, self.invx, out=ret2, transb=True)
        return utils.force_array(ret2),


def batch_inv(a):
    """Computes the inverse of a batch of square matrices.

    Args:
        a (Variable): Input array to compute the determinant for.
        Shape of the array should be (m, n, n) where m is the number
        of matrices in the batch, and n is the dimensionality of a square
        matrix.

    Returns:
        ~chainer.Variable: inverse of every matrix in the batch of matrices
    """
    return BatchInv()(a)


def inv(a):
    """Computes the inverse of of square matrix.

    Args:
        a (Variable): Input array to compute the determinant for.
        Shape of the array should be (n, n) where n is the dimensionality
        of a square matrix.

    Returns:
        ~chainer.Variable: matrix inverse of a
    """
    shape = (1, a.data.shape[0], a.data.shape[1])
    batched_a = Reshape(shape)(a)
    return BatchInv()(batched_a)

import atexit

import numpy
import six

import cupy
from cupy import cuda
from cupy.cuda import cusparse


_handles = {}


def get_handle():
    dev = cuda.get_device_id()
    if dev in _handles:
        return _handles[dev]
    handle = cusparse.create()
    _handles[dev] = handle
    return handle


@atexit.register
def reset_handles():
    global _handles
    handles = _handles
    _handles = {}

    for handle in six.itervalues(handles):
        cusparse.destroy(handle)


class MatDescriptor(object):

    def __init__(self, descriptor):
        self.descriptor = descriptor

    @classmethod
    def create(cls):
        descr = cusparse.createMatDescr()
        return MatDescriptor(descr)

    def __del__(self):
        if self.descriptor:
            cusparse.destroyMatDescr(self.descriptor)
            self.descriptor = None

    def set_mat_type(self, typ):
        cusparse.setMatType(self.descriptor, typ)

    def set_mat_index_base(self, base):
        cusparse.setMatIndexBase(self.descriptor, base)


def _cast_common_type(*xs):
    dtypes = [x.dtype for x in xs if x is not None]
    dtype = numpy.find_common_type(dtypes, [])
    return [x.astype(dtype) if x is not None and x.dtype != dtype else x
            for x in xs]


def _transpose_flag(trans):
    if trans:
        return cusparse.CUSPARSE_OPERATION_TRANSPOSE
    else:
        return cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE


def _call_cusparse(name, dtype, *args):
    if dtype == 'f':
        prefix = 's'
    elif dtype == 'd':
        prefix = 'd'
    else:
        raise TypeError
    f = getattr(cusparse, prefix + name)
    return f(*args)


def csrmv(a, x, y=None, alpha=1, beta=0, transa=False):
    """Matrix-vector product for a CSR-matrix and a dense vector.

    .. math::

       y = \\alpha * op(A) x + \\beta y,

    Args:
        a (cupy.cusparse.csr_matrix): Matrix A.
        x (cupy.ndarray): Vector x.
        y (cupy.ndarray or None): Vector y.
        alpha (float): Coefficient for x.
        beta (float): Coefficient for y.
        transa (bool): If ``True``, transpose of ``A`` is used.

    Returns:
        cupy.ndarray: Calculated ``y``.

    """
    assert a.shape[1] == len(x)
    a_shape = a.shape if not transa else a.shape[::-1]
    m, n = a_shape
    a, x, y = _cast_common_type(a, x, y)
    dtype = a.dtype
    if y is None:
        y = cupy.zeros(m, dtype)
    else:
        y = cupy.asfortranarray(y)
    alpha = numpy.array(alpha, dtype).ctypes
    beta = numpy.array(beta, dtype).ctypes
    _call_cusparse(
        'csrmv', dtype,
        get_handle(), _transpose_flag(transa),
        m, n, a.nnz, alpha.data, a._descr.descriptor,
        a.data.data.ptr, a.indptr.data.ptr, a.indices.data.ptr,
        x.data.ptr, beta.data, y.data.ptr)

    return y


def csrmm(a, b, c, alpha=1, beta=0, transa=False):
    """Matrix-matrix product for CSR-matrixes.

    .. math::

       C = \\alpha o_a(A) B + \\beta C

    Args:
        a (cupy.sparse.csr): Sparse matrix A.
        b (cupy.ndarray): Dense matrix B.
        c (cupy.ndarray): Dense matrix C.
        alpha (float): Coefficient for AB.
        beta (float): Coefficient for C.
        transa (bool): If ``True``, transpose of A is used.

    Returns:
        cupy.ndarray: Calculated C.

    """
    assert a.ndim == b.ndim == 2
    a_shape = a.shape if not transa else a.shape[::-1]
    assert a_shape[1] == b.shape[0]
    m, k = a_shape
    n = b.shape[1]

    a, b, c = _cast_common_type(a, b, c)
    if c is None:
        c = cupy.zeros((m, n), a.dtype, 'F')

    b = cupy.asfortranarray(b)
    c = cupy.asfortranarray(c)
    ldb = k
    ldc = m

    alpha = numpy.array(alpha, a.dtype).ctypes
    beta = numpy.array(beta, a.dtype).ctypes
    _call_cusparse(
        'csrmm', a.dtype,
        get_handle(), _transpose_flag(transa), m, n, k, a.nnz,
        alpha.data, a._descr.descriptor, a.data.data.ptr,
        a.indptr.data.ptr, a.indices.data.ptr,
        b.data.ptr, ldb, beta.data, c.data.ptr, ldc)
    return c


def csrmm2(a, b, c=None, alpha=1, beta=0, transa=False, transb=False):
    """Matrix-matrix product for CSR-matrixes.

    .. math::

       C = \\alpha o_a(A) o_b(B) + \\beta C

    Args:
        a (cupy.sparse.csr): Sparse matrix A.
        b (cupy.ndarray): Dense matrix B.
        c (cupy.ndarray or None): Dense matrix C.
        alpha (float): Coefficient for AB.
        beta (float): Coefficient for C.
        transa (bool): If ``True``, transpose of A is used.
        transb (bool): If ``True``, transpose of B is used.

    Returns:
        cupy.ndarray: Calculated C.

    """
    assert a.ndim == b.ndim == 2
    a_shape = a.shape if not transa else a.shape[::-1]
    b_shape = b.shape if not transb else b.shape[::-1]
    assert a_shape[1] == b_shape[0]
    m, k = a_shape
    n = b_shape[1]

    a, b, c = _cast_common_type(a, b, c)
    if c is None:
        c = cupy.zeros((m, n), a.dtype, 'F')

    b = cupy.asfortranarray(b)
    c = cupy.asfortranarray(c)
    ldb = b.shape[0]
    ldc = c.shape[0]
    op_a = _transpose_flag(transa)
    op_b = _transpose_flag(transb)
    alpha = numpy.array(alpha, a.dtype).ctypes
    beta = numpy.array(beta, a.dtype).ctypes
    _call_cusparse(
        'csrmm2', a.dtype,
        get_handle(), op_a, op_b, m, n, k, a.nnz,
        alpha.data, a._descr.descriptor, a.data.data.ptr,
        a.indptr.data.ptr, a.indices.data.ptr,
        b.data.ptr, ldb, beta.data, c.data.ptr, ldc)
    return c


def csrgeam(a, b, alpha=1, beta=1):
    """Matrix-matrix addition.

    .. math::
        C = \\alpha A + \\beta B

    Args:
        a (cupy.sparse.csr_matrix): Sparse matrix A.
        b (cupy.sparse.csr_matrix): Sparse matrix B.
        alpha (float): Coefficient for A.
        beta (float): Coefficient for B.

    Returns:
        cupy.sparse.csr_matrix: Result matrix.

    """
    if a.shape != b.shape:
        raise ValueError('inconsistent shapes')

    handle = get_handle()
    m, n = a.shape
    a, b = _cast_common_type(a, b)
    nnz = numpy.empty((), 'i')
    cusparse.setPointerMode(
        handle, cusparse.CUSPARSE_POINTER_MODE_HOST)

    c_descr = MatDescriptor.create()
    c_indptr = cupy.empty(m + 1, 'i')

    cusparse.xcsrgeamNnz(
        handle, m, n,
        a._descr.descriptor, a.nnz, a.indptr.data.ptr, a.indices.data.ptr,
        b._descr.descriptor, b.nnz, b.indptr.data.ptr, b.indices.data.ptr,
        c_descr.descriptor, c_indptr.data.ptr, nnz.ctypes.data)

    c_indices = cupy.empty(int(nnz), 'i')
    c_data = cupy.empty(int(nnz), a.dtype)
    alpha = numpy.array(alpha, a.dtype).ctypes
    beta = numpy.array(beta, a.dtype).ctypes
    _call_cusparse(
        'csrgeam', a.dtype,
        handle, m, n, alpha.data,
        a._descr.descriptor, a.nnz, a.data.data.ptr,
        a.indptr.data.ptr, a.indices.data.ptr, beta.data,
        b._descr.descriptor, b.nnz, b.data.data.ptr,
        b.indptr.data.ptr, b.indices.data.ptr,
        c_descr.descriptor, c_data.data.ptr, c_indptr.data.ptr,
        c_indices.data.ptr)

    return cupy.sparse.csr_matrix((c_data, c_indices, c_indptr), shape=a.shape)


def csrgemm(a, b, transa=False, transb=False):
    """Matrix-matrix product for CSR-matrix.

    math::
       C = op(A) op(B),

    Args:
        a (cupy.sparse.csr): Sparse matrix A.
        b (cupy.sparse.csr): Sparse matrix B.
        transa (bool): If ``True``, transpose of A is used.
        transb (bool): If ``True``, transpose of B is used.

    Returns:
        cupy.ndarray: Calculated C.

    """
    assert a.ndim == b.ndim == 2
    a_shape = a.shape if not transa else a.shape[::-1]
    b_shape = b.shape if not transb else b.shape[::-1]
    assert a_shape[1] == b_shape[0]

    handle = get_handle()
    m, k = a_shape
    n = b_shape[1]

    a, b = _cast_common_type(a, b)

    op_a = _transpose_flag(transa)
    op_b = _transpose_flag(transb)

    nnz = numpy.empty((), 'i')
    cusparse.setPointerMode(
        handle, cusparse.CUSPARSE_POINTER_MODE_HOST)

    c_descr = MatDescriptor.create()
    c_indptr = cupy.empty(m + 1, 'i')

    cusparse.xcsrgemmNnz(
        handle, op_a, op_b, m, n, k, a._descr.descriptor, a.nnz,
        a.indptr.data.ptr, a.indices.data.ptr, b._descr.descriptor, b.nnz,
        b.indptr.data.ptr, b.indices.data.ptr, c_descr.descriptor,
        c_indptr.data.ptr, nnz.ctypes.data)

    c_indices = cupy.empty(int(nnz), 'i')
    c_data = cupy.empty(int(nnz), a.dtype)
    _call_cusparse(
        'csrgemm', a.dtype,
        handle, op_a, op_b, m, n, k, a._descr.descriptor, a.nnz,
        a.data.data.ptr, a.indptr.data.ptr, a.indices.data.ptr,
        b._descr.descriptor, b.nnz, b.data.data.ptr, b.indptr.data.ptr,
        b.indices.data.ptr,
        c_descr.descriptor, c_data.data.ptr, c_indptr.data.ptr,
        c_indices.data.ptr)

    return cupy.sparse.csr_matrix((c_data, c_indices, c_indptr), shape=(m, n))


def csr2dense(x, out=None):
    """Convert CSR-matrix to a dense matrix.

    Args:
        x (cupy.sparse.csr_matrix): A sparse matrix to convert.
        out (cupy.ndarray or None): A dense metrix to store the result.

    Returns:
        cupy.ndarray: Converted result.

    """
    dtype = x.dtype
    assert dtype == 'f' or dtype == 'd'
    if out is None:
        out = cupy.empty(x.shape, dtype=dtype, order='F')
    else:
        out = out.asfortranarray(out)

    _call_cusparse(
        'csr2dense', x.dtype,
        get_handle(), x.shape[0], x.shape[1], x._descr.descriptor,
        x.data.data.ptr, x.indptr.data.ptr, x.indices.data.ptr,
        out.data.ptr, x.shape[0])

    return out


def csrsort(x):
    """Sort indices of CSR-matrix in place.

    Args:
        x (cupy.sparse.csr_matrix): A sparse matrix to sort.

    """
    handle = get_handle()
    m, n = x.shape
    nnz = x.nnz

    buffer_size = cusparse.xcsrsort_bufferSizeExt(
        handle, m, n, nnz, x.indptr.data.ptr,
        x.indices.data.ptr)
    buf = cupy.empty(buffer_size, 'b')
    P = cupy.empty(nnz, 'i')
    cusparse.createIdentityPermutation(handle, nnz, P.data.ptr)
    cusparse.xcsrsort(
        handle, m, n, nnz, x._descr.descriptor, x.indptr.data.ptr,
        x.indices.data.ptr, P.data.ptr, buf.data.ptr)
    _call_cusparse(
        'gthr', x.dtype,
        handle, nnz, x.data.data.ptr, x.data.data.ptr,
        P.data.ptr, cusparse.CUSPARSE_INDEX_BASE_ZERO)


def cscsort(x):
    """Sort indices of CSC-matrix in place.

    Args:
        x (cupy.sparse.csc_matrix): A sparse matrix to sort.

    """
    handle = get_handle()
    m, n = x.shape
    nnz = x.nnz

    buffer_size = cusparse.xcscsort_bufferSizeExt(
        handle, m, n, nnz, x.indptr.data.ptr,
        x.indices.data.ptr)
    buf = cupy.empty(buffer_size, 'b')
    P = cupy.empty(nnz, 'i')
    cusparse.createIdentityPermutation(handle, nnz, P.data.ptr)
    cusparse.xcscsort(
        handle, m, n, nnz, x._descr.descriptor, x.indptr.data.ptr,
        x.indices.data.ptr, P.data.ptr, buf.data.ptr)
    _call_cusparse(
        'gthr', x.dtype,
        handle, nnz, x.data.data.ptr, x.data.data.ptr,
        P.data.ptr, cusparse.CUSPARSE_INDEX_BASE_ZERO)


def coosort(x):
    handle = get_handle()
    m, n = x.shape
    nnz = x.nnz

    buffer_size = cusparse.xcoosort_bufferSizeExt(
        handle, m, n, nnz, x.row.data.ptr, x.col.data.ptr)
    buf = cupy.empty(buffer_size, 'b')
    P = cupy.empty(nnz, 'i')
    cusparse.createIdentityPermutation(handle, nnz, P.data.ptr)
    cusparse.xcoosortByRow(
        handle, m, n, nnz, x.row.data.ptr, x.col.data.ptr,
        P.data.ptr, buf.data.ptr)
    cusparse.sgthr(
        handle, nnz, x.data.data.ptr, x.data.data.ptr,
        P.data.ptr, cusparse.CUSPARSE_INDEX_BASE_ZERO)


def coo2csr(x):
    m = x.shape[0]
    indptr = cupy.empty(m + 1, 'i')
    cusparse.xcoo2csr(
        get_handle(), x.row.data.ptr, x.nnz, m,
        indptr.data.ptr, cusparse.CUSPARSE_INDEX_BASE_ZERO)
    return cupy.sparse.csr.csr_matrix(
        (x.data, x.col, indptr), shape=x.shape)


def csr2csc(x):
    m, n = x.shape
    nnz = x.nnz
    data = cupy.empty(nnz, x.dtype)
    indptr = cupy.empty(n + 1, 'i')
    indices = cupy.empty(nnz, 'i')

    _call_cusparse(
        'csr2csc', x.dtype,
        get_handle(), m, n, nnz, x.data.data.ptr,
        x.indptr.data.ptr, x.indices.data.ptr,
        data.data.ptr, indices.data.ptr, indptr.data.ptr,
        cusparse.CUSPARSE_ACTION_NUMERIC,
        cusparse.CUSPARSE_INDEX_BASE_ZERO)
    return cupy.sparse.csc_matrix((data, indices, indptr), shape=x.shape)

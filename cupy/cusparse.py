import functools

import numpy

import cupy
from cupy.cuda import cusparse
from cupy.cuda import runtime
from cupy.cuda import device
from cupy import util
import cupyx.scipy.sparse


class MatDescriptor(object):

    def __init__(self, descriptor):
        self.descriptor = descriptor

    @classmethod
    def create(cls):
        descr = cusparse.createMatDescr()
        return MatDescriptor(descr)

    def __reduce__(self):
        return self.create, ()

    def __del__(self, is_shutting_down=util.is_shutting_down):
        if is_shutting_down():
            return
        if self.descriptor:
            cusparse.destroyMatDescr(self.descriptor)
            self.descriptor = None

    def set_mat_type(self, typ):
        cusparse.setMatType(self.descriptor, typ)

    def set_mat_index_base(self, base):
        cusparse.setMatIndexBase(self.descriptor, base)


def _cast_common_type(*xs):
    dtypes = [x.dtype for x in xs if x is not None]
    dtype = functools.reduce(numpy.promote_types, dtypes)
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
    elif dtype == 'F':
        prefix = 'c'
    elif dtype == 'D':
        prefix = 'z'
    else:
        raise TypeError
    f = getattr(cusparse, prefix + name)
    return f(*args)


def _dtype_to_DataType(dtype):
    if dtype == 'f':
        return runtime.CUDA_R_32F
    elif dtype == 'd':
        return runtime.CUDA_R_64F
    elif dtype == 'F':
        return runtime.CUDA_C_32F
    elif dtype == 'D':
        return runtime.CUDA_C_64F
    else:
        raise TypeError


def csrmv(a, x, y=None, alpha=1, beta=0, transa=False):
    """Matrix-vector product for a CSR-matrix and a dense vector.

    .. math::

       y = \\alpha * o_a(A) x + \\beta y,

    where :math:`o_a` is a transpose function when ``transa`` is ``True`` and
    is an identity function otherwise.

    Args:
        a (cupy.cusparse.csr_matrix): Matrix A.
        x (cupy.ndarray): Vector x.
        y (cupy.ndarray or None): Vector y. It must be F-contiguous.
        alpha (float): Coefficient for x.
        beta (float): Coefficient for y.
        transa (bool): If ``True``, transpose of ``A`` is used.

    Returns:
        cupy.ndarray: Calculated ``y``.

    """
    assert y is None or y.flags.f_contiguous

    a_shape = a.shape if not transa else a.shape[::-1]
    if a_shape[1] != len(x):
        raise ValueError('dimension mismatch')

    handle = device.get_cusparse_handle()
    m, n = a_shape
    a, x, y = _cast_common_type(a, x, y)
    dtype = a.dtype
    if y is None:
        y = cupy.zeros(m, dtype)
    alpha = numpy.array(alpha, dtype).ctypes
    beta = numpy.array(beta, dtype).ctypes

    _call_cusparse(
        'csrmv', dtype,
        handle, _transpose_flag(transa),
        a.shape[0], a.shape[1], a.nnz, alpha.data, a._descr.descriptor,
        a.data.data.ptr, a.indptr.data.ptr, a.indices.data.ptr,
        x.data.ptr, beta.data, y.data.ptr)

    return y


def csrmvExIsAligned(a, x, y=None):
    """Check if the pointers of arguments for csrmvEx are aligned or not

    Args:
        a (cupy.cusparse.csr_matrix): Matrix A.
        x (cupy.ndarray): Vector x.
        y (cupy.ndarray or None): Vector y.

        Check if a, x, y pointers are aligned by 128 bytes as
        required by csrmvEx.

    Returns:
        bool: ``True`` if all pointers are aligned.
              ``False`` if otherwise.

    """

    if a.data.data.ptr % 128 != 0:
        return False
    if a.indptr.data.ptr % 128 != 0:
        return False
    if a.indices.data.ptr % 128 != 0:
        return False
    if x.data.ptr % 128 != 0:
        return False
    if y is not None and y.data.ptr % 128 != 0:
        return False
    return True


def csrmvEx(a, x, y=None, alpha=1, beta=0, merge_path=True):
    """Matrix-vector product for a CSR-matrix and a dense vector.

    .. math::

       y = \\alpha * A x + \\beta y,

    Args:
        a (cupy.cusparse.csr_matrix): Matrix A.
        x (cupy.ndarray): Vector x.
        y (cupy.ndarray or None): Vector y. It must be F-contiguous.
        alpha (float): Coefficient for x.
        beta (float): Coefficient for y.
        merge_path (bool): If ``True``, merge path algorithm is used.

        All pointers must be aligned with 128 bytes.

    Returns:
        cupy.ndarray: Calculated ``y``.

    """
    assert y is None or y.flags.f_contiguous

    if a.shape[1] != len(x):
        raise ValueError('dimension mismatch')

    handle = device.get_cusparse_handle()
    m, n = a.shape

    a, x, y = _cast_common_type(a, x, y)
    dtype = a.dtype
    if y is None:
        y = cupy.zeros(m, dtype)

    datatype = _dtype_to_DataType(dtype)
    algmode = cusparse.CUSPARSE_ALG_MERGE_PATH if \
        merge_path else cusparse.CUSPARSE_ALG_NAIVE
    transa_flag = cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE

    alpha = numpy.array(alpha, dtype).ctypes
    beta = numpy.array(beta, dtype).ctypes

    assert csrmvExIsAligned(a, x, y)

    bufferSize = cusparse.csrmvEx_bufferSize(
        handle, algmode, transa_flag,
        a.shape[0], a.shape[1], a.nnz, alpha.data, datatype,
        a._descr.descriptor, a.data.data.ptr, datatype,
        a.indptr.data.ptr, a.indices.data.ptr,
        x.data.ptr, datatype, beta.data, datatype,
        y.data.ptr, datatype, datatype)

    buf = cupy.empty(bufferSize, 'b')
    assert buf.data.ptr % 128 == 0

    cusparse.csrmvEx(
        handle, algmode, transa_flag,
        a.shape[0], a.shape[1], a.nnz, alpha.data, datatype,
        a._descr.descriptor, a.data.data.ptr, datatype,
        a.indptr.data.ptr, a.indices.data.ptr,
        x.data.ptr, datatype, beta.data, datatype,
        y.data.ptr, datatype, datatype, buf.data.ptr)
    return y


def csrmm(a, b, c=None, alpha=1, beta=0, transa=False):
    """Matrix-matrix product for a CSR-matrix and a dense matrix.

    .. math::

       C = \\alpha o_a(A) B + \\beta C,

    where :math:`o_a` is a transpose function when ``transa`` is ``True`` and
    is an identity function otherwise.

    Args:
        a (cupyx.scipy.sparse.csr): Sparse matrix A.
        b (cupy.ndarray): Dense matrix B. It must be F-contiguous.
        c (cupy.ndarray or None): Dense matrix C. It must be F-contiguous.
        alpha (float): Coefficient for AB.
        beta (float): Coefficient for C.
        transa (bool): If ``True``, transpose of A is used.

    Returns:
        cupy.ndarray: Calculated C.

    """
    assert a.ndim == b.ndim == 2
    assert b.flags.f_contiguous
    assert c is None or c.flags.f_contiguous

    a_shape = a.shape if not transa else a.shape[::-1]
    if a_shape[1] != b.shape[0]:
        raise ValueError('dimension mismatch')

    handle = device.get_cusparse_handle()
    m, k = a_shape
    n = b.shape[1]

    a, b, c = _cast_common_type(a, b, c)
    if c is None:
        c = cupy.zeros((m, n), a.dtype, 'F')

    ldb = k
    ldc = m

    alpha = numpy.array(alpha, a.dtype).ctypes
    beta = numpy.array(beta, a.dtype).ctypes
    _call_cusparse(
        'csrmm', a.dtype,
        handle, _transpose_flag(transa),
        a.shape[0], n, a.shape[1], a.nnz,
        alpha.data, a._descr.descriptor, a.data.data.ptr,
        a.indptr.data.ptr, a.indices.data.ptr,
        b.data.ptr, ldb, beta.data, c.data.ptr, ldc)
    return c


def csrmm2(a, b, c=None, alpha=1.0, beta=0.0, transa=False, transb=False):
    """Matrix-matrix product for a CSR-matrix and a dense matrix.

    .. math::

       C = \\alpha o_a(A) o_b(B) + \\beta C,

    where :math:`o_a` and :math:`o_b` are transpose functions when ``transa``
    and ``tranb`` are ``True`` respectively. And they are identity functions
    otherwise.
    It is forbidden that both ``transa`` and ``transb`` are ``True`` in
    cuSPARSE specification.

    Args:
        a (cupyx.scipy.sparse.csr): Sparse matrix A.
        b (cupy.ndarray): Dense matrix B. It must be F-contiguous.
        c (cupy.ndarray or None): Dense matrix C. It must be F-contiguous.
        alpha (float): Coefficient for AB.
        beta (float): Coefficient for C.
        transa (bool): If ``True``, transpose of A is used.
        transb (bool): If ``True``, transpose of B is used.

    Returns:
        cupy.ndarray: Calculated C.

    """
    assert a.ndim == b.ndim == 2
    assert a.has_canonical_format
    assert b.flags.f_contiguous
    assert c is None or c.flags.f_contiguous
    assert not (transa and transb)

    a_shape = a.shape if not transa else a.shape[::-1]
    b_shape = b.shape if not transb else b.shape[::-1]
    if a_shape[1] != b_shape[0]:
        raise ValueError('dimension mismatch')

    handle = device.get_cusparse_handle()
    m, k = a_shape
    n = b_shape[1]

    a, b, c = _cast_common_type(a, b, c)
    if c is None:
        c = cupy.zeros((m, n), a.dtype, 'F')

    ldb = b.shape[0]
    ldc = c.shape[0]
    op_a = _transpose_flag(transa)
    op_b = _transpose_flag(transb)
    alpha = numpy.array(alpha, a.dtype).ctypes
    beta = numpy.array(beta, a.dtype).ctypes
    _call_cusparse(
        'csrmm2', a.dtype,
        handle, op_a, op_b, a.shape[0], n, a.shape[1], a.nnz,
        alpha.data, a._descr.descriptor, a.data.data.ptr,
        a.indptr.data.ptr, a.indices.data.ptr,
        b.data.ptr, ldb, beta.data, c.data.ptr, ldc)
    return c


def csrgeam(a, b, alpha=1, beta=1):
    """Matrix-matrix addition.

    .. math::
        C = \\alpha A + \\beta B

    Args:
        a (cupyx.scipy.sparse.csr_matrix): Sparse matrix A.
        b (cupyx.scipy.sparse.csr_matrix): Sparse matrix B.
        alpha (float): Coefficient for A.
        beta (float): Coefficient for B.

    Returns:
        cupyx.scipy.sparse.csr_matrix: Result matrix.

    """
    assert a.has_canonical_format
    assert b.has_canonical_format
    if a.shape != b.shape:
        raise ValueError('inconsistent shapes')

    handle = device.get_cusparse_handle()
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

    c = cupyx.scipy.sparse.csr_matrix(
        (c_data, c_indices, c_indptr), shape=a.shape)
    c._has_canonical_format = True
    return c


def csrgemm(a, b, transa=False, transb=False):
    """Matrix-matrix product for CSR-matrix.

    math::
       C = op(A) op(B),

    Args:
        a (cupyx.scipy.sparse.csr_matrix): Sparse matrix A.
        b (cupyx.scipy.sparse.csr_matrix): Sparse matrix B.
        transa (bool): If ``True``, transpose of A is used.
        transb (bool): If ``True``, transpose of B is used.

    Returns:
        cupyx.scipy.sparse.csr_matrix: Calculated C.

    """
    assert a.ndim == b.ndim == 2
    assert a.has_canonical_format
    assert b.has_canonical_format
    a_shape = a.shape if not transa else a.shape[::-1]
    b_shape = b.shape if not transb else b.shape[::-1]
    if a_shape[1] != b_shape[0]:
        raise ValueError('dimension mismatch')

    handle = device.get_cusparse_handle()
    m, k = a_shape
    n = b_shape[1]

    a, b = _cast_common_type(a, b)

    if a.nnz == 0 or b.nnz == 0:
        return cupyx.scipy.sparse.csr_matrix((m, n), dtype=a.dtype)

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

    c = cupyx.scipy.sparse.csr_matrix(
        (c_data, c_indices, c_indptr), shape=(m, n))
    c._has_canonical_format = True
    return c


def csr2dense(x, out=None):
    """Converts CSR-matrix to a dense matrix.

    Args:
        x (cupyx.scipy.sparse.csr_matrix): A sparse matrix to convert.
        out (cupy.ndarray or None): A dense metrix to store the result.
            It must be F-contiguous.

    Returns:
        cupy.ndarray: Converted result.

    """
    dtype = x.dtype
    assert dtype.char in 'fdFD'
    if out is None:
        out = cupy.empty(x.shape, dtype=dtype, order='F')
    else:
        assert out.flags.f_contiguous

    handle = device.get_cusparse_handle()
    _call_cusparse(
        'csr2dense', x.dtype,
        handle, x.shape[0], x.shape[1], x._descr.descriptor,
        x.data.data.ptr, x.indptr.data.ptr, x.indices.data.ptr,
        out.data.ptr, x.shape[0])

    return out


def csc2dense(x, out=None):
    """Converts CSC-matrix to a dense matrix.

    Args:
        x (cupyx.scipy.sparse.csc_matrix): A sparse matrix to convert.
        out (cupy.ndarray or None): A dense metrix to store the result.
            It must be F-contiguous.

    Returns:
        cupy.ndarray: Converted result.

    """
    dtype = x.dtype
    assert dtype.char in 'fdFD'
    if out is None:
        out = cupy.empty(x.shape, dtype=dtype, order='F')
    else:
        assert out.flags.f_contiguous

    handle = device.get_cusparse_handle()
    _call_cusparse(
        'csc2dense', x.dtype,
        handle, x.shape[0], x.shape[1], x._descr.descriptor,
        x.data.data.ptr, x.indices.data.ptr, x.indptr.data.ptr,
        out.data.ptr, x.shape[0])

    return out


def csrsort(x):
    """Sorts indices of CSR-matrix in place.

    Args:
        x (cupyx.scipy.sparse.csr_matrix): A sparse matrix to sort.

    """
    nnz = x.nnz
    if nnz == 0:
        return
    handle = device.get_cusparse_handle()
    m, n = x.shape

    buffer_size = cusparse.xcsrsort_bufferSizeExt(
        handle, m, n, nnz, x.indptr.data.ptr,
        x.indices.data.ptr)
    buf = cupy.empty(buffer_size, 'b')
    P = cupy.empty(nnz, 'i')
    data_orig = x.data.copy()
    cusparse.createIdentityPermutation(handle, nnz, P.data.ptr)
    cusparse.xcsrsort(
        handle, m, n, nnz, x._descr.descriptor, x.indptr.data.ptr,
        x.indices.data.ptr, P.data.ptr, buf.data.ptr)
    _call_cusparse(
        'gthr', x.dtype,
        handle, nnz, data_orig.data.ptr, x.data.data.ptr,
        P.data.ptr, cusparse.CUSPARSE_INDEX_BASE_ZERO)


def cscsort(x):
    """Sorts indices of CSC-matrix in place.

    Args:
        x (cupyx.scipy.sparse.csc_matrix): A sparse matrix to sort.

    """
    nnz = x.nnz
    if nnz == 0:
        return
    handle = device.get_cusparse_handle()
    m, n = x.shape

    buffer_size = cusparse.xcscsort_bufferSizeExt(
        handle, m, n, nnz, x.indptr.data.ptr,
        x.indices.data.ptr)
    buf = cupy.empty(buffer_size, 'b')
    P = cupy.empty(nnz, 'i')
    data_orig = x.data.copy()
    cusparse.createIdentityPermutation(handle, nnz, P.data.ptr)
    cusparse.xcscsort(
        handle, m, n, nnz, x._descr.descriptor, x.indptr.data.ptr,
        x.indices.data.ptr, P.data.ptr, buf.data.ptr)
    _call_cusparse(
        'gthr', x.dtype,
        handle, nnz, data_orig.data.ptr, x.data.data.ptr,
        P.data.ptr, cusparse.CUSPARSE_INDEX_BASE_ZERO)


def coosort(x):
    """Sorts indices of COO-matrix in place.

    Args:
        x (cupyx.scipy.sparse.coo_matrix): A sparse matrix to sort.

    """
    nnz = x.nnz
    if nnz == 0:
        return
    handle = device.get_cusparse_handle()
    m, n = x.shape

    buffer_size = cusparse.xcoosort_bufferSizeExt(
        handle, m, n, nnz, x.row.data.ptr, x.col.data.ptr)
    buf = cupy.empty(buffer_size, 'b')
    P = cupy.empty(nnz, 'i')
    data_orig = x.data.copy()
    cusparse.createIdentityPermutation(handle, nnz, P.data.ptr)
    cusparse.xcoosortByRow(
        handle, m, n, nnz, x.row.data.ptr, x.col.data.ptr,
        P.data.ptr, buf.data.ptr)
    _call_cusparse(
        'gthr', x.dtype,
        handle, nnz, data_orig.data.ptr, x.data.data.ptr,
        P.data.ptr, cusparse.CUSPARSE_INDEX_BASE_ZERO)


def coo2csr(x):
    handle = device.get_cusparse_handle()
    m = x.shape[0]
    indptr = cupy.empty(m + 1, 'i')
    cusparse.xcoo2csr(
        handle, x.row.data.ptr, x.nnz, m,
        indptr.data.ptr, cusparse.CUSPARSE_INDEX_BASE_ZERO)
    return cupyx.scipy.sparse.csr.csr_matrix(
        (x.data, x.col, indptr), shape=x.shape)


def csr2coo(x, data, indices):
    """Converts a CSR-matrix to COO format.

    Args:
        x (cupyx.scipy.sparse.csr_matrix): A matrix to be converted.
        data (cupy.ndarray): A data array for converted data.
        indices (cupy.ndarray): An index array for converted data.

    Returns:
        cupyx.scipy.sparse.coo_matrix: A converted matrix.

    """
    handle = device.get_cusparse_handle()
    m = x.shape[0]
    nnz = len(x.data)
    row = cupy.empty(nnz, 'i')
    cusparse.xcsr2coo(
        handle, x.indptr.data.ptr, nnz, m, row.data.ptr,
        cusparse.CUSPARSE_INDEX_BASE_ZERO)
    # data and indices did not need to be copied already
    return cupyx.scipy.sparse.coo_matrix(
        (data, (row, indices)), shape=x.shape)


def csr2csc(x):
    handle = device.get_cusparse_handle()
    m, n = x.shape
    nnz = x.nnz
    data = cupy.empty(nnz, x.dtype)
    indptr = cupy.empty(n + 1, 'i')
    indices = cupy.empty(nnz, 'i')

    _call_cusparse(
        'csr2csc', x.dtype,
        handle, m, n, nnz, x.data.data.ptr,
        x.indptr.data.ptr, x.indices.data.ptr,
        data.data.ptr, indices.data.ptr, indptr.data.ptr,
        cusparse.CUSPARSE_ACTION_NUMERIC,
        cusparse.CUSPARSE_INDEX_BASE_ZERO)
    return cupyx.scipy.sparse.csc_matrix(
        (data, indices, indptr), shape=x.shape)


def dense2csc(x):
    """Converts a dense matrix in CSC format.

    Args:
        x (cupy.ndarray): A matrix to be converted.

    Returns:
        cupyx.scipy.sparse.csc_matrix: A converted matrix.

    """
    assert x.ndim == 2
    x = cupy.asfortranarray(x)
    nnz = numpy.empty((), dtype='i')
    handle = device.get_cusparse_handle()
    m, n = x.shape

    descr = MatDescriptor.create()
    nnz_per_col = cupy.empty(m, 'i')
    _call_cusparse(
        'nnz', x.dtype,
        handle, cusparse.CUSPARSE_DIRECTION_COLUMN, m, n, descr.descriptor,
        x.data.ptr, m, nnz_per_col.data.ptr, nnz.ctypes.data)

    nnz = int(nnz)
    data = cupy.empty(nnz, x.dtype)
    indptr = cupy.empty(n + 1, 'i')
    indices = cupy.empty(nnz, 'i')

    _call_cusparse(
        'dense2csc', x.dtype,
        handle, m, n, descr.descriptor,
        x.data.ptr, m, nnz_per_col.data.ptr,
        data.data.ptr, indices.data.ptr, indptr.data.ptr)
    # Note that a desciptor is recreated
    csc = cupyx.scipy.sparse.csc_matrix((data, indices, indptr), shape=x.shape)
    csc._has_canonical_format = True
    return csc


def dense2csr(x):
    """Converts a dense matrix in CSR format.

    Args:
        x (cupy.ndarray): A matrix to be converted.

    Returns:
        cupyx.scipy.sparse.csr_matrix: A converted matrix.

    """
    assert x.ndim == 2
    x = cupy.asfortranarray(x)
    nnz = numpy.empty((), dtype='i')
    handle = device.get_cusparse_handle()
    m, n = x.shape

    descr = MatDescriptor.create()
    nnz_per_row = cupy.empty(m, 'i')
    _call_cusparse(
        'nnz', x.dtype,
        handle, cusparse.CUSPARSE_DIRECTION_ROW, m, n, descr.descriptor,
        x.data.ptr, m, nnz_per_row.data.ptr, nnz.ctypes.data)

    nnz = int(nnz)
    data = cupy.empty(nnz, x.dtype)
    indptr = cupy.empty(m + 1, 'i')
    indices = cupy.empty(nnz, 'i')

    _call_cusparse(
        'dense2csr', x.dtype,
        handle, m, n, descr.descriptor,
        x.data.ptr, m, nnz_per_row.data.ptr,
        data.data.ptr, indptr.data.ptr, indices.data.ptr)
    # Note that a desciptor is recreated
    csr = cupyx.scipy.sparse.csr_matrix((data, indices, indptr), shape=x.shape)
    csr._has_canonical_format = True
    return csr


def csr2csr_compress(x, tol):
    assert x.dtype.char in 'fdFD'

    handle = device.get_cusparse_handle()
    m, n = x.shape

    nnz_per_row = cupy.empty(m, 'i')
    nnz = _call_cusparse(
        'nnz_compress', x.dtype,
        handle, m, x._descr.descriptor,
        x.data.data.ptr, x.indptr.data.ptr, nnz_per_row.data.ptr, tol)
    data = cupy.zeros(nnz, x.dtype)
    indptr = cupy.empty(m + 1, 'i')
    indices = cupy.zeros(nnz, 'i')
    _call_cusparse(
        'csr2csr_compress', x.dtype,
        handle, m, n, x._descr.descriptor,
        x.data.data.ptr, x.indices.data.ptr, x.indptr.data.ptr,
        x.nnz, nnz_per_row.data.ptr, data.data.ptr, indices.data.ptr,
        indptr.data.ptr, tol)

    return cupyx.scipy.sparse.csr_matrix(
        (data, indices, indptr), shape=x.shape)

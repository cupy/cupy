import functools

import numpy
import platform

import cupy
from cupy_backends.cuda.libs import cusparse
from cupy_backends.cuda.api import runtime
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


_available_cusparse_version = {
    'csrmv': (8000, 11000),
    'csrmvEx': (8000, 11000),  # TODO(anaruse): failure in cuSparse 11.0
    'csrmm': (8000, 11000),
    'csrmm2': (8000, 11000),
    'csrgeam': (8000, 11000),
    'csrgeam2': (9020, None),
    'csrgemm': (8000, 11000),
    'csrgemm2': (8000, None),
    'spmv': ({'Linux': 10200, 'Windows': 11000}, None),
    'spmm': (10301, None),  # accuracy bugs in cuSparse 10.3.0
    'csr2dense': (8000, None),
    'csc2dense': (8000, None),
    'csrsort': (8000, None),
    'cscsort': (8000, None),
    'coosort': (8000, None),
    'coo2csr': (8000, None),
    'csr2coo': (8000, None),
    'csr2csc': (8000, 11000),
    'csc2csr': (8000, 11000),  # the entity is csr2csc
    'csr2cscEx2': (10200, None),
    'csc2csrEx2': (10200, None),  # the entity is csr2cscEx2
    'dense2csc': (8000, None),
    'dense2csr': (8000, None),
    'csr2csr_compress': (8000, None),
}


def _get_version(x):
    if isinstance(x, dict):
        os_name = platform.system()
        if os_name not in x:
            msg = 'No version information specified for the OS: {}'.format(
                os_name)
            raise ValueError(msg)
        return x[os_name]
    return x


@util.memoize()
def check_availability(name):
    if name not in _available_cusparse_version:
        msg = 'No available version information specified for {}'.format(name)
        raise ValueError(msg)
    version_added, version_removed = _available_cusparse_version[name]
    version_added = _get_version(version_added)
    version_removed = _get_version(version_removed)
    cusparse_version = cusparse.getVersion(device.get_cusparse_handle())
    if version_added is not None and cusparse_version < version_added:
        return False
    if version_removed is not None and cusparse_version >= version_removed:
        return False
    return True


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
    if not check_availability('csrmv'):
        raise RuntimeError('csrmv is not available.')

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
    if not check_availability('csrmvEx'):
        raise RuntimeError('csrmvEx is not available.')

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
    if not check_availability('csrmm'):
        raise RuntimeError('csrmm is not available.')

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
    if not check_availability('csrmm2'):
        raise RuntimeError('csrmm2 is not available.')

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
    if not check_availability('csrgeam'):
        raise RuntimeError('csrgeam is not available.')

    if not isinstance(a, cupyx.scipy.sparse.csr_matrix):
        raise TypeError('unsupported type (actual: {})'.format(type(a)))
    if not isinstance(b, cupyx.scipy.sparse.csr_matrix):
        raise TypeError('unsupported type (actual: {})'.format(type(b)))
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


def csrgeam2(a, b, alpha=1, beta=1):
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
    if not check_availability('csrgeam2'):
        raise RuntimeError('csrgeam2 is not available.')

    if not isinstance(a, cupyx.scipy.sparse.csr_matrix):
        raise TypeError('unsupported type (actual: {})'.format(type(a)))
    if not isinstance(b, cupyx.scipy.sparse.csr_matrix):
        raise TypeError('unsupported type (actual: {})'.format(type(b)))
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

    alpha = numpy.array(alpha, a.dtype).ctypes
    beta = numpy.array(beta, a.dtype).ctypes
    c_descr = MatDescriptor.create()
    c_indptr = cupy.empty(m + 1, 'i')

    null_ptr = 0
    buff_size = _call_cusparse(
        'csrgeam2_bufferSizeExt', a.dtype,
        handle, m, n, alpha.data, a._descr.descriptor, a.nnz, a.data.data.ptr,
        a.indptr.data.ptr, a.indices.data.ptr, beta.data, b._descr.descriptor,
        b.nnz, b.data.data.ptr, b.indptr.data.ptr, b.indices.data.ptr,
        c_descr.descriptor, null_ptr, c_indptr.data.ptr, null_ptr)
    buff = cupy.empty(buff_size, numpy.int8)
    cusparse.xcsrgeam2Nnz(
        handle, m, n, a._descr.descriptor, a.nnz, a.indptr.data.ptr,
        a.indices.data.ptr, b._descr.descriptor, b.nnz, b.indptr.data.ptr,
        b.indices.data.ptr, c_descr.descriptor, c_indptr.data.ptr,
        nnz.ctypes.data, buff.data.ptr)
    c_indices = cupy.empty(int(nnz), 'i')
    c_data = cupy.empty(int(nnz), a.dtype)
    _call_cusparse(
        'csrgeam2', a.dtype,
        handle, m, n, alpha.data, a._descr.descriptor, a.nnz, a.data.data.ptr,
        a.indptr.data.ptr, a.indices.data.ptr, beta.data, b._descr.descriptor,
        b.nnz, b.data.data.ptr, b.indptr.data.ptr, b.indices.data.ptr,
        c_descr.descriptor, c_data.data.ptr, c_indptr.data.ptr,
        c_indices.data.ptr, buff.data.ptr)

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
    if not check_availability('csrgemm'):
        raise RuntimeError('csrgemm is not available.')

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


def csrgemm2(a, b, d=None, alpha=1, beta=1):
    """Matrix-matrix product for CSR-matrix.

    math::
       C = alpha * A * B + beta * D

    Args:
        a (cupyx.scipy.sparse.csr_matrix): Sparse matrix A.
        b (cupyx.scipy.sparse.csr_matrix): Sparse matrix B.
        d (cupyx.scipy.sparse.csr_matrix or None): Sparse matrix D.
        alpha (scalar): Coefficient
        beta (scalar): Coefficient

    Returns:
        cupyx.scipy.sparse.csr_matrix

    """
    if not check_availability('csrgemm2'):
        raise RuntimeError('csrgemm2 is not available.')

    assert a.ndim == b.ndim == 2
    if not isinstance(a, cupyx.scipy.sparse.csr_matrix):
        raise TypeError('unsupported type (actual: {})'.format(type(a)))
    if not isinstance(b, cupyx.scipy.sparse.csr_matrix):
        raise TypeError('unsupported type (actual: {})'.format(type(b)))
    assert a.has_canonical_format
    assert b.has_canonical_format
    if a.shape[1] != b.shape[0]:
        raise ValueError('mismatched shape')
    if d is not None:
        assert d.ndim == 2
        if not isinstance(d, cupyx.scipy.sparse.csr_matrix):
            raise TypeError('unsupported type (actual: {})'.format(type(d)))
        assert d.has_canonical_format
        if a.shape[0] != d.shape[0] or b.shape[1] != d.shape[1]:
            raise ValueError('mismatched shape')

    handle = device.get_cusparse_handle()
    m, k = a.shape
    _, n = b.shape

    if d is None:
        a, b = _cast_common_type(a, b)
    else:
        a, b, d = _cast_common_type(a, b, d)

    info = cusparse.createCsrgemm2Info()
    alpha = numpy.array(alpha, a.dtype).ctypes
    null_ptr = 0
    if d is None:
        beta_data = null_ptr
        d_descr = MatDescriptor.create()
        d_nnz = 0
        d_data = null_ptr
        d_indptr = null_ptr
        d_indices = null_ptr
    else:
        beta = numpy.array(beta, a.dtype).ctypes
        beta_data = beta.data
        d_descr = d._descr
        d_nnz = d.nnz
        d_data = d.data.data.ptr
        d_indptr = d.indptr.data.ptr
        d_indices = d.indices.data.ptr

    buff_size = _call_cusparse(
        'csrgemm2_bufferSizeExt', a.dtype,
        handle, m, n, k, alpha.data, a._descr.descriptor, a.nnz,
        a.indptr.data.ptr, a.indices.data.ptr, b._descr.descriptor, b.nnz,
        b.indptr.data.ptr, b.indices.data.ptr, beta_data, d_descr.descriptor,
        d_nnz, d_indptr, d_indices, info)
    buff = cupy.empty(buff_size, numpy.int8)

    c_nnz = numpy.empty((), 'i')
    cusparse.setPointerMode(handle, cusparse.CUSPARSE_POINTER_MODE_HOST)

    c_descr = MatDescriptor.create()
    c_indptr = cupy.empty(m + 1, 'i')
    cusparse.xcsrgemm2Nnz(
        handle, m, n, k, a._descr.descriptor, a.nnz, a.indptr.data.ptr,
        a.indices.data.ptr, b._descr.descriptor, b.nnz, b.indptr.data.ptr,
        b.indices.data.ptr, d_descr.descriptor, d_nnz, d_indptr, d_indices,
        c_descr.descriptor, c_indptr.data.ptr, c_nnz.ctypes.data, info,
        buff.data.ptr)

    c_indices = cupy.empty(int(c_nnz), 'i')
    c_data = cupy.empty(int(c_nnz), a.dtype)
    _call_cusparse(
        'csrgemm2', a.dtype,
        handle, m, n, k, alpha.data, a._descr.descriptor, a.nnz,
        a.data.data.ptr, a.indptr.data.ptr, a.indices.data.ptr,
        b._descr.descriptor, b.nnz, b.data.data.ptr, b.indptr.data.ptr,
        b.indices.data.ptr, beta_data, d_descr.descriptor, d_nnz, d_data,
        d_indptr, d_indices, c_descr.descriptor, c_data.data.ptr,
        c_indptr.data.ptr, c_indices.data.ptr, info, buff.data.ptr)

    c = cupyx.scipy.sparse.csr_matrix(
        (c_data, c_indices, c_indptr), shape=(m, n))
    c._has_canonical_format = True
    cusparse.destroyCsrgemm2Info(info)
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
    if not check_availability('csr2dense'):
        raise RuntimeError('csr2dense is not available.')

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
    if not check_availability('csc2dense'):
        raise RuntimeError('csc2dense is not available.')

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
    if not check_availability('csrsort'):
        raise RuntimeError('csrsort is not available.')

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
    if not check_availability('cscsort'):
        raise RuntimeError('cscsort is not available.')

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


def coosort(x, sort_by='r'):
    """Sorts indices of COO-matrix in place.

    Args:
        x (cupyx.scipy.sparse.coo_matrix): A sparse matrix to sort.
        sort_by (str): Sort the indices by row ('r', default) or column ('c').

    """
    if not check_availability('coosort'):
        raise RuntimeError('coosort is not available.')

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
    if sort_by == 'r':
        cusparse.xcoosortByRow(
            handle, m, n, nnz, x.row.data.ptr, x.col.data.ptr,
            P.data.ptr, buf.data.ptr)
    elif sort_by == 'c':
        cusparse.xcoosortByColumn(
            handle, m, n, nnz, x.row.data.ptr, x.col.data.ptr,
            P.data.ptr, buf.data.ptr)
    else:
        raise ValueError("sort_by must be either 'r' or 'c'")
    _call_cusparse(
        'gthr', x.dtype,
        handle, nnz, data_orig.data.ptr, x.data.data.ptr,
        P.data.ptr, cusparse.CUSPARSE_INDEX_BASE_ZERO)
    if sort_by == 'c':  # coo is sorted by row first
        x._has_canonical_format = False


def coo2csr(x):
    handle = device.get_cusparse_handle()
    m = x.shape[0]
    indptr = cupy.empty(m + 1, 'i')
    cusparse.xcoo2csr(
        handle, x.row.data.ptr, x.nnz, m,
        indptr.data.ptr, cusparse.CUSPARSE_INDEX_BASE_ZERO)
    return cupyx.scipy.sparse.csr.csr_matrix(
        (x.data, x.col, indptr), shape=x.shape)


def coo2csc(x):
    handle = device.get_cusparse_handle()
    n = x.shape[1]
    indptr = cupy.empty(n + 1, 'i')
    cusparse.xcoo2csr(
        handle, x.col.data.ptr, x.nnz, n,
        indptr.data.ptr, cusparse.CUSPARSE_INDEX_BASE_ZERO)
    return cupyx.scipy.sparse.csc.csc_matrix(
        (x.data, x.row, indptr), shape=x.shape)


def csr2coo(x, data, indices):
    """Converts a CSR-matrix to COO format.

    Args:
        x (cupyx.scipy.sparse.csr_matrix): A matrix to be converted.
        data (cupy.ndarray): A data array for converted data.
        indices (cupy.ndarray): An index array for converted data.

    Returns:
        cupyx.scipy.sparse.coo_matrix: A converted matrix.

    """
    if not check_availability('csr2coo'):
        raise RuntimeError('csr2coo is not available.')

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
    if not check_availability('csr2csc'):
        raise RuntimeError('csr2csc is not available.')

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


def csr2cscEx2(x):
    if not check_availability('csr2cscEx2'):
        raise RuntimeError('csr2cscEx2 is not available.')

    handle = device.get_cusparse_handle()
    m, n = x.shape
    nnz = x.nnz
    data = cupy.empty(nnz, x.dtype)
    indices = cupy.empty(nnz, 'i')
    if nnz == 0:
        indptr = cupy.zeros(n + 1, 'i')
    else:
        indptr = cupy.empty(n + 1, 'i')
        x_dtype = _dtype_to_DataType(x.dtype)
        action = cusparse.CUSPARSE_ACTION_NUMERIC
        ibase = cusparse.CUSPARSE_INDEX_BASE_ZERO
        algo = cusparse.CUSPARSE_CSR2CSC_ALG1
        buffer_size = cusparse.csr2cscEx2_bufferSize(
            handle, m, n, nnz, x.data.data.ptr, x.indptr.data.ptr,
            x.indices.data.ptr, data.data.ptr, indptr.data.ptr,
            indices.data.ptr, x_dtype, action, ibase, algo)
        buffer = cupy.empty(buffer_size, numpy.int8)
        cusparse.csr2cscEx2(
            handle, m, n, nnz, x.data.data.ptr, x.indptr.data.ptr,
            x.indices.data.ptr, data.data.ptr, indptr.data.ptr,
            indices.data.ptr, x_dtype, action, ibase, algo, buffer.data.ptr)
    return cupyx.scipy.sparse.csc_matrix(
        (data, indices, indptr), shape=x.shape)


def csc2coo(x, data, indices):
    """Converts a CSC-matrix to COO format.

    Args:
        x (cupyx.scipy.sparse.csc_matrix): A matrix to be converted.
        data (cupy.ndarray): A data array for converted data.
        indices (cupy.ndarray): An index array for converted data.

    Returns:
        cupyx.scipy.sparse.coo_matrix: A converted matrix.

    """
    handle = device.get_cusparse_handle()
    n = x.shape[1]
    nnz = len(x.data)
    col = cupy.empty(nnz, 'i')
    cusparse.xcsr2coo(
        handle, x.indptr.data.ptr, nnz, n, col.data.ptr,
        cusparse.CUSPARSE_INDEX_BASE_ZERO)
    # data and indices did not need to be copied already
    return cupyx.scipy.sparse.coo_matrix(
        (data, (indices, col)), shape=x.shape)


def csc2csr(x):
    if not check_availability('csc2csr'):
        raise RuntimeError('csr2csc is not available.')

    handle = device.get_cusparse_handle()
    m, n = x.shape
    nnz = x.nnz
    data = cupy.empty(nnz, x.dtype)
    indptr = cupy.empty(m + 1, 'i')
    indices = cupy.empty(nnz, 'i')

    _call_cusparse(
        'csr2csc', x.dtype,
        handle, n, m, nnz, x.data.data.ptr,
        x.indptr.data.ptr, x.indices.data.ptr,
        data.data.ptr, indices.data.ptr, indptr.data.ptr,
        cusparse.CUSPARSE_ACTION_NUMERIC,
        cusparse.CUSPARSE_INDEX_BASE_ZERO)
    return cupyx.scipy.sparse.csr_matrix(
        (data, indices, indptr), shape=x.shape)


def csc2csrEx2(x):
    if not check_availability('csc2csrEx2'):
        raise RuntimeError('csc2csrEx2 is not available.')

    handle = device.get_cusparse_handle()
    m, n = x.shape
    nnz = x.nnz
    data = cupy.empty(nnz, x.dtype)
    indices = cupy.empty(nnz, 'i')
    if nnz == 0:
        indptr = cupy.zeros(m + 1, 'i')
    else:
        indptr = cupy.empty(m + 1, 'i')
        x_dtype = _dtype_to_DataType(x.dtype)
        action = cusparse.CUSPARSE_ACTION_NUMERIC
        ibase = cusparse.CUSPARSE_INDEX_BASE_ZERO
        algo = cusparse.CUSPARSE_CSR2CSC_ALG1
        buffer_size = cusparse.csr2cscEx2_bufferSize(
            handle, n, m, nnz, x.data.data.ptr, x.indptr.data.ptr,
            x.indices.data.ptr, data.data.ptr, indptr.data.ptr,
            indices.data.ptr, x_dtype, action, ibase, algo)
        buffer = cupy.empty(buffer_size, numpy.int8)
        cusparse.csr2cscEx2(
            handle, n, m, nnz, x.data.data.ptr, x.indptr.data.ptr,
            x.indices.data.ptr, data.data.ptr, indptr.data.ptr,
            indices.data.ptr, x_dtype, action, ibase, algo, buffer.data.ptr)
    return cupyx.scipy.sparse.csr_matrix(
        (data, indices, indptr), shape=x.shape)


def dense2csc(x):
    """Converts a dense matrix in CSC format.

    Args:
        x (cupy.ndarray): A matrix to be converted.

    Returns:
        cupyx.scipy.sparse.csc_matrix: A converted matrix.

    """
    if not check_availability('dense2csc'):
        raise RuntimeError('dense2csc is not available.')

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
    if not check_availability('dense2csr'):
        raise RuntimeError('dense2csr is not available.')

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
    if not check_availability('csr2csr_compress'):
        raise RuntimeError('csr2csr_compress is not available.')

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


def _dtype_to_IndexType(dtype):
    if dtype == 'uint16':
        return cusparse.CUSPARSE_INDEX_16U
    elif dtype == 'int32':
        return cusparse.CUSPARSE_INDEX_32I
    elif dtype == 'int64':
        return cusparse.CUSPARSE_INDEX_64I
    else:
        raise TypeError


class BaseDescriptor(object):

    def __init__(self, descriptor, get=None, destroyer=None):
        self.desc = descriptor
        self.get = get
        self.destroy = destroyer

    def __del__(self, is_shutting_down=util.is_shutting_down):
        if is_shutting_down():
            return
        if self.destroy is None:
            self.desc = None
        elif self.desc is not None:
            self.destroy(self.desc)
            self.desc = None

    def __getattr__(self, name):
        if self.get is not None:
            return getattr(self.get(self.desc), name)
        raise AttributeError


class SpMatDescriptor(BaseDescriptor):

    @classmethod
    def create(cls, a):
        assert cupyx.scipy.sparse.issparse(a)
        rows, cols = a.shape
        idx_base = cusparse.CUSPARSE_INDEX_BASE_ZERO
        cuda_dtype = _dtype_to_DataType(a.dtype)
        if a.format == 'csr':
            desc = cusparse.createCsr(
                rows, cols, a.nnz, a.indptr.data.ptr, a.indices.data.ptr,
                a.data.data.ptr, _dtype_to_IndexType(a.indptr.dtype),
                _dtype_to_IndexType(a.indices.dtype), idx_base, cuda_dtype)
            get = cusparse.csrGet
        elif a.format == 'coo':
            desc = cusparse.createCoo(
                rows, cols, a.nnz, a.row.data.ptr, a.col.data.ptr,
                a.data.data.ptr, _dtype_to_IndexType(a.row.dtype),
                idx_base, cuda_dtype)
            get = cusparse.cooGet
        else:
            raise ValueError('csr and coo format are supported '
                             '(actual: {}).'.format(a.format))
        destroy = cusparse.destroySpMat
        return SpMatDescriptor(desc, get, destroy)


class DnVecDescriptor(BaseDescriptor):

    @classmethod
    def create(cls, x):
        cuda_dtype = _dtype_to_DataType(x.dtype)
        desc = cusparse.createDnVec(x.size, x.data.ptr, cuda_dtype)
        get = cusparse.dnVecGet
        destroy = cusparse.destroyDnVec
        return DnVecDescriptor(desc, get, destroy)


class DnMatDescriptor(BaseDescriptor):

    @classmethod
    def create(cls, a):
        assert a.ndim == 2
        assert a.flags.f_contiguous
        rows, cols = a.shape
        ld = rows
        cuda_dtype = _dtype_to_DataType(a.dtype)
        desc = cusparse.createDnMat(rows, cols, ld, a.data.ptr, cuda_dtype,
                                    cusparse.CUSPARSE_ORDER_COL)
        get = cusparse.dnMatGet
        destroy = cusparse.destroyDnMat
        return DnMatDescriptor(desc, get, destroy)


def spmv(a, x, y=None, alpha=1, beta=0, transa=False):
    """Multiplication of sparse matrix and dense vector.

    .. math::

        y = \\alpha * op(A) x + \\beta * y

    Args:
        a (cupyx.scipy.sparse.csr_matrix, csc_matrix or coo_matrix):
            Sparse matrix A
        x (cupy.ndarray): Dense vector x
        y (cupy.ndarray or None): Dense vector y
        alpha (scalar): Coefficent
        beta (scalar): Coefficent
        transa (bool): If ``True``, op(A) = transpose of A.

    Returns:
        cupy.ndarray
    """
    if not check_availability('spmv'):
        raise RuntimeError('spmv is not available.')

    if isinstance(a, cupyx.scipy.sparse.csc_matrix):
        aT = a.T
        if not isinstance(aT, cupyx.scipy.sparse.csr_matrix):
            msg = 'aT must be csr_matrix (actual: {})'.format(type(aT))
            raise TypeError(msg)
        a = aT
        transa = not transa
    if not (isinstance(a, cupyx.scipy.sparse.csr_matrix) or
            isinstance(a, cupyx.scipy.sparse.coo_matrix)):
        raise TypeError('unsupported type (actual: {})'.format(type(a)))
    a_shape = a.shape if not transa else a.shape[::-1]
    if a_shape[1] != len(x):
        raise ValueError('dimension mismatch')
    assert a.has_canonical_format

    m, n = a_shape
    a, x, y = _cast_common_type(a, x, y)
    if y is None:
        y = cupy.zeros(m, a.dtype)
    elif len(y) != m:
        raise ValueError('dimension mismatch')
    if a.nnz == 0:
        y[...] = 0
        return y

    desc_a = SpMatDescriptor.create(a)
    desc_x = DnVecDescriptor.create(x)
    desc_y = DnVecDescriptor.create(y)

    handle = device.get_cusparse_handle()
    op_a = _transpose_flag(transa)
    alpha = numpy.array(alpha, a.dtype).ctypes
    beta = numpy.array(beta, a.dtype).ctypes
    cuda_dtype = _dtype_to_DataType(a.dtype)
    alg = cusparse.CUSPARSE_MV_ALG_DEFAULT
    buff_size = cusparse.spMV_bufferSize(handle, op_a, alpha.data,
                                         desc_a.desc, desc_x.desc, beta.data,
                                         desc_y.desc, cuda_dtype, alg)
    buff = cupy.empty(buff_size, cupy.int8)
    cusparse.spMV(handle, op_a, alpha.data, desc_a.desc, desc_x.desc,
                  beta.data, desc_y.desc, cuda_dtype, alg, buff.data.ptr)

    return y


def spmm(a, b, c=None, alpha=1, beta=0, transa=False, transb=False):
    """Multiplication of sparse matrix and dense matrix.

    .. math::

        C = \\alpha * op(A) op(B) + \\beta * C

    Args:
        a (cupyx.scipy.sparse.csr_matrix, csc_matrix or coo_matrix):
            Sparse matrix A
        b (cupy.ndarray): Dense matrix B
        c (cupy.ndarray or None): Dense matrix C
        alpha (scalar): Coefficent
        beta (scalar): Coefficent
        transa (bool): If ``True``, op(A) = transpose of A.
        transb (bool): If ``True``, op(B) = transpose of B.

    Returns:
        cupy.ndarray
    """
    if not check_availability('spmm'):
        raise RuntimeError('spmm is not available.')

    assert a.ndim == b.ndim == 2
    assert b.flags.f_contiguous
    assert c is None or c.flags.f_contiguous

    if isinstance(a, cupyx.scipy.sparse.csc_matrix):
        aT = a.T
        if not isinstance(aT, cupyx.scipy.sparse.csr_matrix):
            msg = 'aT must be csr_matrix (actual: {})'.format(type(aT))
            raise TypeError(msg)
        a = aT
        transa = not transa
    if not (isinstance(a, cupyx.scipy.sparse.csr_matrix) or
            isinstance(a, cupyx.scipy.sparse.coo_matrix)):
        raise TypeError('unsupported type (actual: {})'.format(type(a)))
    a_shape = a.shape if not transa else a.shape[::-1]
    b_shape = b.shape if not transb else b.shape[::-1]
    if a_shape[1] != b_shape[0]:
        raise ValueError('dimension mismatch')
    assert a.has_canonical_format

    m, k = a_shape
    _, n = b_shape
    a, b, c = _cast_common_type(a, b, c)
    if c is None:
        c = cupy.zeros((m, n), a.dtype, 'F')
    elif c.shape[0] != m or c.shape[1] != n:
        raise ValueError('dimension mismatch')
    if a.nnz == 0:
        c[...] = 0
        return c

    desc_a = SpMatDescriptor.create(a)
    desc_b = DnMatDescriptor.create(b)
    desc_c = DnMatDescriptor.create(c)

    handle = device.get_cusparse_handle()
    op_a = _transpose_flag(transa)
    op_b = _transpose_flag(transb)
    alpha = numpy.array(alpha, a.dtype).ctypes
    beta = numpy.array(beta, a.dtype).ctypes
    cuda_dtype = _dtype_to_DataType(a.dtype)
    alg = cusparse.CUSPARSE_MM_ALG_DEFAULT
    buff_size = cusparse.spMM_bufferSize(handle, op_a, op_b, alpha.data,
                                         desc_a.desc, desc_b.desc, beta.data,
                                         desc_c.desc, cuda_dtype, alg)
    buff = cupy.empty(buff_size, cupy.int8)
    buff_size = cusparse.spMM(handle, op_a, op_b, alpha.data, desc_a.desc,
                              desc_b.desc, beta.data, desc_c.desc,
                              cuda_dtype, alg, buff.data.ptr)

    return c

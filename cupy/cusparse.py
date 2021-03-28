import functools as _functools

import numpy as _numpy
import platform as _platform

import cupy as _cupy
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupy._core import _dtype
from cupy.cuda import device as _device
from cupy import _util
import cupyx.scipy.sparse


class MatDescriptor(object):

    def __init__(self, descriptor):
        self.descriptor = descriptor

    @classmethod
    def create(cls):
        descr = _cusparse.createMatDescr()
        return MatDescriptor(descr)

    def __reduce__(self):
        return self.create, ()

    def __del__(self, is_shutting_down=_util.is_shutting_down):
        if is_shutting_down():
            return
        if self.descriptor:
            _cusparse.destroyMatDescr(self.descriptor)
            self.descriptor = None

    def set_mat_type(self, typ):
        _cusparse.setMatType(self.descriptor, typ)

    def set_mat_index_base(self, base):
        _cusparse.setMatIndexBase(self.descriptor, base)

    def set_mat_fill_mode(self, fill_mode):
        _cusparse.setMatFillMode(self.descriptor, fill_mode)

    def set_mat_diag_type(self, diag_type):
        _cusparse.setMatDiagType(self.descriptor, diag_type)


def _cast_common_type(*xs):
    dtypes = [x.dtype for x in xs if x is not None]
    dtype = _functools.reduce(_numpy.promote_types, dtypes)
    return [x.astype(dtype) if x is not None and x.dtype != dtype else x
            for x in xs]


def _transpose_flag(trans):
    if trans:
        return _cusparse.CUSPARSE_OPERATION_TRANSPOSE
    else:
        return _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE


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
    f = getattr(_cusparse, prefix + name)
    return f(*args)


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
    'csrsm2': (9020, None),
    'csrilu02': (8000, None),
    'denseToSparse': (11300, None),
    'sparseToDense': (11300, None),
}


def _get_version(x):
    if isinstance(x, dict):
        os_name = _platform.system()
        if os_name not in x:
            msg = 'No version information specified for the OS: {}'.format(
                os_name)
            raise ValueError(msg)
        return x[os_name]
    return x


@_util.memoize()
def check_availability(name):
    if name not in _available_cusparse_version:
        msg = 'No available version information specified for {}'.format(name)
        raise ValueError(msg)
    version_added, version_removed = _available_cusparse_version[name]
    version_added = _get_version(version_added)
    version_removed = _get_version(version_removed)
    cusparse_version = _cusparse.getVersion(_device.get_cusparse_handle())
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

    handle = _device.get_cusparse_handle()
    m, n = a_shape
    a, x, y = _cast_common_type(a, x, y)
    dtype = a.dtype
    if y is None:
        y = _cupy.zeros(m, dtype)
    alpha = _numpy.array(alpha, dtype).ctypes
    beta = _numpy.array(beta, dtype).ctypes

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

    handle = _device.get_cusparse_handle()
    m, n = a.shape

    a, x, y = _cast_common_type(a, x, y)
    dtype = a.dtype
    if y is None:
        y = _cupy.zeros(m, dtype)

    datatype = _dtype.to_cuda_dtype(dtype)
    algmode = _cusparse.CUSPARSE_ALG_MERGE_PATH if \
        merge_path else _cusparse.CUSPARSE_ALG_NAIVE
    transa_flag = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE

    alpha = _numpy.array(alpha, dtype).ctypes
    beta = _numpy.array(beta, dtype).ctypes

    assert csrmvExIsAligned(a, x, y)

    bufferSize = _cusparse.csrmvEx_bufferSize(
        handle, algmode, transa_flag,
        a.shape[0], a.shape[1], a.nnz, alpha.data, datatype,
        a._descr.descriptor, a.data.data.ptr, datatype,
        a.indptr.data.ptr, a.indices.data.ptr,
        x.data.ptr, datatype, beta.data, datatype,
        y.data.ptr, datatype, datatype)

    buf = _cupy.empty(bufferSize, 'b')
    assert buf.data.ptr % 128 == 0

    _cusparse.csrmvEx(
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

    handle = _device.get_cusparse_handle()
    m, k = a_shape
    n = b.shape[1]

    a, b, c = _cast_common_type(a, b, c)
    if c is None:
        c = _cupy.zeros((m, n), a.dtype, 'F')

    ldb = k
    ldc = m

    alpha = _numpy.array(alpha, a.dtype).ctypes
    beta = _numpy.array(beta, a.dtype).ctypes
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

    handle = _device.get_cusparse_handle()
    m, k = a_shape
    n = b_shape[1]

    a, b, c = _cast_common_type(a, b, c)
    if c is None:
        c = _cupy.zeros((m, n), a.dtype, 'F')

    ldb = b.shape[0]
    ldc = c.shape[0]
    op_a = _transpose_flag(transa)
    op_b = _transpose_flag(transb)
    alpha = _numpy.array(alpha, a.dtype).ctypes
    beta = _numpy.array(beta, a.dtype).ctypes
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

    handle = _device.get_cusparse_handle()
    m, n = a.shape
    a, b = _cast_common_type(a, b)
    nnz = _numpy.empty((), 'i')
    _cusparse.setPointerMode(
        handle, _cusparse.CUSPARSE_POINTER_MODE_HOST)

    c_descr = MatDescriptor.create()
    c_indptr = _cupy.empty(m + 1, 'i')

    _cusparse.xcsrgeamNnz(
        handle, m, n,
        a._descr.descriptor, a.nnz, a.indptr.data.ptr, a.indices.data.ptr,
        b._descr.descriptor, b.nnz, b.indptr.data.ptr, b.indices.data.ptr,
        c_descr.descriptor, c_indptr.data.ptr, nnz.ctypes.data)

    c_indices = _cupy.empty(int(nnz), 'i')
    c_data = _cupy.empty(int(nnz), a.dtype)
    alpha = _numpy.array(alpha, a.dtype).ctypes
    beta = _numpy.array(beta, a.dtype).ctypes
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

    handle = _device.get_cusparse_handle()
    m, n = a.shape
    a, b = _cast_common_type(a, b)
    nnz = _numpy.empty((), 'i')
    _cusparse.setPointerMode(
        handle, _cusparse.CUSPARSE_POINTER_MODE_HOST)

    alpha = _numpy.array(alpha, a.dtype).ctypes
    beta = _numpy.array(beta, a.dtype).ctypes
    c_descr = MatDescriptor.create()
    c_indptr = _cupy.empty(m + 1, 'i')

    null_ptr = 0
    buff_size = _call_cusparse(
        'csrgeam2_bufferSizeExt', a.dtype,
        handle, m, n, alpha.data, a._descr.descriptor, a.nnz, a.data.data.ptr,
        a.indptr.data.ptr, a.indices.data.ptr, beta.data, b._descr.descriptor,
        b.nnz, b.data.data.ptr, b.indptr.data.ptr, b.indices.data.ptr,
        c_descr.descriptor, null_ptr, c_indptr.data.ptr, null_ptr)
    buff = _cupy.empty(buff_size, _numpy.int8)
    _cusparse.xcsrgeam2Nnz(
        handle, m, n, a._descr.descriptor, a.nnz, a.indptr.data.ptr,
        a.indices.data.ptr, b._descr.descriptor, b.nnz, b.indptr.data.ptr,
        b.indices.data.ptr, c_descr.descriptor, c_indptr.data.ptr,
        nnz.ctypes.data, buff.data.ptr)
    c_indices = _cupy.empty(int(nnz), 'i')
    c_data = _cupy.empty(int(nnz), a.dtype)
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

    handle = _device.get_cusparse_handle()
    m, k = a_shape
    n = b_shape[1]

    a, b = _cast_common_type(a, b)

    if a.nnz == 0 or b.nnz == 0:
        return cupyx.scipy.sparse.csr_matrix((m, n), dtype=a.dtype)

    op_a = _transpose_flag(transa)
    op_b = _transpose_flag(transb)

    nnz = _numpy.empty((), 'i')
    _cusparse.setPointerMode(
        handle, _cusparse.CUSPARSE_POINTER_MODE_HOST)

    c_descr = MatDescriptor.create()
    c_indptr = _cupy.empty(m + 1, 'i')

    _cusparse.xcsrgemmNnz(
        handle, op_a, op_b, m, n, k, a._descr.descriptor, a.nnz,
        a.indptr.data.ptr, a.indices.data.ptr, b._descr.descriptor, b.nnz,
        b.indptr.data.ptr, b.indices.data.ptr, c_descr.descriptor,
        c_indptr.data.ptr, nnz.ctypes.data)

    c_indices = _cupy.empty(int(nnz), 'i')
    c_data = _cupy.empty(int(nnz), a.dtype)
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

    handle = _device.get_cusparse_handle()
    m, k = a.shape
    _, n = b.shape

    if d is None:
        a, b = _cast_common_type(a, b)
    else:
        a, b, d = _cast_common_type(a, b, d)

    info = _cusparse.createCsrgemm2Info()
    alpha = _numpy.array(alpha, a.dtype).ctypes
    null_ptr = 0
    if d is None:
        beta_data = null_ptr
        d_descr = MatDescriptor.create()
        d_nnz = 0
        d_data = null_ptr
        d_indptr = null_ptr
        d_indices = null_ptr
    else:
        beta = _numpy.array(beta, a.dtype).ctypes
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
    buff = _cupy.empty(buff_size, _numpy.int8)

    c_nnz = _numpy.empty((), 'i')
    _cusparse.setPointerMode(handle, _cusparse.CUSPARSE_POINTER_MODE_HOST)

    c_descr = MatDescriptor.create()
    c_indptr = _cupy.empty(m + 1, 'i')
    _cusparse.xcsrgemm2Nnz(
        handle, m, n, k, a._descr.descriptor, a.nnz, a.indptr.data.ptr,
        a.indices.data.ptr, b._descr.descriptor, b.nnz, b.indptr.data.ptr,
        b.indices.data.ptr, d_descr.descriptor, d_nnz, d_indptr, d_indices,
        c_descr.descriptor, c_indptr.data.ptr, c_nnz.ctypes.data, info,
        buff.data.ptr)

    c_indices = _cupy.empty(int(c_nnz), 'i')
    c_data = _cupy.empty(int(c_nnz), a.dtype)
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
    _cusparse.destroyCsrgemm2Info(info)
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
        out = _cupy.empty(x.shape, dtype=dtype, order='F')
    else:
        assert out.flags.f_contiguous

    handle = _device.get_cusparse_handle()
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
        out = _cupy.empty(x.shape, dtype=dtype, order='F')
    else:
        assert out.flags.f_contiguous

    handle = _device.get_cusparse_handle()
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
    handle = _device.get_cusparse_handle()
    m, n = x.shape

    buffer_size = _cusparse.xcsrsort_bufferSizeExt(
        handle, m, n, nnz, x.indptr.data.ptr,
        x.indices.data.ptr)
    buf = _cupy.empty(buffer_size, 'b')
    P = _cupy.empty(nnz, 'i')
    data_orig = x.data.copy()
    _cusparse.createIdentityPermutation(handle, nnz, P.data.ptr)
    _cusparse.xcsrsort(
        handle, m, n, nnz, x._descr.descriptor, x.indptr.data.ptr,
        x.indices.data.ptr, P.data.ptr, buf.data.ptr)
    _call_cusparse(
        'gthr', x.dtype,
        handle, nnz, data_orig.data.ptr, x.data.data.ptr,
        P.data.ptr, _cusparse.CUSPARSE_INDEX_BASE_ZERO)


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
    handle = _device.get_cusparse_handle()
    m, n = x.shape

    buffer_size = _cusparse.xcscsort_bufferSizeExt(
        handle, m, n, nnz, x.indptr.data.ptr,
        x.indices.data.ptr)
    buf = _cupy.empty(buffer_size, 'b')
    P = _cupy.empty(nnz, 'i')
    data_orig = x.data.copy()
    _cusparse.createIdentityPermutation(handle, nnz, P.data.ptr)
    _cusparse.xcscsort(
        handle, m, n, nnz, x._descr.descriptor, x.indptr.data.ptr,
        x.indices.data.ptr, P.data.ptr, buf.data.ptr)
    _call_cusparse(
        'gthr', x.dtype,
        handle, nnz, data_orig.data.ptr, x.data.data.ptr,
        P.data.ptr, _cusparse.CUSPARSE_INDEX_BASE_ZERO)


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
    handle = _device.get_cusparse_handle()
    m, n = x.shape

    buffer_size = _cusparse.xcoosort_bufferSizeExt(
        handle, m, n, nnz, x.row.data.ptr, x.col.data.ptr)
    buf = _cupy.empty(buffer_size, 'b')
    P = _cupy.empty(nnz, 'i')
    data_orig = x.data.copy()
    _cusparse.createIdentityPermutation(handle, nnz, P.data.ptr)
    if sort_by == 'r':
        _cusparse.xcoosortByRow(
            handle, m, n, nnz, x.row.data.ptr, x.col.data.ptr,
            P.data.ptr, buf.data.ptr)
    elif sort_by == 'c':
        _cusparse.xcoosortByColumn(
            handle, m, n, nnz, x.row.data.ptr, x.col.data.ptr,
            P.data.ptr, buf.data.ptr)
    else:
        raise ValueError("sort_by must be either 'r' or 'c'")
    _call_cusparse(
        'gthr', x.dtype,
        handle, nnz, data_orig.data.ptr, x.data.data.ptr,
        P.data.ptr, _cusparse.CUSPARSE_INDEX_BASE_ZERO)
    if sort_by == 'c':  # coo is sorted by row first
        x._has_canonical_format = False


def coo2csr(x):
    handle = _device.get_cusparse_handle()
    m = x.shape[0]
    nnz = x.nnz
    if nnz == 0:
        indptr = _cupy.zeros(m + 1, 'i')
    else:
        indptr = _cupy.empty(m + 1, 'i')
        _cusparse.xcoo2csr(
            handle, x.row.data.ptr, nnz, m,
            indptr.data.ptr, _cusparse.CUSPARSE_INDEX_BASE_ZERO)
    return cupyx.scipy.sparse.csr.csr_matrix(
        (x.data, x.col, indptr), shape=x.shape)


def coo2csc(x):
    handle = _device.get_cusparse_handle()
    n = x.shape[1]
    nnz = x.nnz
    if nnz == 0:
        indptr = _cupy.zeros(n + 1, 'i')
    else:
        indptr = _cupy.empty(n + 1, 'i')
        _cusparse.xcoo2csr(
            handle, x.col.data.ptr, nnz, n,
            indptr.data.ptr, _cusparse.CUSPARSE_INDEX_BASE_ZERO)
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

    handle = _device.get_cusparse_handle()
    m = x.shape[0]
    nnz = x.nnz
    row = _cupy.empty(nnz, 'i')
    _cusparse.xcsr2coo(
        handle, x.indptr.data.ptr, nnz, m, row.data.ptr,
        _cusparse.CUSPARSE_INDEX_BASE_ZERO)
    # data and indices did not need to be copied already
    return cupyx.scipy.sparse.coo_matrix(
        (data, (row, indices)), shape=x.shape)


def csr2csc(x):
    if not check_availability('csr2csc'):
        raise RuntimeError('csr2csc is not available.')

    handle = _device.get_cusparse_handle()
    m, n = x.shape
    nnz = x.nnz
    data = _cupy.empty(nnz, x.dtype)
    indices = _cupy.empty(nnz, 'i')
    if nnz == 0:
        indptr = _cupy.zeros(n + 1, 'i')
    else:
        indptr = _cupy.empty(n + 1, 'i')
        _call_cusparse(
            'csr2csc', x.dtype,
            handle, m, n, nnz, x.data.data.ptr,
            x.indptr.data.ptr, x.indices.data.ptr,
            data.data.ptr, indices.data.ptr, indptr.data.ptr,
            _cusparse.CUSPARSE_ACTION_NUMERIC,
            _cusparse.CUSPARSE_INDEX_BASE_ZERO)
    return cupyx.scipy.sparse.csc_matrix(
        (data, indices, indptr), shape=x.shape)


def csr2cscEx2(x):
    if not check_availability('csr2cscEx2'):
        raise RuntimeError('csr2cscEx2 is not available.')

    handle = _device.get_cusparse_handle()
    m, n = x.shape
    nnz = x.nnz
    data = _cupy.empty(nnz, x.dtype)
    indices = _cupy.empty(nnz, 'i')
    if nnz == 0:
        indptr = _cupy.zeros(n + 1, 'i')
    else:
        indptr = _cupy.empty(n + 1, 'i')
        x_dtype = _dtype.to_cuda_dtype(x.dtype)
        action = _cusparse.CUSPARSE_ACTION_NUMERIC
        ibase = _cusparse.CUSPARSE_INDEX_BASE_ZERO
        algo = _cusparse.CUSPARSE_CSR2CSC_ALG1
        buffer_size = _cusparse.csr2cscEx2_bufferSize(
            handle, m, n, nnz, x.data.data.ptr, x.indptr.data.ptr,
            x.indices.data.ptr, data.data.ptr, indptr.data.ptr,
            indices.data.ptr, x_dtype, action, ibase, algo)
        buffer = _cupy.empty(buffer_size, _numpy.int8)
        _cusparse.csr2cscEx2(
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
    handle = _device.get_cusparse_handle()
    n = x.shape[1]
    nnz = x.nnz
    col = _cupy.empty(nnz, 'i')
    _cusparse.xcsr2coo(
        handle, x.indptr.data.ptr, nnz, n, col.data.ptr,
        _cusparse.CUSPARSE_INDEX_BASE_ZERO)
    # data and indices did not need to be copied already
    return cupyx.scipy.sparse.coo_matrix(
        (data, (indices, col)), shape=x.shape)


def csc2csr(x):
    if not check_availability('csc2csr'):
        raise RuntimeError('csr2csc is not available.')

    handle = _device.get_cusparse_handle()
    m, n = x.shape
    nnz = x.nnz
    data = _cupy.empty(nnz, x.dtype)
    indices = _cupy.empty(nnz, 'i')
    if nnz == 0:
        indptr = _cupy.zeros(m + 1, 'i')
    else:
        indptr = _cupy.empty(m + 1, 'i')
        _call_cusparse(
            'csr2csc', x.dtype,
            handle, n, m, nnz, x.data.data.ptr,
            x.indptr.data.ptr, x.indices.data.ptr,
            data.data.ptr, indices.data.ptr, indptr.data.ptr,
            _cusparse.CUSPARSE_ACTION_NUMERIC,
            _cusparse.CUSPARSE_INDEX_BASE_ZERO)
    return cupyx.scipy.sparse.csr_matrix(
        (data, indices, indptr), shape=x.shape)


def csc2csrEx2(x):
    if not check_availability('csc2csrEx2'):
        raise RuntimeError('csc2csrEx2 is not available.')

    handle = _device.get_cusparse_handle()
    m, n = x.shape
    nnz = x.nnz
    data = _cupy.empty(nnz, x.dtype)
    indices = _cupy.empty(nnz, 'i')
    if nnz == 0:
        indptr = _cupy.zeros(m + 1, 'i')
    else:
        indptr = _cupy.empty(m + 1, 'i')
        x_dtype = _dtype.to_cuda_dtype(x.dtype)
        action = _cusparse.CUSPARSE_ACTION_NUMERIC
        ibase = _cusparse.CUSPARSE_INDEX_BASE_ZERO
        algo = _cusparse.CUSPARSE_CSR2CSC_ALG1
        buffer_size = _cusparse.csr2cscEx2_bufferSize(
            handle, n, m, nnz, x.data.data.ptr, x.indptr.data.ptr,
            x.indices.data.ptr, data.data.ptr, indptr.data.ptr,
            indices.data.ptr, x_dtype, action, ibase, algo)
        buffer = _cupy.empty(buffer_size, _numpy.int8)
        _cusparse.csr2cscEx2(
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
    x = _cupy.asfortranarray(x)
    nnz = _numpy.empty((), dtype='i')
    handle = _device.get_cusparse_handle()
    m, n = x.shape

    descr = MatDescriptor.create()
    nnz_per_col = _cupy.empty(m, 'i')
    _call_cusparse(
        'nnz', x.dtype,
        handle, _cusparse.CUSPARSE_DIRECTION_COLUMN, m, n, descr.descriptor,
        x.data.ptr, m, nnz_per_col.data.ptr, nnz.ctypes.data)

    nnz = int(nnz)
    data = _cupy.empty(nnz, x.dtype)
    indptr = _cupy.empty(n + 1, 'i')
    indices = _cupy.empty(nnz, 'i')

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
    x = _cupy.asfortranarray(x)
    nnz = _numpy.empty((), dtype='i')
    handle = _device.get_cusparse_handle()
    m, n = x.shape

    descr = MatDescriptor.create()
    nnz_per_row = _cupy.empty(m, 'i')
    _call_cusparse(
        'nnz', x.dtype,
        handle, _cusparse.CUSPARSE_DIRECTION_ROW, m, n, descr.descriptor,
        x.data.ptr, m, nnz_per_row.data.ptr, nnz.ctypes.data)

    nnz = int(nnz)
    data = _cupy.empty(nnz, x.dtype)
    indptr = _cupy.empty(m + 1, 'i')
    indices = _cupy.empty(nnz, 'i')

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

    handle = _device.get_cusparse_handle()
    m, n = x.shape

    nnz_per_row = _cupy.empty(m, 'i')
    nnz = _call_cusparse(
        'nnz_compress', x.dtype,
        handle, m, x._descr.descriptor,
        x.data.data.ptr, x.indptr.data.ptr, nnz_per_row.data.ptr, tol)
    data = _cupy.zeros(nnz, x.dtype)
    indptr = _cupy.empty(m + 1, 'i')
    indices = _cupy.zeros(nnz, 'i')
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
        return _cusparse.CUSPARSE_INDEX_16U
    elif dtype == 'int32':
        return _cusparse.CUSPARSE_INDEX_32I
    elif dtype == 'int64':
        return _cusparse.CUSPARSE_INDEX_64I
    else:
        raise TypeError


class BaseDescriptor(object):

    def __init__(self, descriptor, get=None, destroyer=None):
        self.desc = descriptor
        self.get = get
        self.destroy = destroyer

    def __del__(self, is_shutting_down=_util.is_shutting_down):
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
        idx_base = _cusparse.CUSPARSE_INDEX_BASE_ZERO
        cuda_dtype = _dtype.to_cuda_dtype(a.dtype)
        if a.format == 'csr':
            desc = _cusparse.createCsr(
                rows, cols, a.nnz, a.indptr.data.ptr, a.indices.data.ptr,
                a.data.data.ptr, _dtype_to_IndexType(a.indptr.dtype),
                _dtype_to_IndexType(a.indices.dtype), idx_base, cuda_dtype)
            get = _cusparse.csrGet
        elif a.format == 'coo':
            desc = _cusparse.createCoo(
                rows, cols, a.nnz, a.row.data.ptr, a.col.data.ptr,
                a.data.data.ptr, _dtype_to_IndexType(a.row.dtype),
                idx_base, cuda_dtype)
            get = _cusparse.cooGet
        elif a.format == 'csc':
            desc = _cusparse.createCsc(
                rows, cols, a.nnz, a.indptr.data.ptr, a.indices.data.ptr,
                a.data.data.ptr, _dtype_to_IndexType(a.indptr.dtype),
                _dtype_to_IndexType(a.indices.dtype), idx_base, cuda_dtype)
            get = None
        else:
            raise ValueError('csr, csc and coo format are supported '
                             '(actual: {}).'.format(a.format))
        destroy = _cusparse.destroySpMat
        return SpMatDescriptor(desc, get, destroy)


class DnVecDescriptor(BaseDescriptor):

    @classmethod
    def create(cls, x):
        cuda_dtype = _dtype.to_cuda_dtype(x.dtype)
        desc = _cusparse.createDnVec(x.size, x.data.ptr, cuda_dtype)
        get = _cusparse.dnVecGet
        destroy = _cusparse.destroyDnVec
        return DnVecDescriptor(desc, get, destroy)


class DnMatDescriptor(BaseDescriptor):

    @classmethod
    def create(cls, a):
        assert a.ndim == 2
        assert a.flags.f_contiguous
        rows, cols = a.shape
        ld = rows
        cuda_dtype = _dtype.to_cuda_dtype(a.dtype)
        desc = _cusparse.createDnMat(rows, cols, ld, a.data.ptr, cuda_dtype,
                                     _cusparse.CUSPARSE_ORDER_COL)
        get = _cusparse.dnMatGet
        destroy = _cusparse.destroyDnMat
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
    if not isinstance(a, (cupyx.scipy.sparse.csr_matrix,
                          cupyx.scipy.sparse.coo_matrix)):
        raise TypeError('unsupported type (actual: {})'.format(type(a)))
    a_shape = a.shape if not transa else a.shape[::-1]
    if a_shape[1] != len(x):
        raise ValueError('dimension mismatch')
    assert a.has_canonical_format

    m, n = a_shape
    a, x, y = _cast_common_type(a, x, y)
    if y is None:
        y = _cupy.zeros(m, a.dtype)
    elif len(y) != m:
        raise ValueError('dimension mismatch')
    if a.nnz == 0:
        y[...] = 0
        return y

    desc_a = SpMatDescriptor.create(a)
    desc_x = DnVecDescriptor.create(x)
    desc_y = DnVecDescriptor.create(y)

    handle = _device.get_cusparse_handle()
    op_a = _transpose_flag(transa)
    alpha = _numpy.array(alpha, a.dtype).ctypes
    beta = _numpy.array(beta, a.dtype).ctypes
    cuda_dtype = _dtype.to_cuda_dtype(a.dtype)
    alg = _cusparse.CUSPARSE_MV_ALG_DEFAULT
    buff_size = _cusparse.spMV_bufferSize(handle, op_a, alpha.data,
                                          desc_a.desc, desc_x.desc, beta.data,
                                          desc_y.desc, cuda_dtype, alg)
    buff = _cupy.empty(buff_size, _cupy.int8)
    _cusparse.spMV(handle, op_a, alpha.data, desc_a.desc, desc_x.desc,
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
    if not isinstance(a, (cupyx.scipy.sparse.csr_matrix,
                          cupyx.scipy.sparse.coo_matrix)):
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
        c = _cupy.zeros((m, n), a.dtype, 'F')
    elif c.shape[0] != m or c.shape[1] != n:
        raise ValueError('dimension mismatch')
    if a.nnz == 0:
        c[...] = 0
        return c

    desc_a = SpMatDescriptor.create(a)
    desc_b = DnMatDescriptor.create(b)
    desc_c = DnMatDescriptor.create(c)

    handle = _device.get_cusparse_handle()
    op_a = _transpose_flag(transa)
    op_b = _transpose_flag(transb)
    alpha = _numpy.array(alpha, a.dtype).ctypes
    beta = _numpy.array(beta, a.dtype).ctypes
    cuda_dtype = _dtype.to_cuda_dtype(a.dtype)
    alg = _cusparse.CUSPARSE_MM_ALG_DEFAULT
    buff_size = _cusparse.spMM_bufferSize(handle, op_a, op_b, alpha.data,
                                          desc_a.desc, desc_b.desc, beta.data,
                                          desc_c.desc, cuda_dtype, alg)
    buff = _cupy.empty(buff_size, _cupy.int8)
    buff_size = _cusparse.spMM(handle, op_a, op_b, alpha.data, desc_a.desc,
                               desc_b.desc, beta.data, desc_c.desc,
                               cuda_dtype, alg, buff.data.ptr)

    return c


def csrsm2(a, b, alpha=1.0, lower=True, unit_diag=False, transa=False,
           blocking=True, level_info=False):
    """Solves a sparse triangular linear system op(a) * x = alpha * b.

    Args:
        a (cupyx.scipy.sparse.csr_matrix or cupyx.scipy.sparse.csc_matrix):
            Sparse matrix with dimension ``(M, M)``.
        b (cupy.ndarray): Dense vector or matrix with dimension ``(M)`` or
            ``(M, K)``.
        alpha (float or complex): Coefficent.
        lower (bool):
            True: ``a`` is lower triangle matrix.
            False: ``a`` is upper triangle matrix.
        unit_diag (bool):
            True: diagonal part of ``a`` has unit elements.
            False: diagonal part of ``a`` has non-unit elements.
        transa (bool or str): True, False, 'N', 'T' or 'H'.
            'N' or False: op(a) == ``a``.
            'T' or True: op(a) == ``a.T``.
            'H': op(a) == ``a.conj().T``.
        blocking (bool):
            True: blocking algorithm is used.
            False: non-blocking algorithm is used.
        level_info (bool):
            True: solves it with level infromation.
            False: solves it without level information.

    Note: ``b`` will be overwritten.
    """
    if not check_availability('csrsm2'):
        raise RuntimeError('csrsm2 is not available.')

    if not (cupyx.scipy.sparse.isspmatrix_csr(a) or
            cupyx.scipy.sparse.isspmatrix_csc(a)):
        raise ValueError('a must be CSR or CSC sparse matrix')
    if not isinstance(b, _cupy.ndarray):
        raise ValueError('b must be cupy.ndarray')
    if b.ndim not in (1, 2):
        raise ValueError('b.ndim must be 1 or 2')
    if not (a.shape[0] == a.shape[1] == b.shape[0]):
        raise ValueError('invalid shape')
    if a.dtype != b.dtype:
        raise TypeError('dtype mismatch')

    if lower is True:
        fill_mode = _cusparse.CUSPARSE_FILL_MODE_LOWER
    elif lower is False:
        fill_mode = _cusparse.CUSPARSE_FILL_MODE_UPPER
    else:
        raise ValueError('Unknown lower (actual: {})'.format(lower))

    if unit_diag is False:
        diag_type = _cusparse.CUSPARSE_DIAG_TYPE_NON_UNIT
    elif unit_diag is True:
        diag_type = _cusparse.CUSPARSE_DIAG_TYPE_UNIT
    else:
        raise ValueError('Unknown unit_diag (actual: {})'.format(unit_diag))

    if blocking is False:
        algo = 0
    elif blocking is True:
        algo = 1
    else:
        raise ValueError('Unknown blocking (actual: {})'.format(blocking))

    if level_info is False:
        policy = _cusparse.CUSPARSE_SOLVE_POLICY_NO_LEVEL
    elif level_info is True:
        policy = _cusparse.CUSPARSE_SOLVE_POLICY_USE_LEVEL
    else:
        raise ValueError('Unknown level_info (actual: {})'.format(level_info))

    dtype = a.dtype
    if dtype.char == 'f':
        t = 's'
    elif dtype.char == 'd':
        t = 'd'
    elif dtype.char == 'F':
        t = 'c'
    elif dtype.char == 'D':
        t = 'z'
    else:
        raise TypeError('Invalid dtype (actual: {})'.format(dtype))
    helper = getattr(_cusparse, t + 'csrsm2_bufferSizeExt')
    analysis = getattr(_cusparse, t + 'csrsm2_analysis')
    solve = getattr(_cusparse, t + 'csrsm2_solve')

    if transa is False or transa == 'N':
        transa = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
    elif transa is True or transa == 'T':
        transa = _cusparse.CUSPARSE_OPERATION_TRANSPOSE
    elif transa == 'H':
        if dtype.char in 'fd':
            transa = _cusparse.CUSPARSE_OPERATION_TRANSPOSE
        else:
            transa = _cusparse.CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE
    else:
        raise ValueError('Unknown transa (actual: {})'.format(transa))

    if cupyx.scipy.sparse.isspmatrix_csc(a):
        if transa == _cusparse.CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE:
            raise ValueError('If matrix is CSC format and complex dtype,'
                             'transa must not be \'H\'')
        a = a.T
        assert cupyx.scipy.sparse.isspmatrix_csr(a)
        transa = 1 - transa
        fill_mode = 1 - fill_mode

    m = a.shape[0]
    nrhs = 1 if b.ndim == 1 else b.shape[1]
    if b._f_contiguous:
        transb = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
        ldb = m
    elif b._c_contiguous:
        transb = _cusparse.CUSPARSE_OPERATION_TRANSPOSE
        ldb = nrhs
    else:
        raise ValueError('b must be F-contiguous or C-contiguous.')

    handle = _device.get_cusparse_handle()
    alpha = _numpy.array(alpha, dtype=dtype)
    a_desc = MatDescriptor.create()
    a_desc.set_mat_type(_cusparse.CUSPARSE_MATRIX_TYPE_GENERAL)
    a_desc.set_mat_index_base(_cusparse.CUSPARSE_INDEX_BASE_ZERO)
    a_desc.set_mat_fill_mode(fill_mode)
    a_desc.set_mat_diag_type(diag_type)
    info = _cusparse.createCsrsm2Info()
    ws_size = helper(handle, algo, transa, transb, m, nrhs, a.nnz,
                     alpha.ctypes.data, a_desc.descriptor, a.data.data.ptr,
                     a.indptr.data.ptr, a.indices.data.ptr, b.data.ptr, ldb,
                     info, policy)
    ws = _cupy.empty((ws_size,), dtype=_numpy.int8)

    analysis(handle, algo, transa, transb, m, nrhs, a.nnz, alpha.ctypes.data,
             a_desc.descriptor, a.data.data.ptr, a.indptr.data.ptr,
             a.indices.data.ptr, b.data.ptr, ldb, info, policy, ws.data.ptr)

    solve(handle, algo, transa, transb, m, nrhs, a.nnz, alpha.ctypes.data,
          a_desc.descriptor, a.data.data.ptr, a.indptr.data.ptr,
          a.indices.data.ptr, b.data.ptr, ldb, info, policy, ws.data.ptr)


def csrilu02(a, level_info=False):
    """Computes incomplete LU decomposition for a sparse square matrix.

    Args:
        a (cupyx.scipy.sparse.csr_matrix):
            Sparse matrix with dimension ``(M, M)``.
        level_info (bool):
            True: solves it with level infromation.
            False: solves it without level information.

    Note: ``a`` will be overwritten. This function does not support fill-in
        (only ILU(0) is supported) nor pivoting.
    """
    if not check_availability('csrilu02'):
        raise RuntimeError('csrilu02 is not available.')

    if not cupyx.scipy.sparse.isspmatrix_csr(a):
        raise TypeError('a must be CSR sparse matrix')
    if a.shape[0] != a.shape[1]:
        raise ValueError('invalid shape (a.shape: {})'.format(a.shape))

    if level_info is False:
        policy = _cusparse.CUSPARSE_SOLVE_POLICY_NO_LEVEL
    elif level_info is True:
        policy = _cusparse.CUSPARSE_SOLVE_POLICY_USE_LEVEL
    else:
        raise ValueError('Unknown level_info (actual: {})'.format(level_info))

    dtype = a.dtype
    if dtype.char == 'f':
        t = 's'
    elif dtype.char == 'd':
        t = 'd'
    elif dtype.char == 'F':
        t = 'c'
    elif dtype.char == 'D':
        t = 'z'
    else:
        raise TypeError('Invalid dtype (actual: {})'.format(dtype))
    helper = getattr(_cusparse, t + 'csrilu02_bufferSize')
    analysis = getattr(_cusparse, t + 'csrilu02_analysis')
    solve = getattr(_cusparse, t + 'csrilu02')
    check = getattr(_cusparse, 'xcsrilu02_zeroPivot')

    handle = _device.get_cusparse_handle()
    m = a.shape[0]
    nnz = a.nnz
    desc = MatDescriptor.create()
    desc.set_mat_type(_cusparse.CUSPARSE_MATRIX_TYPE_GENERAL)
    desc.set_mat_index_base(_cusparse.CUSPARSE_INDEX_BASE_ZERO)
    info = _cusparse.createCsrilu02Info()
    ws_size = helper(handle, m, nnz, desc.descriptor, a.data.data.ptr,
                     a.indptr.data.ptr, a.indices.data.ptr, info)
    ws = _cupy.empty((ws_size,), dtype=_numpy.int8)
    position = _numpy.empty((1,), dtype=_numpy.int32)

    analysis(handle, m, nnz, desc.descriptor, a.data.data.ptr,
             a.indptr.data.ptr, a.indices.data.ptr, info, policy, ws.data.ptr)
    try:
        check(handle, info, position.ctypes.data)
    except Exception:
        raise ValueError('a({0},{0}) is missing'.format(position[0]))

    solve(handle, m, nnz, desc.descriptor, a.data.data.ptr,
          a.indptr.data.ptr, a.indices.data.ptr, info, policy, ws.data.ptr)
    try:
        check(handle, info, position.ctypes.data)
    except Exception:
        raise ValueError('u({0},{0}) is zero'.format(position[0]))


def denseToSparse(x, format='csr'):
    """Converts a dense matrix into a CSR, CSC or COO format.

    Args:
        x (cupy.ndarray): A matrix to be converted.
        format (str): Format of converted matrix. It must be either 'csr',
            'csc' or 'coo'.

    Returns:
        cupyx.scipy.sparse.spmatrix: A converted sparse matrix.

    """
    if not check_availability('denseToSparse'):
        raise RuntimeError('denseToSparse is not available.')

    assert x.ndim == 2
    assert x.dtype.char in 'fdFD'
    x = _cupy.asfortranarray(x)
    desc_x = DnMatDescriptor.create(x)
    if format == 'csr':
        y = cupyx.scipy.sparse.csr_matrix(x.shape, dtype=x.dtype)
    elif format == 'csc':
        y = cupyx.scipy.sparse.csc_matrix(x.shape, dtype=x.dtype)
    elif format == 'coo':
        y = cupyx.scipy.sparse.coo_matrix(x.shape, dtype=x.dtype)
    else:
        raise TypeError('unsupported format (actual: {})'.format(format))
    desc_y = SpMatDescriptor.create(y)
    algo = _cusparse.CUSPARSE_DENSETOSPARSE_ALG_DEFAULT
    handle = _device.get_cusparse_handle()
    buff_size = _cusparse.denseToSparse_bufferSize(handle, desc_x.desc,
                                                   desc_y.desc, algo)
    buff = _cupy.empty(buff_size, _cupy.int8)
    _cusparse.denseToSparse_analysis(handle, desc_x.desc,
                                     desc_y.desc, algo, buff.data.ptr)
    num_rows_tmp = _numpy.array(0, dtype='int64')
    num_cols_tmp = _numpy.array(0, dtype='int64')
    nnz = _numpy.array(0, dtype='int64')
    _cusparse.spMatGetSize(desc_y.desc, num_rows_tmp.ctypes.data,
                           num_cols_tmp.ctypes.data, nnz.ctypes.data)
    nnz = int(nnz)
    print('# nnz: {}'.format(nnz))
    if format == 'csr':
        indptr = y.indptr
        indices = _cupy.empty(nnz, 'i')
        data = _cupy.empty(nnz, x.dtype)
        y = cupyx.scipy.sparse.csr_matrix((data, indices, indptr),
                                          shape=x.shape)
    elif format == 'csc':
        indptr = y.indptr
        indices = _cupy.empty(nnz, 'i')
        data = _cupy.empty(nnz, x.dtype)
        y = cupyx.scipy.sparse.csc_matrix((data, indices, indptr),
                                          shape=x.shape)
    elif format == 'coo':
        row = _cupy.zeros(nnz, 'i')
        col = _cupy.zeros(nnz, 'i')
        # Note: I would like to use empty() here, but that might cause an
        # exeption in the row/col number check when creating the coo_matrix,
        # so I used zeros() instead.
        data = _cupy.empty(nnz, x.dtype)
        y = cupyx.scipy.sparse.coo_matrix((data, (row, col)), shape=x.shape)
    desc_y = SpMatDescriptor.create(y)
    _cusparse.denseToSparse_convert(handle, desc_x.desc,
                                    desc_y.desc, algo, buff.data.ptr)
    y._has_canonical_format = True
    return y


def sparseToDense(x, out=None):
    """Converts sparse matrix to a dense matrix.

    Args:
        x (cupyx.scipy.sparse.spmatrix): A sparse matrix to convert.
        out (cupy.ndarray or None): A dense metrix to store the result.
            It must be F-contiguous.

    Returns:
        cupy.ndarray: A converted dense matrix.

    """
    if not check_availability('sparseToDense'):
        raise RuntimeError('sparseToDense is not available.')

    dtype = x.dtype
    assert dtype.char in 'fdFD'
    if out is None:
        out = _cupy.zeros(x.shape, dtype=dtype, order='F')
    else:
        assert out.flags.f_contiguous
        assert out.dtype == dtype

    desc_x = SpMatDescriptor.create(x)
    desc_out = DnMatDescriptor.create(out)
    algo = _cusparse.CUSPARSE_SPARSETODENSE_ALG_DEFAULT
    handle = _device.get_cusparse_handle()
    buff_size = _cusparse.sparseToDense_bufferSize(handle, desc_x.desc,
                                                   desc_out.desc, algo)
    buff = _cupy.empty(buff_size, _cupy.int8)
    _cusparse.sparseToDense(handle, desc_x.desc,
                            desc_out.desc, algo, buff.data.ptr)

    return out

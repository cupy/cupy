from __future__ import annotations

import functools as _functools

import numpy as _numpy
import platform as _platform

import cupy as _cupy
from cupy_backends.cuda.api import driver as _driver
from cupy_backends.cuda.api import runtime as _runtime
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupy._core import _dtype
from cupy.cuda import device as _device
from cupy.cuda import stream as _stream
from cupy import _util
import cupyx.scipy.sparse


class MatDescriptor:

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


def _check_int32_indices(a, func_name):
    """Raise ValueError if *a* has int64 indices."""
    if a.indices.dtype == _cupy.int64:
        raise ValueError(
            '{} does not support int64 indices '
            '(cuSPARSE {} is int32-only)'.format(
                func_name, func_name))


def _indptr_to_coo(indptr, dtype=None):
    """Expand compressed ``indptr`` to per-nnz major-axis indices.

    Inverse of :func:`_build_indptr`.  ``dtype`` defaults to ``indptr.dtype``.

    For most matrices the ``cupy.repeat(arange(major), diff)`` formula
    is the right call: O(major + nnz) memory, no host sync.  But when
    the major axis dwarfs ``nnz`` (e.g. the (2, 2**31+5) CSC produced
    by transposing a wide CSR with a single stored entry), the
    ``arange(major)`` allocation dominates -- 17 GB of intermediate
    state for one nnz.  In that regime fall back to a searchsorted
    formula that uses O(nnz log major) memory at the cost of one D2H
    sync to read ``int(indptr[-1])``.
    """
    if dtype is None:
        dtype = indptr.dtype
    nrows = indptr.shape[0] - 1
    # Below 16K rows the ``arange`` is cheap (<= 128 KB at int64) and
    # the log factor of searchsorted is a wash, so prefer ``repeat``.
    # Above the threshold, sync once to read nnz; only switch to
    # searchsorted if the major axis is much larger than nnz (4x gives
    # a safety margin against marginal cases).
    if nrows > (1 << 14):
        nnz = int(indptr[-1])  # synchronize!
        if nrows > 4 * max(nnz, 1):
            arange = _cupy.arange(nnz, dtype=dtype)
            # ``searchsorted`` returns intp; cast back to ``dtype``.
            return _cupy.searchsorted(
                indptr[1:], arange, side='right').astype(dtype, copy=False)
    return _cupy.repeat(
        _cupy.arange(nrows, dtype=dtype), _cupy.diff(indptr))


def _build_indptr(row_indices, n_rows, dtype):
    """Build compressed indptr from per-nnz major-axis assignments.

    Mirrors the histogram + prefix-sum recipe used by scipy.sparse,
    works with both int32 and int64 ``dtype``.
    """
    indptr = _cupy.zeros(n_rows + 1, dtype=dtype)
    _cupy.add.at(indptr[1:], row_indices, 1)
    _cupy.cumsum(indptr, out=indptr)
    return indptr


def _with_indices_dtype(m, dtype):
    """Return a CSR/CSC matrix with indices and indptr cast to *dtype*.

    Data array is shared (no copy).  Used to promote int32 index arrays to
    int64 before calling a cuSPARSE function that requires uniform int64.
    """
    # The short-circuit relies on the invariant indices.dtype == indptr.dtype
    # (enforced by _from_parts and the public constructor).
    if m.indptr.dtype == dtype:
        return m
    # Read private attrs to avoid triggering the lazy property getter
    # (GPU kernel + D2H sync) when the flags have not been computed.
    return m.__class__._from_parts(
        m.data, m.indices.astype(dtype), m.indptr.astype(dtype),
        m.shape,
        has_canonical_format=getattr(
            m, '_has_canonical_format', None),
        has_sorted_indices=getattr(
            m, '_has_sorted_indices', None))


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
    'csrgemm2': (8000, 12000),
    'gthr': (8000, 12000),

    # Generic APIs are not available on CUDA 10.2 on Windows.
    'spmv': ({'Linux': 10200, 'Windows': 11000}, None),
    # accuracy bugs in cuSparse 10.3.0
    'spmm': ({'Linux': 10301, 'Windows': 11000}, None),

    'csr2dense': (8000, 12000),
    'csc2dense': (8000, 12000),
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
    'csrsm2': (9020, 12000),
    'csrilu02': (8000, None),
    'denseToSparse': (11300, None),
    'sparseToDense': (11300, None),
    'spgemm': (11100, None),
    'spsm': (11600, None),  # CUDA 11.3.1
    # TODO(eriknw): cuSPARSE--update when SpGEAM ships in a public release.
    # Present in dev, absent from all public releases through
    # CUDA 13.2 (checked 2026-04-03).  Verified working with dev build:
    # SpGEAM Generic API is ~2x faster than csrgeam2 Legacy for int32,
    # and supports int64 natively.  When shipped, route ALL sparse
    # addition through spgeam() (not just int64) for the speedup.
    'spgeam': (99000, None),
    # CUSPARSE-2365 added int64 SpGEMM in CUDA 13.0, but cuSPARSE ships
    # as version 12.7.9 (12709) for both CUDA 12.7 and 13.0.  The
    # cuSPARSE version alone cannot distinguish the two, so the spgemm()
    # dispatch checks CUDA runtime version (>= 13000) directly instead
    # of using check_availability('spgemm_int64').  This dict entry is
    # kept as a fallback for a future cuSPARSE version that bumps past
    # 12709 and also supports int64 SpGEMM.
    # hipSPARSE entry is (_numpy.inf, None) below -- hipSPARSE spGEMM is
    # int32-only.
    'spgemm_int64': (13000, None),
}


_available_hipsparse_version = {
    # For APIs supported by CUDA but not yet by HIP, we still need them here
    # so that our test suite can cover both platforms.
    'csrmv': (305, None),
    'csrmvEx': (_numpy.inf, None),
    'csrmm': (305, None),
    'csrmm2': (305, None),
    'csrgeam': (305, None),
    'csrgeam2': (305, None),
    'csrgemm': (305, None),
    'csrgemm2': (305, None),
    'gthr': (305, None),
    'spmv': (402, None),
    'spmm': (402, None),
    'csr2dense': (305, None),
    'csc2dense': (305, None),
    'csrsort': (305, None),
    'cscsort': (305, None),
    'coosort': (305, None),
    'coo2csr': (305, None),
    'csr2coo': (305, None),
    'csr2csc': (305, None),
    'csc2csr': (305, None),  # the entity is csr2csc
    'csr2cscEx2': (_numpy.inf, None),
    'csc2csrEx2': (_numpy.inf, None),  # the entity is csr2cscEx2
    'dense2csc': (305, None),
    'dense2csr': (305, None),
    'csr2csr_compress': (305, None),
    'csrsm2': (305, None),  # available since 305 but seems buggy
    'csrilu02': (305, None),
    'denseToSparse': (402, None),
    'sparseToDense': (402, None),
    'spgemm': (_numpy.inf, None),
    'spsm': (50000000, None),
    # TODO(eriknw): hipSPARSE--update when hipSPARSE adds these
    'spgeam': (_numpy.inf, None),
    'spgemm_int64': (_numpy.inf, None),
}


def _get_avail_version_from_spec(x):
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
    if not _runtime.is_hip:
        available_version = _available_cusparse_version
        version = _cusparse.get_build_version()
    else:
        available_version = _available_hipsparse_version
        version = _driver.get_build_version()  # = HIP_VERSION
    if name not in available_version:
        msg = 'No available version information specified for {}'.format(name)
        raise ValueError(msg)
    version_added, version_removed = available_version[name]
    version_added = _get_avail_version_from_spec(version_added)
    version_removed = _get_avail_version_from_spec(version_removed)
    if version_added is not None and version < version_added:
        return False
    if version_removed is not None and version >= version_removed:
        return False
    return True


def getVersion() -> int:
    return _cusparse.getVersion(_device.get_cusparse_handle())


def csrmv(a, x, y=None, alpha=1, beta=0, transa=False):
    """Matrix-vector product for a CSR-matrix and a dense vector.

    .. math::

       y = \\alpha * o_a(A) x + \\beta y,

    where :math:`o_a` is a transpose function when ``transa`` is ``True`` and
    is an identity function otherwise.

    Args:
        a (cupyx.cusparse.csr_matrix): Matrix A.
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

    if y is not None and not y.flags.f_contiguous:
        raise ValueError('expected F-contiguous array for y')

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
        a (cupyx.cusparse.csr_matrix): Matrix A.
        x (cupy.ndarray): Vector x.
        y (cupy.ndarray or None): Vector y.

        Check if a, x, y pointers are aligned by 128 bytes as
        required by csrmvEx.

    Returns:
        bool:
        ``True`` if all pointers are aligned.
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
        a (cupyx.cusparse.csr_matrix): Matrix A.
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

    if y is not None and not y.flags.f_contiguous:
        raise ValueError('expected F-contiguous array for y')

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

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError('expected 2-D matrices')
    if not b.flags.f_contiguous:
        raise ValueError('expected F-contiguous array for b')
    if c is not None and not c.flags.f_contiguous:
        raise ValueError('expected F-contiguous array for c')

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

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError('expected 2-D matrices')
    if not a.has_canonical_format:
        raise ValueError('expected canonical format for a')
    if not b.flags.f_contiguous:
        raise ValueError('expected F-contiguous array for b')
    if c is not None and not c.flags.f_contiguous:
        raise ValueError('expected F-contiguous array for c')
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
    _check_int32_indices(a, 'csrgeam')
    _check_int32_indices(b, 'csrgeam')
    if not a.has_canonical_format:
        raise ValueError('expected canonical format for a')
    if not b.has_canonical_format:
        raise ValueError('expected canonical format for b')
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


def _cupy_csrgeam_int64(a, b, alpha, beta):
    # TODO(eriknw): cuSPARSE--removable once SpGEAM ships and all supported
    # CUDA versions include it.  Keep as fallback for older CUDA.
    """Pure-CuPy CSR addition for int64: C = alpha*A + beta*B.

    Uses COO concatenation + sum_duplicates.  O(nnz log nnz).
    """
    idx_dtype = _numpy.result_type(a.indices.dtype, b.indices.dtype)
    # Use dtype.type() to get a numpy scalar (not a 0-d array) so that CuPy's
    # ufunc accepts it as a broadcast scalar.  _numpy.array(...).ctypes is
    # correct for passing to C but gives a 0-d ndarray that __mul__ rejects.
    a_data = a.data * a.dtype.type(alpha) if alpha != 1 else a.data
    b_data = b.data * b.dtype.type(beta) if beta != 1 else b.data

    a_rows = _indptr_to_coo(a.indptr, idx_dtype)
    b_rows = _indptr_to_coo(b.indptr, idx_dtype)

    rows = _cupy.concatenate([a_rows, b_rows])
    cols = _cupy.concatenate([a.indices, b.indices])
    data = _cupy.concatenate([a_data, b_data])
    coo = cupyx.scipy.sparse.coo_matrix._from_parts(
        data, rows, cols, a.shape)
    coo.has_canonical_format = False
    coo.sum_duplicates()
    return coo.tocsr()


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
    if not isinstance(a, cupyx.scipy.sparse.csr_matrix):
        raise TypeError('unsupported type (actual: {})'.format(type(a)))
    if not isinstance(b, cupyx.scipy.sparse.csr_matrix):
        raise TypeError('unsupported type (actual: {})'.format(type(b)))

    # TODO(eriknw): cuSPARSE--when SpGEAM ships, route ALL addition through
    # spgeam() (not just int64)--benchmarks show SpGEAM Generic API
    # is ~2x faster than csrgeam2 Legacy for int32 at large sizes.
    if a.indices.dtype == _cupy.int64 or b.indices.dtype == _cupy.int64:
        if check_availability('spgeam'):
            return spgeam(a, b, alpha, beta)
        a, b = _cast_common_type(a, b)
        return _cupy_csrgeam_int64(a, b, alpha, beta)

    if not check_availability('csrgeam2'):
        raise RuntimeError('csrgeam2 is not available.')

    if not a.has_canonical_format:
        raise ValueError('expected canonical format for a')
    if not b.has_canonical_format:
        raise ValueError('expected canonical format for b')
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


def spgeam(a, b, alpha=1, beta=1):
    """Sparse matrix addition using the Generic API: C = alpha*A + beta*B.

    Uses ``cusparseSpGEAM`` when available.  Not yet in any public CUDA
    release as of 13.2, but present in dev and verified working
    (~2x faster than csrgeam2 for int32, supports int64 natively).
    Falls back to ``_cupy_csrgeam_int64`` for int64 or ``csrgeam2``
    for int32.

    Args:
        a (cupyx.scipy.sparse.csr_matrix): Sparse matrix A.
        b (cupyx.scipy.sparse.csr_matrix): Sparse matrix B.
        alpha (scalar): Coefficient for A.
        beta (scalar): Coefficient for B.

    Returns:
        cupyx.scipy.sparse.csr_matrix: Result matrix C.

    """
    if not check_availability('spgeam'):
        # spgeam (Generic API) not available on this CUDA version.
        # For int64: use the pure-CuPy fallback directly (no cuSPARSE needed).
        # For int32: delegate to csrgeam2 (the legacy int32-only API).
        if a.indices.dtype == _cupy.int64 or b.indices.dtype == _cupy.int64:
            a, b = _cast_common_type(a, b)
            return _cupy_csrgeam_int64(a, b, alpha, beta)
        return csrgeam2(a, b, alpha, beta)

    if not isinstance(a, cupyx.scipy.sparse.csr_matrix):
        raise TypeError('unsupported type (actual: {})'.format(type(a)))
    if not isinstance(b, cupyx.scipy.sparse.csr_matrix):
        raise TypeError('unsupported type (actual: {})'.format(type(b)))
    if not a.has_canonical_format:
        raise ValueError('expected canonical format for a')
    if not b.has_canonical_format:
        raise ValueError('expected canonical format for b')
    if a.shape != b.shape:
        raise ValueError('inconsistent shapes')

    idx_dtype = _numpy.result_type(a.indices.dtype, b.indices.dtype)
    a, b = _cast_common_type(a, b)
    # cuSPARSE requires uniform index dtype across operands; promote any
    # int32 input to match the common idx_dtype before SpMatDescr.create.
    # _cast_common_type only promotes data dtype, not indices.
    a = _with_indices_dtype(a, idx_dtype)
    b = _with_indices_dtype(b, idx_dtype)
    m, n = a.shape
    handle = _device.get_cusparse_handle()

    # Build an empty C with the right index dtype so SpMatDescr carries it.
    # Use _from_parts to preserve int64 (public constructor downcasts).
    c_indptr = _cupy.empty(m + 1, dtype=idx_dtype)
    c = cupyx.scipy.sparse.csr_matrix._from_parts(
        _cupy.empty(0, a.dtype), _cupy.empty(0, idx_dtype),
        c_indptr, (m, n))

    mat_a = SpMatDescriptor.create(a)
    mat_b = SpMatDescriptor.create(b)
    mat_c = SpMatDescriptor.create(c)

    alpha_arr = _numpy.array(alpha, dtype=a.dtype)
    beta_arr = _numpy.array(beta, dtype=b.dtype)
    cuda_dtype = _dtype.to_cuda_dtype(a.dtype)
    alg = _cusparse.CUSPARSE_SPGEAM_ALG_DEFAULT
    op = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE

    desc = _cusparse.spGEAM_createDescr()
    try:
        # Phase 1: determine external buffer size.
        buf_size = _cusparse.spGEAM_bufferSize(
            handle, op, op,
            alpha_arr.ctypes.data, mat_a.desc,
            beta_arr.ctypes.data, mat_b.desc,
            mat_c.desc, cuda_dtype, alg, desc)
        buf = _cupy.empty(buf_size, _cupy.int8)

        # Phase 2: compute output nnz and fill c's indptr in-place.
        _cusparse.spGEAM_nnz(
            handle, op, op,
            alpha_arr.ctypes.data, mat_a.desc,
            beta_arr.ctypes.data, mat_b.desc,
            mat_c.desc, cuda_dtype, alg, desc, buf.data.ptr)

        # Read output nnz from the descriptor after spGEAM_nnz.
        c_num_rows = _numpy.array(0, dtype='int64')
        c_num_cols = _numpy.array(0, dtype='int64')
        c_nnz_arr = _numpy.array(0, dtype='int64')
        _cusparse.spMatGetSize(mat_c.desc, c_num_rows.ctypes.data,
                               c_num_cols.ctypes.data, c_nnz_arr.ctypes.data)
        c_nnz = int(c_nnz_arr)  # synchronize!

        c_indices = _cupy.empty(c_nnz, idx_dtype)
        c_data = _cupy.empty(c_nnz, a.dtype)
        _cusparse.csrSetPointers(mat_c.desc, c_indptr.data.ptr,
                                 c_indices.data.ptr, c_data.data.ptr)

        # Phase 3: compute values (reuses the same buffer as phase 2).
        _cusparse.spGEAM(
            handle, op, op,
            alpha_arr.ctypes.data, mat_a.desc,
            beta_arr.ctypes.data, mat_b.desc,
            mat_c.desc, cuda_dtype, alg, desc, buf.data.ptr)
    finally:
        _cusparse.spGEAM_destroyDescr(desc)

    c = cupyx.scipy.sparse.csr_matrix._from_parts(
        c_data, c_indices, c_indptr, (m, n),
        has_canonical_format=True)
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

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError('expected 2-D matrices')
    _check_int32_indices(a, 'csrgemm')
    _check_int32_indices(b, 'csrgemm')
    if not a.has_canonical_format:
        raise ValueError('expected canonical format for a')
    if not b.has_canonical_format:
        raise ValueError('expected canonical format for b')
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

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError('expected 2-D matrices')
    if not isinstance(a, cupyx.scipy.sparse.csr_matrix):
        raise TypeError('unsupported type (actual: {})'.format(type(a)))
    if not isinstance(b, cupyx.scipy.sparse.csr_matrix):
        raise TypeError('unsupported type (actual: {})'.format(type(b)))
    _check_int32_indices(a, 'csrgemm2')
    _check_int32_indices(b, 'csrgemm2')
    if not a.has_canonical_format:
        raise ValueError('expected canonical format for a')
    if not b.has_canonical_format:
        raise ValueError('expected canonical format for b')
    if a.shape[1] != b.shape[0]:
        raise ValueError('mismatched shape')
    if d is not None:
        if d.ndim != 2:
            raise ValueError('expected 2-D matrix for d')
        if not isinstance(d, cupyx.scipy.sparse.csr_matrix):
            raise TypeError('unsupported type (actual: {})'.format(type(d)))
        _check_int32_indices(d, 'csrgemm2')
        if not d.has_canonical_format:
            raise ValueError('expected canonical format for d')
        if a.shape[0] != d.shape[0] or b.shape[1] != d.shape[1]:
            raise ValueError('mismatched shape')
        if _runtime.is_hip and _driver.get_build_version() < 402:
            raise RuntimeError('d != None is supported since ROCm 4.2.0')

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
        out (cupy.ndarray or None): A dense matrix to store the result.
            It must be F-contiguous.

    Returns:
        cupy.ndarray: Converted result.

    """
    if not check_availability('csr2dense'):
        raise RuntimeError('csr2dense is not available.')

    dtype = x.dtype
    if dtype.char not in 'fdFD':
        raise ValueError('unsupported dtype')
    if out is None:
        out = _cupy.empty(x.shape, dtype=dtype, order='F')
    else:
        if not out.flags.f_contiguous:
            raise ValueError('expected F-contiguous array for out')

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
        out (cupy.ndarray or None): A dense matrix to store the result.
            It must be F-contiguous.

    Returns:
        cupy.ndarray: Converted result.

    """
    if not check_availability('csc2dense'):
        raise RuntimeError('csc2dense is not available.')

    dtype = x.dtype
    if dtype.char not in 'fdFD':
        raise ValueError('unsupported dtype')
    if out is None:
        out = _cupy.empty(x.shape, dtype=dtype, order='F')
    else:
        if not out.flags.f_contiguous:
            raise ValueError('expected F-contiguous array for out')

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
    nnz = x.nnz
    if nnz == 0:
        return
    m, n = x.shape

    if x.indices.dtype == _cupy.int64:
        # TODO(eriknw): cuSPARSE--remove when xcsrsort supports int64
        row = _indptr_to_coo(x.indptr)
        order = _cupy.lexsort(_cupy.stack([x.indices, row]))
        x.indices[:] = x.indices[order]
        x.data[:] = x.data[order]
        x.has_sorted_indices = True
        return

    if not check_availability('csrsort'):
        raise RuntimeError('csrsort is not available.')
    handle = _device.get_cusparse_handle()
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

    if check_availability('gthr'):
        _call_cusparse(
            'gthr', x.dtype,
            handle, nnz, data_orig.data.ptr, x.data.data.ptr,
            P.data.ptr, _cusparse.CUSPARSE_INDEX_BASE_ZERO)
    else:
        desc_x = SpVecDescriptor.create(P, x.data)
        desc_y = DnVecDescriptor.create(data_orig)
        _cusparse.gather(handle, desc_y.desc, desc_x.desc)


def cscsort(x):
    """Sorts indices of CSC-matrix in place.

    Args:
        x (cupyx.scipy.sparse.csc_matrix): A sparse matrix to sort.

    """
    nnz = x.nnz
    if nnz == 0:
        return
    m, n = x.shape

    if x.indices.dtype == _cupy.int64:
        # TODO(eriknw): cuSPARSE--remove when xcscsort supports int64
        col = _indptr_to_coo(x.indptr)
        order = _cupy.lexsort(_cupy.stack([x.indices, col]))
        x.indices[:] = x.indices[order]
        x.data[:] = x.data[order]
        x.has_sorted_indices = True
        return

    if not check_availability('cscsort'):
        raise RuntimeError('cscsort is not available.')
    handle = _device.get_cusparse_handle()
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

    if check_availability('gthr'):
        _call_cusparse(
            'gthr', x.dtype,
            handle, nnz, data_orig.data.ptr, x.data.data.ptr,
            P.data.ptr, _cusparse.CUSPARSE_INDEX_BASE_ZERO)
    else:
        desc_x = SpVecDescriptor.create(P, x.data)
        desc_y = DnVecDescriptor.create(data_orig)
        _cusparse.gather(handle, desc_y.desc, desc_x.desc)


def coosort(x, sort_by='r'):
    """Sorts indices of COO-matrix in place.

    Args:
        x (cupyx.scipy.sparse.coo_matrix): A sparse matrix to sort.
        sort_by (str): Sort the indices by row ('r', default) or column ('c').

    """
    nnz = x.nnz
    if nnz == 0:
        return
    if sort_by == 'r' and x.has_canonical_format:
        return
    m, n = x.shape

    if x.row.dtype == _cupy.int64:
        # TODO(eriknw): cuSPARSE--remove when xcoosort supports int64
        if sort_by == 'r':
            # Sort by (row, col): primary=row, secondary=col
            order = _cupy.lexsort(_cupy.stack([x.col, x.row]))
        elif sort_by == 'c':
            # Sort by (col, row): primary=col, secondary=row
            order = _cupy.lexsort(_cupy.stack([x.row, x.col]))
        else:
            raise ValueError("sort_by must be either 'r' or 'c'")
        x.row[:] = x.row[order]
        x.col[:] = x.col[order]
        x.data[:] = x.data[order]
        if sort_by == 'c':
            x.has_canonical_format = False
        return

    if not check_availability('coosort'):
        raise RuntimeError('coosort is not available.')
    handle = _device.get_cusparse_handle()
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

    if x.dtype.char != '?':
        if check_availability('gthr'):
            _call_cusparse(
                'gthr', x.dtype,
                handle, nnz, data_orig.data.ptr, x.data.data.ptr,
                P.data.ptr, _cusparse.CUSPARSE_INDEX_BASE_ZERO)
        else:
            desc_x = SpVecDescriptor.create(P, x.data)
            desc_y = DnVecDescriptor.create(data_orig)
            _cusparse.gather(handle, desc_y.desc, desc_x.desc)

    if sort_by == 'c':  # coo canonical order is row-major
        x.has_canonical_format = False


def coo2csr(x):
    m = int(x.shape[0])
    nnz = x.nnz
    idx_dtype = x.row.dtype
    if nnz == 0:
        indptr = _cupy.zeros(m + 1, dtype=idx_dtype)
    elif idx_dtype == _cupy.int64:
        # TODO(eriknw): cuSPARSE--remove when xcoo2csr supports int64
        indptr = _build_indptr(x.row, m, idx_dtype)
    else:
        handle = _device.get_cusparse_handle()
        indptr = _cupy.empty(m + 1, dtype=idx_dtype)
        _cusparse.xcoo2csr(
            handle, x.row.data.ptr, nnz, m,
            indptr.data.ptr, _cusparse.CUSPARSE_INDEX_BASE_ZERO)
    return cupyx.scipy.sparse.csr_matrix._from_parts(
        x.data, x.col, indptr, x.shape)


def coo2csc(x):
    n = int(x.shape[1])
    nnz = x.nnz
    idx_dtype = x.col.dtype
    if nnz == 0:
        indptr = _cupy.zeros(n + 1, dtype=idx_dtype)
    elif idx_dtype == _cupy.int64:
        # TODO(eriknw): cuSPARSE--remove when xcoo2csr supports int64
        indptr = _build_indptr(x.col, n, idx_dtype)
    else:
        handle = _device.get_cusparse_handle()
        indptr = _cupy.empty(n + 1, dtype=idx_dtype)
        _cusparse.xcoo2csr(
            handle, x.col.data.ptr, nnz, n,
            indptr.data.ptr, _cusparse.CUSPARSE_INDEX_BASE_ZERO)
    return cupyx.scipy.sparse.csc_matrix._from_parts(
        x.data, x.row, indptr, x.shape)


def csr2coo(x, data, indices):
    """Converts a CSR-matrix to COO format.

    Args:
        x (cupyx.scipy.sparse.csr_matrix): A matrix to be converted.
        data (cupy.ndarray): A data array for converted data.
        indices (cupy.ndarray): An index array for converted data.

    Returns:
        cupyx.scipy.sparse.coo_matrix: A converted matrix.

    """
    m = int(x.shape[0])
    nnz = x.nnz
    idx_dtype = x.indptr.dtype

    if idx_dtype == _cupy.int64:
        # TODO(eriknw): cuSPARSE--remove when xcsr2coo supports int64
        row = _indptr_to_coo(x.indptr)
    else:
        if not check_availability('csr2coo'):
            raise RuntimeError('csr2coo is not available.')
        handle = _device.get_cusparse_handle()
        row = _cupy.empty(nnz, idx_dtype)
        _cusparse.xcsr2coo(
            handle, x.indptr.data.ptr, nnz, m, row.data.ptr,
            _cusparse.CUSPARSE_INDEX_BASE_ZERO)
    # Preserve has_canonical_format: a canonical CSR (sorted column
    # indices within each row, no duplicates) expands to row-major
    # lexicographic COO order, which is exactly the COO canonical form.
    # Read the cached flag directly to avoid triggering the lazy GPU
    # kernel on the source; uncached/False both fall through to False.
    A = cupyx.scipy.sparse.coo_matrix._from_parts(
        data, row, indices, x.shape,
        has_canonical_format=bool(
            getattr(x, '_has_canonical_format', False)))
    return A


def _cupy_transpose_compressed_int64(x, output_cls, out_dim):
    """Pure-CuPy CSR<->CSC transpose for int64 indices.

    Uses ``_build_indptr`` + ``lexsort`` -- O(nnz log nnz) time.
    Used as the int64 fallback because no Generic API CSR<->CSC
    function exists as of CUDA 13.2 / dev.  This is the biggest
    perf gap: int64 tocsc is 7-10x slower than int32 (measured on
    Blackwell, 10K-100K matrices).

    Args:
        x: Input compressed sparse matrix.
        output_cls: Output class (csc_matrix or csr_matrix).
        out_dim: Size of the output's major axis (n for CSC, m for CSR).
    """
    # TODO(eriknw): cuSPARSE--remove when a Generic API CSR<->CSC
    # function ships.
    nnz = x.nnz
    idx_dtype = x.indices.dtype

    if nnz == 0:
        return output_cls._from_parts(
            _cupy.empty(0, x.dtype),
            _cupy.empty(0, idx_dtype),
            _cupy.zeros(out_dim + 1, idx_dtype),
            x.shape,
            has_sorted_indices=True)

    out_indptr = _build_indptr(x.indices, out_dim, idx_dtype)
    expanded = _indptr_to_coo(x.indptr)

    # Sort by (output major, output minor) for canonical order.
    order = _cupy.lexsort(_cupy.stack([expanded, x.indices]))

    # Preserve has_canonical_format: a canonical input (sorted col
    # indices, no duplicates) transposes to a CSC/CSR whose minor
    # indices are also sorted-no-dup within each major slot.  When the
    # input is *not* canonical (or canonical state is unknown), the
    # output may still be canonical -- the lexsort makes the output
    # sorted regardless of the input's order, and the output has
    # duplicates iff the input did.  Conservatively propagate only
    # known-True; False / None both leave the output's flag unset.
    canonical = getattr(x, '_has_canonical_format', None)
    return output_cls._from_parts(
        x.data[order], expanded[order], out_indptr, x.shape,
        has_canonical_format=True if canonical else None,
        has_sorted_indices=True)


def csr2csc(x):
    if x.indices.dtype == _cupy.int64:
        # TODO(eriknw): cuSPARSE--remove when a Generic API CSR<->CSC ships
        # (or when csr2csc supports int64).  No such API exists as
        # of dev or CUDA 13.2.  Biggest perf gap: 7-10x.
        return _cupy_transpose_compressed_int64(
            x, cupyx.scipy.sparse.csc_matrix, int(x.shape[1]))

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
    if x.indices.dtype == _cupy.int64:
        # TODO(eriknw): cuSPARSE--see csr2csc above (same gap: 7-10x)
        return _cupy_transpose_compressed_int64(
            x, cupyx.scipy.sparse.csc_matrix, int(x.shape[1]))

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
    n = int(x.shape[1])
    nnz = x.nnz
    idx_dtype = x.indptr.dtype

    if idx_dtype == _cupy.int64:
        # TODO(eriknw): cuSPARSE--remove when xcsr2coo supports int64
        col = _indptr_to_coo(x.indptr)
    else:
        handle = _device.get_cusparse_handle()
        col = _cupy.empty(nnz, idx_dtype)
        _cusparse.xcsr2coo(
            handle, x.indptr.data.ptr, nnz, n, col.data.ptr,
            _cusparse.CUSPARSE_INDEX_BASE_ZERO)
    A = cupyx.scipy.sparse.coo_matrix._from_parts(
        data, indices, col, x.shape)
    A.has_canonical_format = False
    return A


def csc2csr(x):
    if x.indices.dtype == _cupy.int64:
        # TODO(eriknw): cuSPARSE--see csr2csc above (same gap: 7-10x)
        return _cupy_transpose_compressed_int64(
            x, cupyx.scipy.sparse.csr_matrix, int(x.shape[0]))

    if not check_availability('csc2csr'):
        raise RuntimeError('csc2csr is not available.')

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
    if x.indices.dtype == _cupy.int64:
        # TODO(eriknw): cuSPARSE--see csr2csc above (same gap: 7-10x)
        return _cupy_transpose_compressed_int64(
            x, cupyx.scipy.sparse.csr_matrix, int(x.shape[0]))

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

    if x.ndim != 2:
        raise ValueError('expected 2-D matrix')
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
    # Note that a descriptor is recreated
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

    if x.ndim != 2:
        raise ValueError('expected 2-D matrix')
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
    if _runtime.is_hip:
        if nnz == 0:
            raise ValueError('hipSPARSE currently cannot handle '
                             'sparse matrices with null ptrs')
    data = _cupy.empty(nnz, x.dtype)
    indptr = _cupy.empty(m + 1, 'i')
    indices = _cupy.empty(nnz, 'i')

    _call_cusparse(
        'dense2csr', x.dtype,
        handle, m, n, descr.descriptor,
        x.data.ptr, m, nnz_per_row.data.ptr,
        data.data.ptr, indptr.data.ptr, indices.data.ptr)
    # Note that a descriptor is recreated
    csr = cupyx.scipy.sparse.csr_matrix((data, indices, indptr), shape=x.shape)
    csr._has_canonical_format = True
    return csr


def csr2csr_compress(x, tol):
    if not check_availability('csr2csr_compress'):
        raise RuntimeError('csr2csr_compress is not available.')

    if x.dtype.char not in 'fdFD':
        raise ValueError('unsupported dtype')

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


class BaseDescriptor:

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
        if not cupyx.scipy.sparse.issparse(a):
            raise TypeError('expected sparse matrix')
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

    def set_attribute(self, attribute, data):
        _cusparse.spMatSetAttribute(self.desc, attribute, data)


class SpVecDescriptor(BaseDescriptor):

    @classmethod
    def create(cls, idx, x):
        nnz = x.size
        cuda_dtype = _dtype.to_cuda_dtype(x.dtype)
        desc = _cusparse.createSpVec(nnz, nnz, idx.data.ptr, x.data.ptr,
                                     _dtype_to_IndexType(idx.dtype),
                                     _cusparse.CUSPARSE_INDEX_BASE_ZERO,
                                     cuda_dtype)
        get = _cusparse.spVecGet
        destroy = _cusparse.destroySpVec
        return SpVecDescriptor(desc, get, destroy)


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
        if a.ndim != 2:
            raise ValueError('expected 2-D matrix')
        if not a.flags.f_contiguous:
            raise ValueError('expected F-contiguous array')
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
        alpha (scalar): Coefficient
        beta (scalar): Coefficient
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
    if not a.has_canonical_format:
        raise ValueError('expected canonical format')

    m, n = a_shape
    a, x, y = _cast_common_type(a, x, y)
    if y is None:
        y = _cupy.zeros(m, a.dtype)
    elif len(y) != m:
        raise ValueError('dimension mismatch')
    if a.nnz == 0:
        y.fill(0)
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


# cuSPARSE csrmm_alg1 gridDim.y overflow threshold (#9850).
# The kernel sets gridDim.y = ceil(n / 16); hardware max is 65535.
# Fixed in cuSPARSE 12.7.20+ (CUDA 13.2+).
_SPMM_MAX_DENSE_COLS = 65535 * 16  # 1,048,560


def spmm(a, b, c=None, alpha=1, beta=0, transa=False, transb=False):
    """Multiplication of sparse matrix and dense matrix.

    .. math::

        C = \\alpha * op(A) op(B) + \\beta * C

    Args:
        a (cupyx.scipy.sparse.csr_matrix, csc_matrix or coo_matrix):
            Sparse matrix A
        b (cupy.ndarray): Dense matrix B
        c (cupy.ndarray or None): Dense matrix C
        alpha (scalar): Coefficient
        beta (scalar): Coefficient
        transa (bool): If ``True``, op(A) = transpose of A.
        transb (bool): If ``True``, op(B) = transpose of B.

    Returns:
        cupy.ndarray
    """
    if not check_availability('spmm'):
        raise RuntimeError('spmm is not available.')

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError('expected 2-D matrices')
    if not b.flags.f_contiguous:
        raise ValueError('expected F-contiguous array for b')
    if c is not None and not c.flags.f_contiguous:
        raise ValueError('expected F-contiguous array for c')

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
    if not a.has_canonical_format:
        raise ValueError('expected canonical format for a')

    m, k = a_shape
    _, n = b_shape
    a, b, c = _cast_common_type(a, b, c)
    if c is None:
        c = _cupy.zeros((m, n), a.dtype, 'F')
    elif c.shape[0] != m or c.shape[1] != n:
        raise ValueError('dimension mismatch')
    if a.nnz == 0:
        c.fill(0)
        return c

    if n > _SPMM_MAX_DENSE_COLS and getVersion() < 12720:
        # Workaround for cuSPARSE csrmm_alg1 gridDim.y overflow (#9850)
        for col0 in range(0, n, _SPMM_MAX_DENSE_COLS):
            col1 = min(col0 + _SPMM_MAX_DENSE_COLS, n)
            if transb:
                # Row slice of F-contiguous is not F-contiguous; copy needed
                b_chunk = _cupy.asfortranarray(b[col0:col1, :])
            else:
                # Column slice of F-contiguous is F-contiguous; no copy
                b_chunk = b[:, col0:col1]
            c_chunk = c[:, col0:col1]
            spmm(a, b_chunk, c=c_chunk, alpha=alpha, beta=beta,
                 transa=transa, transb=transb)
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
        alpha (float or complex): Coefficient.
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
            True: solves it with level information.
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
    _check_int32_indices(a, 'csrsm2')

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
        if not cupyx.scipy.sparse.isspmatrix_csr(a):
            raise TypeError('expected CSR matrix after transpose')
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

    # without sync we'd get either segfault or cuda context error
    _stream.get_current_stream().synchronize()
    _cusparse.destroyCsrsm2Info(info)


def csrilu02(a, level_info=False):
    """Computes incomplete LU decomposition for a sparse square matrix.

    Args:
        a (cupyx.scipy.sparse.csr_matrix):
            Sparse matrix with dimension ``(M, M)``.
        level_info (bool):
            True: solves it with level information.
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
    _check_int32_indices(a, 'csrilu02')

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

    if x.ndim != 2:
        raise ValueError('expected 2-D matrix')
    if x.dtype.char not in 'fdFD':
        raise ValueError('unsupported dtype')
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
    if _runtime.is_hip:
        if nnz == 0:
            raise ValueError('hipSPARSE currently cannot handle '
                             'sparse matrices with null ptrs')
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
        # exception in the row/col number check when creating the coo_matrix,
        # so I used zeros() instead.
        data = _cupy.empty(nnz, x.dtype)
        y = cupyx.scipy.sparse.coo_matrix((data, (row, col)), shape=x.shape)
    desc_y = SpMatDescriptor.create(y)
    _cusparse.denseToSparse_convert(handle, desc_x.desc,
                                    desc_y.desc, algo, buff.data.ptr)
    # COO uses ``has_canonical_format`` (no leading underscore) as its
    # plain attribute -- writing ``_has_canonical_format`` would set a
    # different attribute and the result would not be marked canonical.
    y.has_canonical_format = True
    return y


def sparseToDense(x, out=None):
    """Converts sparse matrix to a dense matrix.

    Args:
        x (cupyx.scipy.sparse.spmatrix): A sparse matrix to convert.
        out (cupy.ndarray or None): A dense matrix to store the result.
            It must be F-contiguous.

    Returns:
        cupy.ndarray: A converted dense matrix.

    """
    if not check_availability('sparseToDense'):
        raise RuntimeError('sparseToDense is not available.')

    dtype = x.dtype
    if dtype.char not in 'fdFD':
        raise ValueError('unsupported dtype')
    if out is None:
        out = _cupy.zeros(x.shape, dtype=dtype, order='F')
    else:
        if not out.flags.f_contiguous:
            raise ValueError('expected F-contiguous array for out')
        if out.dtype != dtype:
            raise ValueError('out dtype mismatch')

    desc_x = SpMatDescriptor.create(x)
    desc_out = DnMatDescriptor.create(out)
    algo = _cusparse.CUSPARSE_SPARSETODENSE_ALG_DEFAULT
    handle = _device.get_cusparse_handle()
    buff_size = _cusparse.sparseToDense_bufferSize(handle, desc_x.desc,
                                                   desc_out.desc, algo)
    buff = _cupy.empty(buff_size, _cupy.int8)
    if _runtime.is_hip:
        if x.nnz == 0:
            raise ValueError('hipSPARSE currently cannot handle '
                             'sparse matrices with null ptrs')
    _cusparse.sparseToDense(handle, desc_x.desc,
                            desc_out.desc, algo, buff.data.ptr)

    return out


def spsm(a, b, alpha=1.0, lower=True, unit_diag=False, transa=False):
    """Solves a sparse triangular linear system op(a) * x = alpha * op(b).

    Args:
        a (cupyx.scipy.sparse.csr_matrix or cupyx.scipy.sparse.coo_matrix):
            Sparse matrix with dimension ``(M, M)``.
        b (cupy.ndarray): Dense matrix with dimension ``(M, K)``.
        alpha (float or complex): Coefficient.
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
    """
    if not check_availability('spsm'):
        raise RuntimeError('spsm is not available.')

    # Canonicalise transa
    if transa is False:
        transa = 'N'
    elif transa is True:
        transa = 'T'
    elif transa not in 'NTH':
        raise ValueError(f'Unknown transa (actual: {transa})')

    # Check A's type and sparse format
    if cupyx.scipy.sparse.isspmatrix_csr(a):
        pass
    elif cupyx.scipy.sparse.isspmatrix_csc(a):
        if transa == 'N':
            a = a.T
            transa = 'T'
        elif transa == 'T':
            a = a.T
            transa = 'N'
        elif transa == 'H':
            a = a.conj().T
            transa = 'N'
        lower = not lower
    elif cupyx.scipy.sparse.isspmatrix_coo(a):
        pass
    else:
        raise ValueError('a must be CSR, CSC or COO sparse matrix')
    if not a.has_canonical_format:
        raise ValueError('expected canonical format for a')

    # Check B's ndim
    if b.ndim == 1:
        is_b_vector = True
        b = b.reshape(-1, 1)
    elif b.ndim == 2:
        is_b_vector = False
    else:
        raise ValueError('b.ndim must be 1 or 2')

    # Check shapes
    if not (a.shape[0] == a.shape[1] == b.shape[0]):
        raise ValueError('mismatched shape')

    # Check dtypes
    dtype = a.dtype
    if dtype.char not in 'fdFD':
        raise TypeError('Invalid dtype (actual: {})'.format(dtype))
    if dtype != b.dtype:
        raise TypeError('dtype mismatch')

    # Prepare fill mode
    if lower is True:
        fill_mode = _cusparse.CUSPARSE_FILL_MODE_LOWER
    elif lower is False:
        fill_mode = _cusparse.CUSPARSE_FILL_MODE_UPPER
    else:
        raise ValueError('Unknown lower (actual: {})'.format(lower))

    # Prepare diag type
    if unit_diag is False:
        diag_type = _cusparse.CUSPARSE_DIAG_TYPE_NON_UNIT
    elif unit_diag is True:
        diag_type = _cusparse.CUSPARSE_DIAG_TYPE_UNIT
    else:
        raise ValueError('Unknown unit_diag (actual: {})'.format(unit_diag))

    # Prepare op_a
    if transa == 'N':
        op_a = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
    elif transa == 'T':
        op_a = _cusparse.CUSPARSE_OPERATION_TRANSPOSE
    else:  # transa == 'H'
        if dtype.char in 'fd':
            op_a = _cusparse.CUSPARSE_OPERATION_TRANSPOSE
        else:
            op_a = _cusparse.CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE

    # Prepare op_b
    if b._f_contiguous:
        op_b = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
    elif b._c_contiguous:
        if _cusparse.get_build_version() < 11701:  # earlier than CUDA 11.6
            raise ValueError('b must be F-contiguous.')
        b = b.T
        op_b = _cusparse.CUSPARSE_OPERATION_TRANSPOSE
    else:
        raise ValueError('b must be F-contiguous or C-contiguous.')

    # Allocate space for matrix C. Note that it is known cusparseSpSM requires
    # the output matrix zero initialized.
    m, _ = a.shape
    if op_b == _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE:
        _, n = b.shape
    else:
        n, _ = b.shape
    c_shape = m, n
    c = _cupy.zeros(c_shape, dtype=a.dtype, order='f')

    # Prepare descriptors and other parameters
    handle = _device.get_cusparse_handle()
    mat_a = SpMatDescriptor.create(a)
    mat_b = DnMatDescriptor.create(b)
    mat_c = DnMatDescriptor.create(c)
    spsm_descr = _cusparse.spSM_createDescr()
    alpha = _numpy.array(alpha, dtype=c.dtype).ctypes
    cuda_dtype = _dtype.to_cuda_dtype(c.dtype)
    algo = _cusparse.CUSPARSE_SPSM_ALG_DEFAULT

    try:
        # Specify Lower|Upper fill mode
        mat_a.set_attribute(_cusparse.CUSPARSE_SPMAT_FILL_MODE, fill_mode)

        # Specify Unit|Non-Unit diagonal type
        mat_a.set_attribute(_cusparse.CUSPARSE_SPMAT_DIAG_TYPE, diag_type)

        # Allocate the workspace needed by the succeeding phases
        buff_size = _cusparse.spSM_bufferSize(
            handle, op_a, op_b, alpha.data, mat_a.desc, mat_b.desc,
            mat_c.desc, cuda_dtype, algo, spsm_descr)
        buff = _cupy.empty(buff_size, dtype=_cupy.int8)

        # Perform the analysis phase
        _cusparse.spSM_analysis(
            handle, op_a, op_b, alpha.data, mat_a.desc, mat_b.desc,
            mat_c.desc, cuda_dtype, algo, spsm_descr, buff.data.ptr)

        # Executes the solve phase
        _cusparse.spSM_solve(
            handle, op_a, op_b, alpha.data, mat_a.desc, mat_b.desc,
            mat_c.desc, cuda_dtype, algo, spsm_descr, buff.data.ptr)

        # Reshape back if B was a vector
        if is_b_vector:
            c = c.reshape(-1)

        return c

    finally:
        # Destroy matrix/vector descriptors
        _cusparse.spSM_destroyDescr(spsm_descr)


def _cupy_spgemm_int64(a, b, alpha):
    # TODO(eriknw): cuSPARSE--removable once all supported CUDA versions have
    # native int64 SpGEMM.  Verified working on CUDA 13.0+ (native is
    # ~14x faster).  Keep as fallback for CUDA < 13.0.
    """Pure-CuPy sort-merge SpGEMM for int64: C = alpha * A * B.

    Used on CUDA < 13.0 where cuSPARSE spGEMM lacks int64.
    Expands all (i,j,a*b) products into COO, then sum_duplicates.
    O(P log P) time, O(P + nnz_C) space, where P = total products.
    """
    idx_dtype = _cupy.int64
    # Normalise both operands to int64 (no copy if already int64).
    a_indices = a.indices.astype(idx_dtype, copy=False)
    a_indptr = a.indptr.astype(idx_dtype, copy=False)
    b_indices = b.indices.astype(idx_dtype, copy=False)
    b_indptr = b.indptr.astype(idx_dtype, copy=False)

    m = a.shape[0]
    n = b.shape[1]

    # Number of B-row-k entries accessible from each nonzero of A at column k.
    b_row_len = _cupy.diff(b_indptr)              # shape (K,)
    products_per_a = b_row_len[a_indices]         # shape (a.nnz,)
    total_products = int(products_per_a.sum())  # synchronize!

    if total_products == 0:
        return cupyx.scipy.sparse.csr_matrix._from_parts(
            _cupy.empty(0, a.dtype),
            _cupy.empty(0, idx_dtype),
            _cupy.zeros(m + 1, idx_dtype),
            (m, n))

    a_src = _cupy.repeat(
        _cupy.arange(a.nnz, dtype=idx_dtype), products_per_a)
    # b_offset: position within each B row (grouped arange)
    cum_prod = _cupy.zeros(a.nnz + 1, dtype=idx_dtype)
    _cupy.cumsum(products_per_a, out=cum_prod[1:])
    b_offset = (
        _cupy.arange(total_products, dtype=idx_dtype) - cum_prod[a_src])
    del cum_prod

    # Expand A indptr -> row index for each A nonzero.
    a_rows = _indptr_to_coo(a_indptr)

    # Gather the (output_row, output_col, value) triple for each product.
    a_col_k = a_indices[a_src]                       # column of A = row of B
    b_row_start = b_indptr[a_col_k]                  # start pos in B arrays
    b_col_pos = b_row_start + b_offset
    c_cols = b_indices[b_col_pos]
    c_rows = a_rows[a_src]
    c_vals = a.data[a_src] * b.data[b_col_pos]
    del a_src, b_offset, b_col_pos, a_rows, a_col_k  # free P-sized temporaries
    if alpha != 1:
        c_vals = c_vals * a.dtype.type(alpha)

    # Merge products with the same (row, col) position.
    coo = cupyx.scipy.sparse.coo_matrix._from_parts(
        c_vals, c_rows, c_cols, shape=(m, n))
    coo.sum_duplicates()
    return coo.tocsr()


def spgemm(a, b, alpha=1):
    """Matrix-matrix product for CSR-matrix.

    math::
       C = alpha * A * B

    Args:
        a (cupyx.scipy.sparse.csr_matrix): Sparse matrix A.
        b (cupyx.scipy.sparse.csr_matrix): Sparse matrix B.
        alpha (scalar): Coefficient

    Returns:
        cupyx.scipy.sparse.csr_matrix

    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError('expected 2-D matrices')
    if not isinstance(a, cupyx.scipy.sparse.csr_matrix):
        raise TypeError('unsupported type (actual: {})'.format(type(a)))
    if not isinstance(b, cupyx.scipy.sparse.csr_matrix):
        raise TypeError('unsupported type (actual: {})'.format(type(b)))

    # TODO(eriknw): cuSPARSE--remove pure-CuPy fallback when all supported CUDA
    # versions have native int64 SpGEMM.  Native int64 SpGEMM is ~14x
    # faster and within 1.6x of int32 (verified on CUDA 13.0+).
    # cuSPARSE 12.7.9 ships with both CUDA 12.7 and 13.0 but only
    # CUDA 13.0+ supports int64 SpGEMM (CUSPARSE-2365), so we check
    # the CUDA runtime version rather than check_availability().
    # Mixed int32/int64 inputs: upcast to int64, prefer cuSPARSE.
    if a.indices.dtype == _cupy.int64 or b.indices.dtype == _cupy.int64:
        if a.shape[1] != b.shape[0]:
            raise ValueError('mismatched shape')
        _native_int64 = (
            not _runtime.is_hip
            and _runtime.runtimeGetVersion() >= 13000
            and check_availability('spgemm')
        ) or check_availability('spgemm_int64')
        if _native_int64:
            # Native int64 available: cuSPARSE requires uniform index
            # dtype, so upcast any int32 operand to int64 before falling
            # through to the regular cuSPARSE path below.
            a = _with_indices_dtype(a, _cupy.int64)
            b = _with_indices_dtype(b, _cupy.int64)
        else:
            a, b = _cast_common_type(a, b)
            return _cupy_spgemm_int64(a, b, alpha)

    if not check_availability('spgemm'):
        raise RuntimeError('spgemm is not available.')

    if not a.has_canonical_format:
        raise ValueError('expected canonical format for a')
    if not b.has_canonical_format:
        raise ValueError('expected canonical format for b')
    if a.shape[1] != b.shape[0]:
        raise ValueError('mismatched shape')

    m, k = a.shape
    _, n = b.shape
    idx_dtype = _numpy.result_type(a.indices.dtype, b.indices.dtype)
    a, b = _cast_common_type(a, b)
    c_shape = (m, n)
    c_empty_indptr = _cupy.zeros(m + 1, dtype=idx_dtype)
    # Use _from_parts to preserve int64 index dtype -- the public
    # constructor would downcast empty int64 arrays to int32 via
    # check_contents, causing cuSPARSE to reject mixed index types.
    c = cupyx.scipy.sparse.csr_matrix._from_parts(
        _cupy.empty(0, a.dtype), _cupy.empty(0, idx_dtype),
        c_empty_indptr, c_shape)

    handle = _device.get_cusparse_handle()
    mat_a = SpMatDescriptor.create(a)
    mat_b = SpMatDescriptor.create(b)
    mat_c = SpMatDescriptor.create(c)
    spgemm_descr = _cusparse.spGEMM_createDescr()
    op_a = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
    op_b = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
    alpha = _numpy.array(alpha, dtype=c.dtype).ctypes
    beta = _numpy.array(0, dtype=c.dtype).ctypes
    cuda_dtype = _dtype.to_cuda_dtype(c.dtype)
    algo = _cusparse.CUSPARSE_SPGEMM_DEFAULT
    null_ptr = 0

    try:
        # Analyze the matrices A and B to understand the memory requirement
        buff1_size = _cusparse.spGEMM_workEstimation(
            handle, op_a, op_b, alpha.data, mat_a.desc, mat_b.desc, beta.data,
            mat_c.desc, cuda_dtype, algo, spgemm_descr, 0, null_ptr)
        buff1 = _cupy.empty(buff1_size, _cupy.int8)
        _cusparse.spGEMM_workEstimation(
            handle, op_a, op_b, alpha.data, mat_a.desc, mat_b.desc, beta.data,
            mat_c.desc, cuda_dtype, algo, spgemm_descr, buff1_size,
            buff1.data.ptr)

    except _cusparse.CuSparseError as cse:
        # If the memory required is too high and cuSPARSE >= 12.0, fall back
        # to ALG2
        if getVersion() < 12000:
            raise cse
        algo = _cusparse.CUSPARSE_SPGEMM_ALG2
        buff1_size = _cusparse.spGEMM_workEstimation(
            handle, op_a, op_b, alpha.data, mat_a.desc, mat_b.desc, beta.data,
            mat_c.desc, cuda_dtype, algo, spgemm_descr, 0, null_ptr)
        buff1 = _cupy.empty(buff1_size, _cupy.int8)
        _cusparse.spGEMM_workEstimation(
            handle, op_a, op_b, alpha.data, mat_a.desc, mat_b.desc, beta.data,
            mat_c.desc, cuda_dtype, algo, spgemm_descr, buff1_size,
            buff1.data.ptr)

    # Compute the intermediate product of A and B
    buff2_size = _cusparse.spGEMM_compute(
        handle, op_a, op_b, alpha.data, mat_a.desc, mat_b.desc, beta.data,
        mat_c.desc, cuda_dtype, algo, spgemm_descr, 0, null_ptr)
    buff2 = _cupy.empty(buff2_size, _cupy.int8)
    _cusparse.spGEMM_compute(
        handle, op_a, op_b, alpha.data, mat_a.desc, mat_b.desc, beta.data,
        mat_c.desc, cuda_dtype, algo, spgemm_descr, buff2_size, buff2.data.ptr)

    # Prepare the arrays for matrix C
    c_num_rows = _numpy.array(0, dtype='int64')
    c_num_cols = _numpy.array(0, dtype='int64')
    c_nnz = _numpy.array(0, dtype='int64')
    _cusparse.spMatGetSize(mat_c.desc, c_num_rows.ctypes.data,
                           c_num_cols.ctypes.data, c_nnz.ctypes.data)
    assert c_shape[0] == int(c_num_rows)
    assert c_shape[1] == int(c_num_cols)
    c_nnz = int(c_nnz)
    c_indptr = c.indptr
    c_indices = _cupy.empty(c_nnz, idx_dtype)
    c_data = _cupy.empty(c_nnz, c.dtype)
    _cusparse.csrSetPointers(mat_c.desc, c_indptr.data.ptr, c_indices.data.ptr,
                             c_data.data.ptr)

    # Copy the final product to the matrix C
    _cusparse.spGEMM_copy(
        handle, op_a, op_b, alpha.data, mat_a.desc, mat_b.desc, beta.data,
        mat_c.desc, cuda_dtype, algo, spgemm_descr)
    c = cupyx.scipy.sparse.csr_matrix._from_parts(
        c_data, c_indices, c_indptr, c_shape,
        has_canonical_format=True, has_sorted_indices=True)

    _cusparse.spGEMM_destroyDescr(spgemm_descr)
    return c

import atexit
import collections

import numpy
import six

import cupy
from cupy import cuda
from cupy.cuda import cusparse


_handles = collections.defaultdict(cusparse.create)


def get_handle():
    dev = cuda.get_device_id()
    return _handles[dev]


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


def _call_cusparse(name, dtype, *args):
    if dtype == 'f':
        prefix = 's'
    elif dtype == 'd':
        prefix = 'd'
    else:
        raise TypeError
    f = getattr(cusparse, prefix + name)
    return f(*args)


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
    _call_cusparse(
        'gthr', x.dtype,
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


def csr2coo(x, data, indices):
    """Convert a CSR-matrix to COO format.

    Args:
        x (cupy.sparse.csr_matrix): A matrix to be converted.
        data (cupy.ndarray): A data array for converted data.
        indices (cupy.ndarray): A index array for converted data.

    Returns:
        cupy.sparse.coo_matrix: Returns a converted matrix.

    """
    m = x.shape[0]
    nnz = len(x.data)
    row = cupy.empty(nnz, 'i')
    cusparse.xcsr2coo(
        get_handle(), x.indptr.data.ptr, nnz, m, row.data.ptr,
        cusparse.CUSPARSE_INDEX_BASE_ZERO)
    # data and indices did not need to be copied already
    return cupy.sparse.coo_matrix(
        (data, (row, indices)), shape=x.shape)


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

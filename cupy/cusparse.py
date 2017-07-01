import cupy
from cupy.cuda import cusparse
from cupy.cuda import device


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


def _call_cusparse(name, dtype, *args):
    if dtype == 'f':
        prefix = 's'
    elif dtype == 'd':
        prefix = 'd'
    else:
        raise TypeError
    f = getattr(cusparse, prefix + name)
    return f(*args)


def csr2dense(x, out=None):
    """Converts CSR-matrix to a dense matrix.

    Args:
        x (cupy.sparse.csr_matrix): A sparse matrix to convert.
        out (cupy.ndarray or None): A dense metrix to store the result.
            It must be F-contiguous.

    Returns:
        cupy.ndarray: Converted result.

    """
    dtype = x.dtype
    assert dtype == 'f' or dtype == 'd'
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


def csrsort(x):
    """Sorts indices of CSR-matrix in place.

    Args:
        x (cupy.sparse.csr_matrix): A sparse matrix to sort.

    """
    handle = device.get_cusparse_handle()
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
    """Sorts indices of CSC-matrix in place.

    Args:
        x (cupy.sparse.csc_matrix): A sparse matrix to sort.

    """
    handle = device.get_cusparse_handle()
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
    handle = device.get_cusparse_handle()
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
    handle = device.get_cusparse_handle()
    m = x.shape[0]
    indptr = cupy.empty(m + 1, 'i')
    cusparse.xcoo2csr(
        handle, x.row.data.ptr, x.nnz, m,
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
    return cupy.sparse.csc_matrix((data, indices, indptr), shape=x.shape)

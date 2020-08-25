import numpy

try:
    import scipy.sparse
    _scipy_available = True
except ImportError:
    _scipy_available = False

import cupy
from cupy.core import _accelerator
from cupy.cuda import cub
from cupy import cusparse
from cupyx.scipy.sparse import base
from cupyx.scipy.sparse import compressed
from cupyx.scipy.sparse import csc
from cupyx.scipy.sparse import _index
from cupyx.scipy.sparse import util


class csr_matrix(compressed._compressed_sparse_matrix):

    """Compressed Sparse Row matrix.

    Now it has only part of initializer formats:

    ``csr_matrix(D)``
        ``D`` is a rank-2 :class:`cupy.ndarray`.
    ``csr_matrix(S)``
        ``S`` is another sparse matrix. It is equivalent to ``S.tocsr()``.
    ``csr_matrix((M, N), [dtype])``
        It constructs an empty matrix whose shape is ``(M, N)``. Default dtype
        is float64.
    ``csr_matrix((data, indices, indptr))``
        All ``data``, ``indices`` and ``indptr`` are one-dimenaional
        :class:`cupy.ndarray`.

    Args:
        arg1: Arguments for the initializer.
        shape (tuple): Shape of a matrix. Its length must be two.
        dtype: Data type. It must be an argument of :class:`numpy.dtype`.
        copy (bool): If ``True``, copies of given arrays are always used.

    .. seealso::
        :class:`scipy.sparse.csr_matrix`

    """

    format = 'csr'

    def get(self, stream=None):
        """Returns a copy of the array on host memory.

        Args:
            stream (cupy.cuda.Stream): CUDA stream object. If it is given, the
                copy runs asynchronously. Otherwise, the copy is synchronous.

        Returns:
            scipy.sparse.csr_matrix: Copy of the array on host memory.

        """
        if not _scipy_available:
            raise RuntimeError('scipy is not available')
        data = self.data.get(stream)
        indices = self.indices.get(stream)
        indptr = self.indptr.get(stream)
        return scipy.sparse.csr_matrix(
            (data, indices, indptr), shape=self._shape)

    def _convert_dense(self, x):
        m = cusparse.dense2csr(x)
        return m.data, m.indices, m.indptr

    def _swap(self, x, y):
        return (x, y)

    def _add_sparse(self, other, alpha, beta):
        self.sum_duplicates()
        other = other.tocsr()
        other.sum_duplicates()
        if cusparse.check_availability('csrgeam2'):
            csrgeam = cusparse.csrgeam2
        elif cusparse.check_availability('csrgeam'):
            csrgeam = cusparse.csrgeam
        else:
            raise NotImplementedError
        return csrgeam(self, other, alpha, beta)

    def __eq__(self, other):
        raise NotImplementedError

    def __ne__(self, other):
        raise NotImplementedError

    def __lt__(self, other):
        raise NotImplementedError

    def __gt__(self, other):
        raise NotImplementedError

    def __le__(self, other):
        raise NotImplementedError

    def __ge__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        if cupy.isscalar(other):
            self.sum_duplicates()
            return self._with_data(self.data * other)
        elif isspmatrix_csr(other):
            self.sum_duplicates()
            other.sum_duplicates()
            if cusparse.check_availability('csrgemm2'):
                return cusparse.csrgemm2(self, other)
            elif cusparse.check_availability('csrgemm'):
                return cusparse.csrgemm(self, other)
            else:
                raise NotImplementedError
        elif csc.isspmatrix_csc(other):
            self.sum_duplicates()
            other.sum_duplicates()
            if cusparse.check_availability('csrgemm'):
                return cusparse.csrgemm(self, other.T, transb=True)
            elif cusparse.check_availability('csrgemm2'):
                b = other.tocsr()
                b.sum_duplicates()
                return cusparse.csrgemm2(self, b)
            else:
                raise NotImplementedError
        elif base.isspmatrix(other):
            return self * other.tocsr()
        elif base.isdense(other):
            if other.ndim == 0:
                self.sum_duplicates()
                return self._with_data(self.data * other)
            elif other.ndim == 1:
                self.sum_duplicates()
                other = cupy.asfortranarray(other)
                # need extra padding to ensure not stepping on the CUB bug,
                # see cupy/cupy#3679 for discussion
                is_cub_safe = (self.indptr.data.mem.size
                               > self.indptr.size * self.indptr.dtype.itemsize)
                for accelerator in _accelerator.get_routine_accelerators():
                    if (accelerator == _accelerator.ACCELERATOR_CUB
                            and is_cub_safe and other.flags.c_contiguous):
                        return cub.device_csrmv(
                            self.shape[0], self.shape[1], self.nnz,
                            self.data, self.indptr, self.indices, other)
                if (cusparse.check_availability('csrmvEx') and self.nnz > 0 and
                        cusparse.csrmvExIsAligned(self, other)):
                    # csrmvEx does not work if nnz == 0
                    csrmv = cusparse.csrmvEx
                elif cusparse.check_availability('csrmv'):
                    csrmv = cusparse.csrmv
                elif cusparse.check_availability('spmv'):
                    csrmv = cusparse.spmv
                else:
                    raise NotImplementedError
                return csrmv(self, other)
            elif other.ndim == 2:
                self.sum_duplicates()
                if cusparse.check_availability('csrmm2'):
                    csrmm = cusparse.csrmm2
                elif cusparse.check_availability('spmm'):
                    csrmm = cusparse.spmm
                else:
                    raise NotImplementedError
                return csrmm(self, cupy.asfortranarray(other))
            else:
                raise ValueError('could not interpret dimensions')
        else:
            return NotImplemented

    def __div__(self, other):
        raise NotImplementedError

    def __rdiv__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        """Point-wise division by scalar"""
        if util.isscalarlike(other):
            if self.dtype == numpy.complex64:
                # Note: This is a work-around to make the output dtype the same
                # as SciPy. It might be SciPy version dependent.
                dtype = numpy.float32
            else:
                if cupy.isscalar(other):
                    dtype = numpy.float64
                else:
                    dtype = numpy.promote_types(numpy.float64, other.dtype)
            d = cupy.array(1. / other, dtype=dtype)
            return multiply_by_scalar(self, d)
        # TODO(anaruse): Implement divide by dense or sparse matrix
        raise NotImplementedError

    def __rtruediv__(self, other):
        raise NotImplementedError

    # TODO(unno): Implement check_format

    def diagonal(self, k=0):
        # TODO(unno): Implement diagonal
        raise NotImplementedError

    def eliminate_zeros(self):
        """Removes zero entories in place."""
        compress = cusparse.csr2csr_compress(self, 0)
        self.data = compress.data
        self.indices = compress.indices
        self.indptr = compress.indptr

    def maximum(self, other):
        # TODO(unno): Implement maximum
        raise NotImplementedError

    def minimum(self, other):
        # TODO(unno): Implement minimum
        raise NotImplementedError

    def multiply(self, other):
        """Point-wise multiplication by another matrix, vector or scalar"""
        if cupy.isscalar(other):
            return multiply_by_scalar(self, other)
        elif util.isdense(other):
            self.sum_duplicates()
            other = cupy.atleast_2d(other)
            return multiply_by_dense(self, other)
        elif isspmatrix_csr(other):
            self.sum_duplicates()
            other.sum_duplicates()
            return multiply_by_csr(self, other)
        else:
            msg = 'expected scalar, dense matrix/vector or csr matrixr'
            raise TypeError(msg)

    # TODO(unno): Implement prune
    # TODO(unno): Implement reshape

    def sort_indices(self):
        """Sorts the indices of this matrix *in place*.

        .. warning::
            Calling this function might synchronize the device.

        """
        if not self.has_sorted_indices:
            cusparse.csrsort(self)
            self.has_sorted_indices = True

    def toarray(self, order=None, out=None):
        """Returns a dense matrix representing the same value.

        Args:
            order ({'C', 'F', None}): Whether to store data in C (row-major)
                order or F (column-major) order. Default is C-order.
            out: Not supported.

        Returns:
            cupy.ndarray: Dense array representing the same matrix.

        .. seealso:: :meth:`scipy.sparse.csr_matrix.toarray`

        """
        order = 'C' if order is None else order.upper()
        if self.nnz == 0:
            return cupy.zeros(shape=self.shape, dtype=self.dtype, order=order)

        x = self.copy()
        x.has_canonical_format = False  # need to enforce sum_duplicates
        x.sum_duplicates()
        # csr2dense returns F-contiguous array.
        if order == 'C':
            # To return C-contiguous array, it uses transpose.
            return cusparse.csc2dense(x.T).T
        elif order == 'F':
            return cusparse.csr2dense(x)
        else:
            raise ValueError('order not understood')

    def tobsr(self, blocksize=None, copy=False):
        # TODO(unno): Implement tobsr
        raise NotImplementedError

    def tocoo(self, copy=False):
        """Converts the matrix to COOdinate format.

        Args:
            copy (bool): If ``False``, it shares data arrays as much as
                possible.

        Returns:
            cupyx.scipy.sparse.coo_matrix: Converted matrix.

        """
        if copy:
            data = self.data.copy()
            indices = self.indices.copy()
        else:
            data = self.data
            indices = self.indices

        return cusparse.csr2coo(self, data, indices)

    def tocsc(self, copy=False):
        """Converts the matrix to Compressed Sparse Column format.

        Args:
            copy (bool): If ``False``, it shares data arrays as much as
                possible. Actually this option is ignored because all
                arrays in a matrix cannot be shared in csr to csc conversion.

        Returns:
            cupyx.scipy.sparse.csc_matrix: Converted matrix.

        """
        # copy is ignored
        if cusparse.check_availability('csr2csc'):
            csr2csc = cusparse.csr2csc
        elif cusparse.check_availability('csr2cscEx2'):
            csr2csc = cusparse.csr2cscEx2
        else:
            raise NotImplementedError
        # don't touch has_sorted_indices, as cuSPARSE made no guarantee
        return csr2csc(self)

    def tocsr(self, copy=False):
        """Converts the matrix to Compressed Sparse Row format.

        Args:
            copy (bool): If ``False``, the method returns itself.
                Otherwise it makes a copy of the matrix.

        Returns:
            cupyx.scipy.sparse.csr_matrix: Converted matrix.

        """
        if copy:
            return self.copy()
        else:
            return self

    def todia(self, copy=False):
        # TODO(unno): Implement todia
        raise NotImplementedError

    def todok(self, copy=False):
        # TODO(unno): Implement todok
        raise NotImplementedError

    def tolil(self, copy=False):
        # TODO(unno): Implement tolil
        raise NotImplementedError

    def transpose(self, axes=None, copy=False):
        """Returns a transpose matrix.

        Args:
            axes: This option is not supported.
            copy (bool): If ``True``, a returned matrix shares no data.
                Otherwise, it shared data arrays as much as possible.

        Returns:
            cupyx.scipy.sparse.spmatrix: Transpose matrix.

        """
        if axes is not None:
            raise ValueError(
                'Sparse matrices do not support an \'axes\' parameter because '
                'swapping dimensions is the only logical permutation.')

        shape = self.shape[1], self.shape[0]
        trans = csc.csc_matrix(
            (self.data, self.indices, self.indptr), shape=shape, copy=copy)
        trans.has_canonical_format = self.has_canonical_format
        return trans

    def getrow(self, i):
        """Returns a copy of row i of the matrix, as a (1 x n)
        CSR matrix (row vector).

        Args:
            i (integer): Row

        Returns:
            cupyx.scipy.sparse.csr_matrix: Sparse matrix with single row
        """
        M, N = self.shape
        i = _index._normalize_index(i, M, 'index')
        indptr, indices, data = _index._get_csr_submatrix_major_axis(
            self.indptr, self.indices, self.data, i, i + 1)
        return csr_matrix((data, indices, indptr), shape=(1, N),
                          dtype=self.dtype, copy=False)

    def getcol(self, i):
        """Returns a copy of column i of the matrix, as a (m x 1)
        CSR matrix (column vector).

        Args:
            i (integer): Column

        Returns:
            cupyx.scipy.sparse.csr_matrix: Sparse matrix with single column
        """
        M, N = self.shape
        i = _index._normalize_index(i, N, 'index')
        indptr, indices, data = _index._get_csr_submatrix_minor_axis(
            self.indptr, self.indices, self.data, i, i + 1)
        return csr_matrix((data, indices, indptr), shape=(M, 1),
                          dtype=self.dtype, copy=False)

    def _get_intXarray(self, row, col):
        return self.getrow(row)._minor_index_fancy(col)

    def _get_intXslice(self, row, col):
        if col.step in (1, None):
            return self._get_submatrix(slice(row, row+1, 1), col, copy=True)

        M, N = self.shape
        start, stop, stride = col.indices(N)

        ii, jj = self.indptr[row:row+2]
        row_indices = self.indices[ii:jj]
        row_data = self.data[ii:jj]

        if stride > 0:
            ind = (row_indices >= start) & (row_indices < stop)
        else:
            ind = (row_indices <= start) & (row_indices > stop)

        if abs(stride) > 1:
            ind &= (row_indices - start) % stride == 0

        row_indices = (row_indices[ind] - start) // stride
        row_data = row_data[ind]
        row_indptr = cupy.array([0, row_indices.size])

        if stride < 0:
            row_data = row_data[::-1]
            row_indices = cupy.abs(row_indices[::-1])

        shape = (1, (stop - start + stride - 1) // stride)
        return csr_matrix((row_data, row_indices, row_indptr), shape=shape,
                          dtype=self.dtype, copy=False)

    def _get_sliceXint(self, row, col):
        if row.step in (1, None):
            return self._get_submatrix(row, slice(col, col+1, 1), copy=True)
        return self._major_slice(row)._get_submatrix(
            minor=slice(col, col+1, 1))

    def _get_sliceXarray(self, row, col):
        return self._major_slice(row)._minor_index_fancy(col)

    def _get_arrayXint(self, row, col):
        return self._major_index_fancy(row)._get_submatrix(minor=col)

    def _get_arrayXslice(self, row, col):
        if col.step not in (1, None):
            col = cupy.arange(*col.indices(self.shape[1]))
            return self._get_arrayXarray(row, col)
        return self._major_index_fancy(row)._get_submatrix(minor=col)


def isspmatrix_csr(x):
    """Checks if a given matrix is of CSR format.

    Returns:
        bool: Returns if ``x`` is :class:`cupyx.scipy.sparse.csr_matrix`.

    """
    return isinstance(x, csr_matrix)


def multiply_by_scalar(sp, a):
    data = sp.data * a
    indices = sp.indices.copy()
    indptr = sp.indptr.copy()
    return csr_matrix((data, indices, indptr), shape=sp.shape)


def multiply_by_dense(sp, dn):
    sp_m, sp_n = sp.shape
    dn_m, dn_n = dn.shape
    if not (sp_m == dn_m or sp_m == 1 or dn_m == 1):
        raise ValueError('inconsistent shape')
    if not (sp_n == dn_n or sp_n == 1 or dn_n == 1):
        raise ValueError('inconsistent shape')
    m, n = max(sp_m, dn_m), max(sp_n, dn_n)
    nnz = sp.nnz * (m // sp_m) * (n // sp_n)
    dtype = numpy.promote_types(sp.dtype, dn.dtype)
    data = cupy.empty(nnz, dtype=dtype)
    indices = cupy.empty(nnz, dtype=sp.indices.dtype)
    if m > sp_m:
        if n > sp_n:
            indptr = cupy.arange(0, nnz+1, n, dtype=sp.indptr.dtype)
        else:
            indptr = cupy.arange(0, nnz+1, sp.nnz, dtype=sp.indptr.dtype)
    else:
        indptr = sp.indptr.copy()
        if n > sp_n:
            indptr *= n

    # out = sp * dn
    cupy_multiply_by_dense()(sp.data, sp.indptr, sp.indices, sp_m, sp_n,
                             dn, dn_m, dn_n, indptr, m, n, data, indices)

    return csr_matrix((data, indices, indptr), shape=(m, n))


@cupy.util.memoize(for_each_device=True)
def cupy_multiply_by_dense():
    return cupy.ElementwiseKernel(
        '''
        raw S SP_DATA, raw I SP_INDPTR, raw I SP_INDICES,
        int32 SP_M, int32 SP_N,
        raw D DN_DATA, int32 DN_M, int32 DN_N,
        raw I OUT_INDPTR, int32 OUT_M, int32 OUT_N
        ''',
        'O OUT_DATA, I OUT_INDICES',
        '''
        int i_out = i;
        int _min = 0;
        int _max = OUT_M - 1;
        int m_out = (_min + _max) / 2;
        while (_min < _max) {
            if (i_out < OUT_INDPTR[m_out]) {
                _max = m_out - 1;
            }
            else if (i_out >= OUT_INDPTR[m_out+1]) {
                _min = m_out + 1;
            }
            else {
                break;
            }
            m_out = (_min + _max) / 2;
        }
        int i_sp = i_out;
        if (OUT_M > SP_M && SP_M == 1) {
            i_sp -= OUT_INDPTR[m_out];
        }
        if (OUT_N > SP_N && SP_N == 1) {
            i_sp /= OUT_N;
        }
        int n_out = SP_INDICES[i_sp];
        if (OUT_N > SP_N && SP_N == 1) {
            n_out = i_out - OUT_INDPTR[m_out];
        }
        int m_dn = m_out;
        if (OUT_M > DN_M && DN_M == 1) {
            m_dn = 0;
        }
        int n_dn = n_out;
        if (OUT_N > DN_N && DN_N == 1) {
            n_dn = 0;
        }
        OUT_DATA = (O)(SP_DATA[i_sp] * DN_DATA[n_dn + (DN_N * m_dn)]);
        OUT_INDICES = n_out;
        ''',
        'cupy_multiply_by_dense'
    )


def multiply_by_csr(a, b):
    a_m, a_n = a.shape
    b_m, b_n = b.shape
    if not (a_m == b_m or a_m == 1 or b_m == 1):
        raise ValueError('inconsistent shape')
    if not (a_n == b_n or a_n == 1 or b_n == 1):
        raise ValueError('inconsistent shape')
    m, n = max(a_m, b_m), max(a_n, b_n)
    a_nnz = a.nnz * (m // a_m) * (n // a_n)
    b_nnz = b.nnz * (m // b_m) * (n // b_n)
    if a_nnz > b_nnz:
        return multiply_by_csr(b, a)
    c_nnz = a_nnz
    dtype = numpy.promote_types(a.dtype, b.dtype)
    c_data = cupy.empty(c_nnz, dtype=dtype)
    c_indices = cupy.empty(c_nnz, dtype=a.indices.dtype)
    if m > a_m:
        if n > a_n:
            c_indptr = cupy.arange(0, c_nnz+1, n, dtype=a.indptr.dtype)
        else:
            c_indptr = cupy.arange(0, c_nnz+1, a.nnz, dtype=a.indptr.dtype)
    else:
        c_indptr = a.indptr.copy()
        if n > a_n:
            c_indptr *= n
    flags = cupy.zeros(c_nnz+1, dtype=a.indices.dtype)
    nnz_each_row = cupy.zeros(m+1, dtype=a.indptr.dtype)

    # compute c = a * b where necessary and get sparsity pattern of matrix d
    cupy_multiply_by_csr_step1()(
        a.data, a.indptr, a.indices, a_m, a_n,
        b.data, b.indptr, b.indices, b_m, b_n,
        c_indptr, m, n, c_data, c_indices, flags, nnz_each_row)

    flags = cupy.cumsum(flags, dtype=a.indptr.dtype)
    d_indptr = cupy.cumsum(nnz_each_row, dtype=a.indptr.dtype)
    d_nnz = int(d_indptr[-1])
    d_data = cupy.empty(d_nnz, dtype=dtype)
    d_indices = cupy.empty(d_nnz, dtype=a.indices.dtype)

    # remove zero elements in matric c
    cupy_multiply_by_csr_step2()(c_data, c_indices, flags, d_data, d_indices)

    return csr_matrix((d_data, d_indices, d_indptr), shape=(m, n))


@cupy.util.memoize(for_each_device=True)
def cupy_multiply_by_csr_step1():
    return cupy.ElementwiseKernel(
        '''
        raw A A_DATA, raw I A_INDPTR, raw I A_INDICES, int32 A_M, int32 A_N,
        raw B B_DATA, raw I B_INDPTR, raw I B_INDICES, int32 B_M, int32 B_N,
        raw I C_INDPTR, int32 C_M, int32 C_N
        ''',
        'C C_DATA, I C_INDICES, raw I FLAGS, raw I NNZ_EACH_ROW',
        '''
        int i_c = i;
        int _min = 0;
        int _max = C_M - 1;
        int m_c;
        while (_min <= _max) {
            m_c = (_min + _max) / 2;
            if (i_c < C_INDPTR[m_c]) {
                _max = m_c - 1;
            }
            else if (i_c >= C_INDPTR[m_c+1]) {
                _min = m_c + 1;
            }
            else {
                break;
            }
        }
        int i_a = i;
        if (C_M > A_M && A_M == 1) {
            i_a -= C_INDPTR[m_c];
        }
        if (C_N > A_N && A_N == 1) {
            i_a /= C_N;
        }
        int n_c = A_INDICES[i_a];
        if (C_N > A_N && A_N == 1) {
            n_c = i % C_N;
        }
        int m_b = m_c;
        if (C_M > B_M && B_M == 1) {
            m_b = 0;
        }
        int n_b = n_c;
        if (C_N > B_N && B_N == 1) {
            n_b = 0;
        }
        int i_b = -1;
        int j_min = B_INDPTR[m_b];
        int j_max = B_INDPTR[m_b+1] - 1;
        while (j_min <= j_max) {
            int j = (j_min + j_max) / 2;
            if (n_b < B_INDICES[j]) {
                j_max = j - 1;
            }
            else if (n_b > B_INDICES[j]) {
                j_min = j + 1;
            }
            else {
                i_b = j;
                break;
            }
        }
        if (i_b >= 0) {
            atomicAdd(&(NNZ_EACH_ROW[m_c+1]), 1);
            FLAGS[i+1] = 1;
            C_DATA = (C)(A_DATA[i_a] * B_DATA[i_b]);
            C_INDICES = n_c;
        }
        ''',
        'cupy_multiply_by_csr_step1'
    )


@cupy.util.memoize(for_each_device=True)
def cupy_multiply_by_csr_step2():
    return cupy.ElementwiseKernel(
        'T C_DATA, I C_INDICES, raw I FLAGS',
        'raw D D_DATA, raw I D_INDICES',
        '''
        int j = FLAGS[i];
        if (j < FLAGS[i+1]) {
            D_DATA[j] = (D)(C_DATA);
            D_INDICES[j] = C_INDICES;
        }
        ''',
        'cupy_multiply_by_csr_step2'
    )

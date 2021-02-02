import numpy
import operator

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
from cupyx.scipy.sparse import _util


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
        m = dense2csr(x)
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

    def _comparison(self, other, op, op_name):
        if _util.isscalarlike(other):
            data = cupy.asarray(other, dtype=self.dtype).reshape(1)
            if numpy.isnan(data[0]):
                if op_name == '_ne_':
                    return csr_matrix(cupy.ones(self.shape, dtype=numpy.bool_))
                else:
                    return csr_matrix(self.shape, dtype=numpy.bool_)
            indices = cupy.zeros((1,), dtype=numpy.int32)
            indptr = cupy.arange(2, dtype=numpy.int32)
            other = csr_matrix((data, indices, indptr), shape=(1, 1))
            return binopt_csr(self, other, op_name)
        elif _util.isdense(other):
            return op(self.todense(), other)
        elif isspmatrix_csr(other):
            self.sum_duplicates()
            other.sum_duplicates()
            if op_name in ('_ne_', '_lt_', '_gt_'):
                return binopt_csr(self, other, op_name)

            if op_name == '_eq_':
                opposite_op_name = '_ne_'
            elif op_name == '_le_':
                opposite_op_name = '_gt_'
            elif op_name == '_ge_':
                opposite_op_name = '_lt_'
            res = binopt_csr(self, other, opposite_op_name)
            out = cupy.logical_not(res.toarray())
            return csr_matrix(out)
        raise NotImplementedError

    def __eq__(self, other):
        return self._comparison(other, operator.eq, '_eq_')

    def __ne__(self, other):
        return self._comparison(other, operator.ne, '_ne_')

    def __lt__(self, other):
        return self._comparison(other, operator.lt, '_lt_')

    def __gt__(self, other):
        return self._comparison(other, operator.gt, '_gt_')

    def __le__(self, other):
        return self._comparison(other, operator.le, '_le_')

    def __ge__(self, other):
        return self._comparison(other, operator.ge, '_ge_')

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
        """Point-wise division by another matrix, vector or scalar"""
        if _util.isscalarlike(other):
            dtype = self.dtype
            if dtype == numpy.float32:
                # Note: This is a work-around to make the output dtype the same
                # as SciPy. It might be SciPy version dependent.
                dtype = numpy.float64
            dtype = cupy.result_type(dtype, other)
            d = cupy.reciprocal(other, dtype=dtype)
            return multiply_by_scalar(self, d)
        elif _util.isdense(other):
            other = cupy.atleast_2d(other)
            check_shape_for_pointwise_op(self.shape, other.shape)
            return self.todense() / other
        elif base.isspmatrix(other):
            # Note: If broadcasting is needed, an exception is raised here for
            # compatibility with SciPy, as SciPy does not support broadcasting
            # in the "sparse / sparse" case.
            check_shape_for_pointwise_op(self.shape, other.shape,
                                         allow_broadcasting=False)
            dtype = numpy.promote_types(self.dtype, other.dtype)
            if dtype.char not in 'FD':
                dtype = numpy.promote_types(numpy.float64, dtype)
            # Note: The following implementation converts two sparse matrices
            # into dense matrices and then performs a point-wise division,
            # which can use lots of memory.
            self_dense = self.todense().astype(dtype, copy=False)
            return self_dense / other.todense()
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

    def _maximum_minimum(self, other, cupy_op, op_name, dense_check):
        if _util.isscalarlike(other):
            other = cupy.asarray(other, dtype=self.dtype)
            if dense_check(other):
                dtype = self.dtype
                # Note: This is a work-around to make the output dtype the same
                # as SciPy. It might be SciPy version dependent.
                if dtype == numpy.float32:
                    dtype = numpy.float64
                elif dtype == numpy.complex64:
                    dtype = numpy.complex128
                dtype = cupy.result_type(dtype, other)
                other = other.astype(dtype, copy=False)
                # Note: The computation steps below are different from SciPy.
                new_array = cupy_op(self.todense(), other)
                return csr_matrix(new_array)
            else:
                self.sum_duplicates()
                new_data = cupy_op(self.data, other)
                return csr_matrix((new_data, self.indices, self.indptr),
                                  shape=self.shape, dtype=self.dtype)
        elif _util.isdense(other):
            self.sum_duplicates()
            other = cupy.atleast_2d(other)
            return cupy_op(self.todense(), other)
        elif isspmatrix_csr(other):
            self.sum_duplicates()
            other.sum_duplicates()
            return binopt_csr(self, other, op_name)
        raise NotImplementedError

    def maximum(self, other):
        return self._maximum_minimum(other, cupy.maximum, '_maximum_',
                                     lambda x: x > 0)

    def minimum(self, other):
        return self._maximum_minimum(other, cupy.minimum, '_minimum_',
                                     lambda x: x < 0)

    def multiply(self, other):
        """Point-wise multiplication by another matrix, vector or scalar"""
        if cupy.isscalar(other):
            return multiply_by_scalar(self, other)
        elif _util.isdense(other):
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

        if self.dtype.char not in 'fdFD':
            return csr2dense(self, order)

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

    def _tocsx(self):
        """Inverts the format.
        """
        return self.tocsc()

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
        return self._major_slice(slice(i, i + 1), copy=True)

    def getcol(self, i):
        """Returns a copy of column i of the matrix, as a (m x 1)
        CSR matrix (column vector).

        Args:
            i (integer): Column

        Returns:
            cupyx.scipy.sparse.csr_matrix: Sparse matrix with single column
        """
        return self._minor_slice(slice(i, i + 1), copy=True)

    def _get_intXarray(self, row, col):
        row = slice(row, row + 1)
        return self._major_slice(row)._minor_index_fancy(col)

    def _get_intXslice(self, row, col):
        row = slice(row, row + 1)
        return self._major_slice(row)._minor_slice(col, copy=True)

    def _get_sliceXint(self, row, col):
        col = slice(col, col + 1)
        copy = row.step in (1, None)
        return self._major_slice(row)._minor_slice(col, copy=copy)

    def _get_sliceXarray(self, row, col):
        return self._major_slice(row)._minor_index_fancy(col)

    def _get_arrayXint(self, row, col):
        col = slice(col, col + 1)
        return self._major_index_fancy(row)._minor_slice(col)

    def _get_arrayXslice(self, row, col):
        if col.step not in (1, None):
            start, stop, step = col.indices(self.shape[1])
            cols = cupy.arange(start, stop, step, self.indices.dtype)
            return self._get_arrayXarray(row, cols)
        return self._major_index_fancy(row)._minor_slice(col)


def isspmatrix_csr(x):
    """Checks if a given matrix is of CSR format.

    Returns:
        bool: Returns if ``x`` is :class:`cupyx.scipy.sparse.csr_matrix`.

    """
    return isinstance(x, csr_matrix)


def check_shape_for_pointwise_op(a_shape, b_shape, allow_broadcasting=True):
    if allow_broadcasting:
        a_m, a_n = a_shape
        b_m, b_n = b_shape
        if not (a_m == b_m or a_m == 1 or b_m == 1):
            raise ValueError('inconsistent shape')
        if not (a_n == b_n or a_n == 1 or b_n == 1):
            raise ValueError('inconsistent shape')
    else:
        if a_shape != b_shape:
            raise ValueError('inconsistent shape')


def multiply_by_scalar(sp, a):
    data = sp.data * a
    indices = sp.indices.copy()
    indptr = sp.indptr.copy()
    return csr_matrix((data, indices, indptr), shape=sp.shape)


def multiply_by_dense(sp, dn):
    check_shape_for_pointwise_op(sp.shape, dn.shape)
    sp_m, sp_n = sp.shape
    dn_m, dn_n = dn.shape
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


_GET_ROW_ID_ = '''
__device__ inline int get_row_id(int i, int min, int max, const int *indptr) {
    int row = (min + max) / 2;
    while (min < max) {
        if (i < indptr[row]) {
            max = row - 1;
        } else if (i >= indptr[row + 1]) {
            min = row + 1;
        } else {
            break;
        }
        row = (min + max) / 2;
    }
    return row;
}
'''


@cupy._util.memoize(for_each_device=True)
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
        int m_out = get_row_id(i_out, 0, OUT_M - 1, &(OUT_INDPTR[0]));
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
        'cupy_multiply_by_dense',
        preamble=_GET_ROW_ID_
    )


def multiply_by_csr(a, b):
    check_shape_for_pointwise_op(a.shape, b.shape)
    a_m, a_n = a.shape
    b_m, b_n = b.shape
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


@cupy._util.memoize(for_each_device=True)
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
        int m_c = get_row_id(i_c, 0, C_M - 1, &(C_INDPTR[0]));

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
        'cupy_multiply_by_csr_step1',
        preamble=_GET_ROW_ID_
    )


@cupy._util.memoize(for_each_device=True)
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


_BINOPT_MAX_ = '''
__device__ inline O binopt(T in1, T in2) {
    return max(in1, in2);
}
'''
_BINOPT_MIN_ = '''
__device__ inline O binopt(T in1, T in2) {
    return min(in1, in2);
}
'''
_BINOPT_EQ_ = '''
__device__ inline O binopt(T in1, T in2) {
    return (in1 == in2);
}
'''
_BINOPT_NE_ = '''
__device__ inline O binopt(T in1, T in2) {
    return (in1 != in2);
}
'''
_BINOPT_LT_ = '''
__device__ inline O binopt(T in1, T in2) {
    return (in1 < in2);
}
'''
_BINOPT_GT_ = '''
__device__ inline O binopt(T in1, T in2) {
    return (in1 > in2);
}
'''
_BINOPT_LE_ = '''
__device__ inline O binopt(T in1, T in2) {
    return (in1 <= in2);
}
'''
_BINOPT_GE_ = '''
__device__ inline O binopt(T in1, T in2) {
    return (in1 >= in2);
}
'''


def binopt_csr(a, b, op_name):
    check_shape_for_pointwise_op(a.shape, b.shape)
    a_m, a_n = a.shape
    b_m, b_n = b.shape
    m, n = max(a_m, b_m), max(a_n, b_n)
    a_nnz = a.nnz * (m // a_m) * (n // a_n)
    b_nnz = b.nnz * (m // b_m) * (n // b_n)

    a_info = cupy.zeros(a_nnz + 1, dtype=a.indices.dtype)
    b_info = cupy.zeros(b_nnz + 1, dtype=b.indices.dtype)
    a_valid = cupy.zeros(a_nnz, dtype=numpy.int8)
    b_valid = cupy.zeros(b_nnz, dtype=numpy.int8)
    c_indptr = cupy.zeros(m + 1, dtype=a.indptr.dtype)
    in_dtype = numpy.promote_types(a.dtype, b.dtype)
    a_data = a.data.astype(in_dtype, copy=False)
    b_data = b.data.astype(in_dtype, copy=False)
    funcs = _GET_ROW_ID_
    if op_name == '_maximum_':
        funcs += _BINOPT_MAX_
        out_dtype = in_dtype
    elif op_name == '_minimum_':
        funcs += _BINOPT_MIN_
        out_dtype = in_dtype
    elif op_name == '_eq_':
        funcs += _BINOPT_EQ_
        out_dtype = numpy.bool_
    elif op_name == '_ne_':
        funcs += _BINOPT_NE_
        out_dtype = numpy.bool_
    elif op_name == '_lt_':
        funcs += _BINOPT_LT_
        out_dtype = numpy.bool_
    elif op_name == '_gt_':
        funcs += _BINOPT_GT_
        out_dtype = numpy.bool_
    elif op_name == '_le_':
        funcs += _BINOPT_LE_
        out_dtype = numpy.bool_
    elif op_name == '_ge_':
        funcs += _BINOPT_GE_
        out_dtype = numpy.bool_
    else:
        raise ValueError('invalid op_name: {}'.format(op_name))
    a_tmp_data = cupy.empty(a_nnz, dtype=out_dtype)
    b_tmp_data = cupy.empty(b_nnz, dtype=out_dtype)
    a_tmp_indices = cupy.empty(a_nnz, dtype=a.indices.dtype)
    b_tmp_indices = cupy.empty(b_nnz, dtype=b.indices.dtype)
    _size = a_nnz + b_nnz
    cupy_binopt_csr_step1(op_name, preamble=funcs)(
        m, n,
        a.indptr, a.indices, a_data, a_m, a_n, a.nnz, a_nnz,
        b.indptr, b.indices, b_data, b_m, b_n, b.nnz, b_nnz,
        a_info, a_valid, a_tmp_indices, a_tmp_data,
        b_info, b_valid, b_tmp_indices, b_tmp_data,
        c_indptr, size=_size)
    a_info = cupy.cumsum(a_info, dtype=a_info.dtype)
    b_info = cupy.cumsum(b_info, dtype=b_info.dtype)
    c_indptr = cupy.cumsum(c_indptr, dtype=c_indptr.dtype)
    c_nnz = int(c_indptr[-1])
    c_indices = cupy.empty(c_nnz, dtype=a.indices.dtype)
    c_data = cupy.empty(c_nnz, dtype=out_dtype)
    cupy_binopt_csr_step2(op_name)(
        a_info, a_valid, a_tmp_indices, a_tmp_data, a_nnz,
        b_info, b_valid, b_tmp_indices, b_tmp_data, b_nnz,
        c_indices, c_data, size=_size)
    return csr_matrix((c_data, c_indices, c_indptr), shape=(m, n))


@cupy._util.memoize(for_each_device=True)
def cupy_binopt_csr_step1(op_name, preamble=''):
    name = 'cupy_binopt_csr' + op_name + 'step1'
    return cupy.ElementwiseKernel(
        '''
        int32 M, int32 N,
        raw I A_INDPTR, raw I A_INDICES, raw T A_DATA,
        int32 A_M, int32 A_N, int32 A_NNZ_ACT, int32 A_NNZ,
        raw I B_INDPTR, raw I B_INDICES, raw T B_DATA,
        int32 B_M, int32 B_N, int32 B_NNZ_ACT, int32 B_NNZ
        ''',
        '''
        raw I A_INFO, raw B A_VALID, raw I A_TMP_INDICES, raw O A_TMP_DATA,
        raw I B_INFO, raw B B_VALID, raw I B_TMP_INDICES, raw O B_TMP_DATA,
        raw I C_INFO
        ''',
        '''
        if (i >= A_NNZ + B_NNZ) return;

        const int *MY_INDPTR, *MY_INDICES;  int *MY_INFO;  const T *MY_DATA;
        const int *OP_INDPTR, *OP_INDICES;  int *OP_INFO;  const T *OP_DATA;
        int MY_M, MY_N, MY_NNZ_ACT, MY_NNZ;
        int OP_M, OP_N, OP_NNZ_ACT, OP_NNZ;
        signed char *MY_VALID;  I *MY_TMP_INDICES;  O *MY_TMP_DATA;

        int my_j;
        if (i < A_NNZ) {
            // in charge of one of non-zero element of sparse matrix A
            my_j = i;
            MY_INDPTR  = &(A_INDPTR[0]);   OP_INDPTR  = &(B_INDPTR[0]);
            MY_INDICES = &(A_INDICES[0]);  OP_INDICES = &(B_INDICES[0]);
            MY_INFO    = &(A_INFO[0]);     OP_INFO    = &(B_INFO[0]);
            MY_DATA    = &(A_DATA[0]);     OP_DATA    = &(B_DATA[0]);
            MY_M       = A_M;              OP_M       = B_M;
            MY_N       = A_N;              OP_N       = B_N;
            MY_NNZ_ACT = A_NNZ_ACT;        OP_NNZ_ACT = B_NNZ_ACT;
            MY_NNZ     = A_NNZ;            OP_NNZ     = B_NNZ;
            MY_VALID   = &(A_VALID[0]);
            MY_TMP_DATA= &(A_TMP_DATA[0]);
            MY_TMP_INDICES = &(A_TMP_INDICES[0]);
        } else {
            // in charge of one of non-zero element of sparse matrix B
            my_j = i - A_NNZ;
            MY_INDPTR  = &(B_INDPTR[0]);   OP_INDPTR  = &(A_INDPTR[0]);
            MY_INDICES = &(B_INDICES[0]);  OP_INDICES = &(A_INDICES[0]);
            MY_INFO    = &(B_INFO[0]);     OP_INFO    = &(A_INFO[0]);
            MY_DATA    = &(B_DATA[0]);     OP_DATA    = &(A_DATA[0]);
            MY_M       = B_M;              OP_M       = A_M;
            MY_N       = B_N;              OP_N       = A_N;
            MY_NNZ_ACT = B_NNZ_ACT;        OP_NNZ_ACT = A_NNZ_ACT;
            MY_NNZ     = B_NNZ;            OP_NNZ     = A_NNZ;
            MY_VALID   = &(B_VALID[0]);
            MY_TMP_DATA= &(B_TMP_DATA[0]);
            MY_TMP_INDICES = &(B_TMP_INDICES[0]);
        }
        int _min, _max, _mid;

        // get column location
        int my_col;
        int my_j_act = my_j;
        if (MY_M == 1 && MY_M < M) {
            if (MY_N == 1 && MY_N < N) my_j_act = 0;
            else                       my_j_act = my_j % MY_NNZ_ACT;
        } else {
            if (MY_N == 1 && MY_N < N) my_j_act = my_j / N;
        }
        my_col = MY_INDICES[my_j_act];
        if (MY_N == 1 && MY_N < N) {
            my_col = my_j % N;
        }

        // get row location
        int my_row = get_row_id(my_j_act, 0, MY_M - 1, &(MY_INDPTR[0]));
        if (MY_M == 1 && MY_M < M) {
            if (MY_N == 1 && MY_N < N) my_row = my_j / N;
            else                       my_row = my_j / MY_NNZ_ACT;
        }

        int op_row = my_row;
        int op_row_act = op_row;
        if (OP_M == 1 && OP_M < M) {
            op_row_act = 0;
        }

        int op_col = 0;
        _min = OP_INDPTR[op_row_act];
        _max = OP_INDPTR[op_row_act + 1] - 1;
        int op_j_act = _min;
        bool op_nz = false;
        if (_min <= _max) {
            if (OP_N == 1 && OP_N < N) {
                op_col = my_col;
                op_nz = true;
            }
            else {
                _mid = (_min + _max) / 2;
                op_col = OP_INDICES[_mid];
                while (_min < _max) {
                    if (op_col < my_col) {
                        _min = _mid + 1;
                    } else if (op_col > my_col) {
                        _max = _mid;
                    } else {
                        break;
                    }
                    _mid = (_min + _max) / 2;
                    op_col = OP_INDICES[_mid];
                }
                op_j_act = _mid;
                if (op_col == my_col) {
                    op_nz = true;
                } else if (op_col < my_col) {
                    op_col = N;
                    op_j_act += 1;
                }
            }
        }

        int op_j = op_j_act;
        if (OP_M == 1 && OP_M < M) {
            if (OP_N == 1 && OP_N < N) {
                op_j = (op_col + N * op_row) * OP_NNZ_ACT;
            } else {
                op_j = op_j_act + OP_NNZ_ACT * op_row;
            }
        } else {
            if (OP_N == 1 && OP_N < N) {
                op_j = op_col + N * op_j_act;
            }
        }

        if (i < A_NNZ || !op_nz) {
            T my_data = MY_DATA[my_j_act];
            T op_data = 0;
            if (op_nz) op_data = OP_DATA[op_j_act];
            O out;
            if (i < A_NNZ) out = binopt(my_data, op_data);
            else           out = binopt(op_data, my_data);
            if (out != static_cast<O>(0)) {
                MY_VALID[my_j] = 1;
                MY_TMP_DATA[my_j] = out;
                MY_TMP_INDICES[my_j] = my_col;
                atomicAdd( &(C_INFO[my_row + 1]), 1 );
                atomicAdd( &(MY_INFO[my_j + 1]), 1 );
                atomicAdd( &(OP_INFO[op_j]), 1 );
            }
        }
        ''',
        name, preamble=preamble,
    )


@cupy._util.memoize(for_each_device=True)
def cupy_binopt_csr_step2(op_name):
    name = 'cupy_binopt_csr' + op_name + 'step2'
    return cupy.ElementwiseKernel(
        '''
        raw I A_INFO, raw B A_VALID, raw I A_TMP_INDICES, raw O A_TMP_DATA,
        int32 A_NNZ,
        raw I B_INFO, raw B B_VALID, raw I B_TMP_INDICES, raw O B_TMP_DATA,
        int32 B_NNZ
        ''',
        'raw I C_INDICES, raw O C_DATA',
        '''
        if (i < A_NNZ) {
            int j = i;
            if (A_VALID[j]) {
                C_INDICES[A_INFO[j]] = A_TMP_INDICES[j];
                C_DATA[A_INFO[j]]    = A_TMP_DATA[j];
            }
        } else if (i < A_NNZ + B_NNZ) {
            int j = i - A_NNZ;
            if (B_VALID[j]) {
                C_INDICES[B_INFO[j]] = B_TMP_INDICES[j];
                C_DATA[B_INFO[j]]    = B_TMP_DATA[j];
            }
        }
        ''',
        name,
    )


def csr2dense(a, order):
    out = cupy.zeros(a.shape, dtype=a.dtype, order=order)
    m, n = a.shape
    cupy_csr2dense()(m, n, a.indptr, a.indices, a.data,
                     (order == 'C'), out)
    return out


@cupy._util.memoize(for_each_device=True)
def cupy_csr2dense():
    return cupy.ElementwiseKernel(
        'int32 M, int32 N, raw I INDPTR, I INDICES, T DATA, bool C_ORDER',
        'raw T OUT',
        '''
        int row = get_row_id(i, 0, M - 1, &(INDPTR[0]));
        int col = INDICES;
        if (C_ORDER) {
            OUT[col + N * row] += DATA;
        } else {
            OUT[row + M * col] += DATA;
        }
        ''',
        'cupy_csr2dense',
        preamble=_GET_ROW_ID_
    )


def dense2csr(a):
    if a.dtype.char in 'fdFD':
        return cusparse.dense2csr(a)
    m, n = a.shape
    a = cupy.ascontiguousarray(a)
    indptr = cupy.zeros(m + 1, dtype=numpy.int32)
    info = cupy.zeros(m * n + 1, dtype=numpy.int32)
    cupy_dense2csr_step1()(m, n, a, indptr, info)
    indptr = cupy.cumsum(indptr, dtype=numpy.int32)
    info = cupy.cumsum(info, dtype=numpy.int32)
    nnz = int(indptr[-1])
    indices = cupy.empty(nnz, dtype=numpy.int32)
    data = cupy.empty(nnz, dtype=a.dtype)
    cupy_dense2csr_step2()(m, n, a, info, indices, data)
    return csr_matrix((data, indices, indptr), shape=(m, n))


@cupy._util.memoize(for_each_device=True)
def cupy_dense2csr_step1():
    return cupy.ElementwiseKernel(
        'int32 M, int32 N, T A',
        'raw I INDPTR, raw I INFO',
        '''
        int row = i / N;
        int col = i % N;
        if (A != static_cast<T>(0)) {
            atomicAdd( &(INDPTR[row + 1]), 1 );
            INFO[i + 1] = 1;
        }
        ''',
        'cupy_dense2csr_step1')


@cupy._util.memoize(for_each_device=True)
def cupy_dense2csr_step2():
    return cupy.ElementwiseKernel(
        'int32 M, int32 N, T A, raw I INFO',
        'raw I INDICES, raw T DATA',
        '''
        int row = i / N;
        int col = i % N;
        if (A != static_cast<T>(0)) {
            int idx = INFO[i];
            INDICES[idx] = col;
            DATA[idx] = A;
        }
        ''',
        'cupy_dense2csr_step2')

import numpy
try:
    import scipy.sparse
    scipy_available = True
except ImportError:
    scipy_available = False

import cupy
from cupy import core
from cupy._creation import basic
from cupy import cusparse
from cupyx.scipy.sparse import base
from cupyx.scipy.sparse import data as sparse_data
from cupyx.scipy.sparse import util

from cupyx.scipy.sparse import _index


class _compressed_sparse_matrix(sparse_data._data_matrix,
                                sparse_data._minmax_mixin,
                                _index.IndexMixin):

    _compress_getitem_kern = core.ElementwiseKernel(
        'T d, S ind, int32 minor', 'raw T answer',
        'if (ind == minor) atomicAdd(&answer[0], d);',
        'compress_getitem')

    _compress_getitem_complex_kern = core.ElementwiseKernel(
        'T real, T imag, S ind, int32 minor',
        'raw T answer_real, raw T answer_imag',
        '''
        if (ind == minor) {
          atomicAdd(&answer_real[0], real);
          atomicAdd(&answer_imag[0], imag);
        }
        ''',
        'compress_getitem_complex')

    _max_reduction_kern = core.RawKernel(r'''
        extern "C" __global__
        void max_reduction(double* data, int* x, int* y, int length,
                           double* z) {
            // Get the index of the block
            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            // Calculate the block length
            int block_length = y[tid] - x[tid];

            // Select initial value based on the block density
            double running_value = 0;
            if (block_length == length){
                running_value = data[x[tid]];
            } else {
                running_value = 0;
            }

            // Iterate over the block and update
            for (int entry = x[tid]; entry < y[tid]; entry++){
                if (data[entry] != data[entry]){
                    // Check for NaN
                    running_value = nan("");
                    break;
                } else {
                    // Check for a value update
                    if (data[entry] > running_value){
                        running_value = data[entry];
                    }
                }
            }

            // Store in the return function
            z[tid] = running_value;
        }
        ''', 'max_reduction')

    _max_nonzero_reduction_kern = core.RawKernel(r'''
        extern "C" __global__
        void max_nonzero_reduction(double* data, int* x, int* y, int length,
                                   double* z) {
            // Get the index of the block
            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            // Calculate the block length
            int block_length = y[tid] - x[tid];

            // Select initial value based on the block density
            double running_value = 0;
            if (block_length > 0){
                running_value = data[x[tid]];
            } else {
                running_value = 0;
            }

            // Iterate over the section of the sparse matrix
            for (int entry = x[tid]; entry < y[tid]; entry++){
                if (data[entry] != data[entry]){
                    // Check for NaN
                    running_value = nan("");
                    break;
                } else {
                    // Check for a value update
                    if (running_value < data[entry]){
                        running_value = data[entry];
                    }
                }
            }

            // Store in the return function
            z[tid] = running_value;
        }
        ''', 'max_nonzero_reduction')

    _min_reduction_kern = core.RawKernel(r'''
        extern "C" __global__
        void min_reduction(double* data, int* x, int* y, int length,
                           double* z) {
            // Get the index of the block
            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            // Calculate the block length
            int block_length = y[tid] - x[tid];

            // Select initial value based on the block density
            double running_value = 0;
            if (block_length == length){
                running_value = data[x[tid]];
            } else {
                running_value = 0;
            }

            // Iterate over the block to update the initial value
            for (int entry = x[tid]; entry < y[tid]; entry++){
                if (data[entry] != data[entry]){
                    // Check for NaN
                    running_value = nan("");
                    break;
                } else {
                    // Check for a value update
                    if (data[entry] < running_value){
                        running_value = data[entry];
                    }
                }
            }

            // Store in the return function
            z[tid] = running_value;
        }
        ''', 'min_reduction')

    _min_nonzero_reduction_kern = core.RawKernel(r'''
        extern "C" __global__
        void min_nonzero_reduction(double* data, int* x, int* y, int length,
                                   double* z) {
            // Get the index of hte block
            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            // Calculate the block length
            int block_length = y[tid] - x[tid];

            // Select initial value based on the block density
            double running_value = 0;
            if (block_length > 0){
                running_value = data[x[tid]];
            } else {
                running_value = 0;
            }

            // Iterate over the section of the sparse matrix
            for (int entry = x[tid]; entry < y[tid]; entry++){
                if (data[entry] != data[entry]){
                    // Check for NaN
                    running_value = nan("");
                    break;
                } else {
                    // Check for a value update
                    if (running_value > data[entry]){
                        running_value = data[entry];
                    }
                }
            }

            // Store in the return function
            z[tid] = running_value;
        }
        ''', 'min_nonzero_reduction')

    _max_arg_reduction_kern = core.RawKernel(r'''
        extern "C" __global__
        void max_arg_reduction(double* data, int* indices, int* x, int* y,
                               int length, long long* z) {
            // Get the index of the block
            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            // Calculate the block length
            int block_length = y[tid] - x[tid];

            // Select initial value based on the block density
            int data_index = 0;
            double data_value = 0;

            if (block_length == length){
                // Block is dense. Fill the first value
                data_value = data[x[tid]];
                data_index = indices[x[tid]];
            } else if (block_length > 0)  {
                // Block has at least one zero. Assign first occurrence as the
                // starting reference
                data_value = 0;
                for (data_index = 0; data_index < length; data_index++){
                    if (data_index != indices[x[tid] + data_index] ||
                        x[tid] + data_index >= y[tid]){
                        break;
                    }
                }
            } else {
                 // Zero valued array
                data_value = 0;
                data_index = 0;
            }

            // Iterate over the section of the sparse matrix
            for (int entry = x[tid]; entry < y[tid]; entry++){
                if (data[entry] != data[entry]){
                    // Check for NaN
                    data_value = nan("");
                    data_index = 0;
                    break;
                } else {
                    // Check for a value update
                    if (data[entry] > data_value){
                        data_index = indices[entry];
                        data_value = data[entry];
                    }
                }
            }

            // Store in the return function
            z[tid] = data_index;
        }
        ''', 'max_arg_reduction')

    _min_arg_reduction_kern = core.RawKernel(r'''
        extern "C" __global__
        void min_arg_reduction(double* data, int* indices, int* x, int* y,
                               int length, long long* z) {
            // Get the index of hte block
            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            // Calculate the block length
            int block_length = y[tid] - x[tid];

            // Select initial value based on the block density
            int data_index = 0;
            double data_value = 0;

            if (block_length == length){
                // Block is dense. Fill the first value
                data_value = data[x[tid]];
                data_index = indices[x[tid]];
            } else if (block_length > 0)  {
                // Block has at least one zero. Assign first occurrence as the
                // starting reference
                data_value = 0;
                for (data_index = 0; data_index < length; data_index++){
                    if (data_index != indices[x[tid] + data_index] ||
                        x[tid] + data_index >= y[tid]){
                        break;
                    }
                }
            } else {
                // Zero valued array
                data_value = 0;
                data_index = 0;
            }

            // Iterate over the section of the sparse matrix
            for (int entry = x[tid]; entry < y[tid]; entry++){
                if (data[entry] != data[entry]){
                    // Check for NaN
                    data_value = nan("");
                    data_index = 0;
                    break;
                } else {
                    // Check for a value update
                    if (data[entry] < data_value){
                        data_index = indices[entry];
                        data_value = data[entry];
                    }
                }
            }

            // Store in the return function
            z[tid] = data_index;

        }
        ''', 'min_arg_reduction')

    # TODO(leofang): rewrite a more load-balanced approach than this naive one?
    _has_sorted_indices_kern = core.ElementwiseKernel(
        'raw T indptr, raw T indices',
        'bool diff',
        '''
        bool diff_out = true;
        for (T jj = indptr[i]; jj < indptr[i+1] - 1; jj++) {
            if (indices[jj] > indices[jj+1]){
                diff_out = false;
            }
        }
        diff = diff_out;
        ''', 'has_sorted_indices')

    # TODO(leofang): rewrite a more load-balanced approach than this naive one?
    _has_canonical_format_kern = core.ElementwiseKernel(
        'raw T indptr, raw T indices',
        'bool diff',
        '''
        bool diff_out = true;
        if (indptr[i] > indptr[i+1]) {
            diff = false;
            return;
        }
        for (T jj = indptr[i]; jj < indptr[i+1] - 1; jj++) {
            if (indices[jj] >= indices[jj+1]) {
                diff_out = false;
            }
        }
        diff = diff_out;
        ''', 'has_canonical_format')

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        if shape is not None:
            if not util.isshape(shape):
                raise ValueError('invalid shape (must be a 2-tuple of int)')
            shape = int(shape[0]), int(shape[1])

        if base.issparse(arg1):
            x = arg1.asformat(self.format)
            data = x.data
            indices = x.indices
            indptr = x.indptr

            if arg1.format != self.format:
                # When formats are differnent, all arrays are already copied
                copy = False

            if shape is None:
                shape = arg1.shape

        elif util.isshape(arg1):
            m, n = arg1
            m, n = int(m), int(n)
            data = basic.zeros(0, dtype if dtype else 'd')
            indices = basic.zeros(0, 'i')
            indptr = basic.zeros(self._swap(m, n)[0] + 1, dtype='i')
            # shape and copy argument is ignored
            shape = (m, n)
            copy = False

        elif scipy_available and scipy.sparse.issparse(arg1):
            # Convert scipy.sparse to cupyx.scipy.sparse
            x = arg1.asformat(self.format)
            data = cupy.array(x.data)
            indices = cupy.array(x.indices, dtype='i')
            indptr = cupy.array(x.indptr, dtype='i')
            copy = False

            if shape is None:
                shape = arg1.shape

        elif isinstance(arg1, tuple) and len(arg1) == 3:
            data, indices, indptr = arg1
            if not (base.isdense(data) and data.ndim == 1 and
                    base.isdense(indices) and indices.ndim == 1 and
                    base.isdense(indptr) and indptr.ndim == 1):
                raise ValueError(
                    'data, indices, and indptr should be 1-D')

            if len(data) != len(indices):
                raise ValueError('indices and data should have the same size')

        elif base.isdense(arg1):
            if arg1.ndim > 2:
                raise TypeError('expected dimension <= 2 array or matrix')
            elif arg1.ndim == 1:
                arg1 = arg1[None]
            elif arg1.ndim == 0:
                arg1 = arg1[None, None]
            data, indices, indptr = self._convert_dense(arg1)
            copy = False
            if shape is None:
                shape = arg1.shape

        else:
            raise ValueError(
                'Unsupported initializer format')

        if dtype is None:
            dtype = data.dtype
        else:
            dtype = numpy.dtype(dtype)

        if dtype != 'f' and dtype != 'd' and dtype != 'F' and dtype != 'D':
            raise ValueError(
                'Only float32, float64, complex64 and complex128 '
                'are supported')

        data = data.astype(dtype, copy=copy)
        sparse_data._data_matrix.__init__(self, data)

        self.indices = indices.astype('i', copy=copy)
        self.indptr = indptr.astype('i', copy=copy)

        if shape is None:
            shape = self._swap(len(indptr) - 1, int(indices.max()) + 1)

        major, minor = self._swap(*shape)
        if len(indptr) != major + 1:
            raise ValueError('index pointer size (%d) should be (%d)'
                             % (len(indptr), major + 1))

        self._descr = cusparse.MatDescriptor.create()
        self._shape = shape

    def _with_data(self, data, copy=True):
        if copy:
            return self.__class__(
                (data, self.indices.copy(), self.indptr.copy()),
                shape=self.shape,
                dtype=data.dtype)
        else:
            return self.__class__(
                (data, self.indices, self.indptr),
                shape=self.shape,
                dtype=data.dtype)

    def _convert_dense(self, x):
        raise NotImplementedError

    def _swap(self, x, y):
        raise NotImplementedError

    def _add_sparse(self, other, alpha, beta):
        raise NotImplementedError

    def _add(self, other, lhs_negative, rhs_negative):
        if cupy.isscalar(other):
            if other == 0:
                if lhs_negative:
                    return -self
                else:
                    return self.copy()
            else:
                raise NotImplementedError(
                    'adding a nonzero scalar to a sparse matrix is not '
                    'supported')
        elif base.isspmatrix(other):
            alpha = -1 if lhs_negative else 1
            beta = -1 if rhs_negative else 1
            return self._add_sparse(other, alpha, beta)
        elif base.isdense(other):
            if lhs_negative:
                if rhs_negative:
                    return -self.todense() - other
                else:
                    return other - self.todense()
            else:
                if rhs_negative:
                    return self.todense() - other
                else:
                    return self.todense() + other
        else:
            return NotImplemented

    def __add__(self, other):
        return self._add(other, False, False)

    def __radd__(self, other):
        return self._add(other, False, False)

    def __sub__(self, other):
        return self._add(other, False, True)

    def __rsub__(self, other):
        return self._add(other, True, False)

    def _get_intXint(self, row, col):
        major, minor = self._swap(row, col)

        indptr, indices, data = _index._get_csr_submatrix_major_axis(
            self.indptr, self.indices, self.data, major, major + 1)
        indptr, indices, data = _index._get_csr_submatrix_minor_axis(
            indptr, indices, data, minor, minor + 1)
        return data.sum(dtype=self.dtype)

    def _get_sliceXslice(self, row, col):
        major, minor = self._swap(row, col)
        if major.step in (1, None) and minor.step in (1, None):
            return self._get_submatrix(major, minor, copy=True)
        return self._major_slice(major)._minor_slice(minor)

    def _get_arrayXarray(self, row, col):
        # inner indexing
        idx_dtype = self.indices.dtype
        M, N = self._swap(*self.shape)
        major, minor = self._swap(row, col)
        major = cupy.asarray(major, dtype=idx_dtype)
        minor = cupy.asarray(minor, dtype=idx_dtype)

        val = _index._csr_sample_values(
            M, N, self.indptr, self.indices, self.data,
            major.ravel(), minor.ravel())

        if major.ndim == 1:
            # Scipy returns `matrix` here
            return cupy.expand_dims(val, 0)
        return self.__class__(val.reshape(major.shape))

    def _get_columnXarray(self, row, col):
        # outer indexing
        major, minor = self._swap(row, col)
        return self._major_index_fancy(major)._minor_index_fancy(minor)

    def _major_index_fancy(self, idx):
        """Index along the major axis where idx is an array of ints.
        """
        _, N = self._swap(*self.shape)
        M = len(idx)
        new_shape = self._swap(M, N)
        if M == 0:
            return self.__class__(new_shape)

        row_nnz = cupy.diff(self.indptr)
        idx_dtype = self.indices.dtype
        res_indptr = cupy.zeros(M+1, dtype=idx_dtype)
        cupy.cumsum(row_nnz[idx], out=res_indptr[1:])

        res_indices, res_data = _index._csr_row_index(
            idx, self.indptr,
            self.indices, self.data, res_indptr)

        return self.__class__((res_data, res_indices, res_indptr),
                              shape=new_shape, copy=False)

    def _minor_index_fancy(self, idx):
        """Index along the minor axis where idx is an array of ints.
        """

        idx_dtype = self.indices.dtype
        idx = cupy.asarray(idx, dtype=idx_dtype).ravel()

        M, N = self._swap(*self.shape)
        k = len(idx)
        new_shape = self._swap(M, k)
        if k == 0:
            return self.__class__(new_shape)

        # pass 1: count idx entries and compute new indptr
        col_order = cupy.argsort(idx).astype(idx_dtype, copy=False)

        index1_outs = _index._csr_column_index1(idx, self.indptr, self.indices)
        res_indptr, indices_mask, col_counts, sort_idxs = index1_outs

        # pass 2: copy indices/data for selected idxs

        res_indices, res_data = _index._csr_column_index2(
            col_order, col_counts, sort_idxs, self.indptr, indices_mask,
            self.data, res_indptr)

        return self.__class__((res_data, res_indices, res_indptr),
                              shape=new_shape, copy=False)

    def _minor_slice(self, idx, copy=False):
        """Index along the minor axis where idx is a slice object.
        """

        if idx == slice(None):
            return self.copy() if copy else self

        M, N = self._swap(*self.shape)
        start, stop, step = idx.indices(N)
        N = len(range(start, stop, step))
        if N == 0:
            return self.__class__(self._swap(M, N))
        if step == 1:
            return self._get_submatrix(minor=idx, copy=copy)
        return self._minor_index_fancy(cupy.arange(start, stop, step))

    @staticmethod
    def _process_slice(sl, num):
        if sl is None:
            i0, i1 = 0, num
        elif isinstance(sl, slice):
            i0, i1, stride = sl.indices(num)
            if stride != 1:
                raise ValueError('slicing with step != 1 not supported')
            i0 = min(i0, i1)  # give an empty slice when i0 > i1

        # Scipy calls sputils.isintlike(). Comparing directly to int
        # here to minimize the impact of nested exception catching
        elif isinstance(sl, _index._int_scalar_types):
            if sl < 0:
                sl += num
            i0, i1 = sl, sl + 1
            if i0 < 0 or i1 > num:
                raise IndexError('index out of bounds: 0 <= %d < %d <= %d' %
                                 (i0, i1, num))
        else:
            raise TypeError('expected slice or scalar')

        return i0, i1

    def _get_single(self, major, minor):
        start = self.indptr[major]
        end = self.indptr[major + 1]
        answer = cupy.zeros((), self.dtype)
        data = self.data[start:end]
        indices = self.indices[start:end]
        if self.dtype.kind == 'c':
            self._compress_getitem_complex_kern(
                data.real, data.imag, indices, minor, answer.real, answer.imag)
        else:
            self._compress_getitem_kern(
                data, indices, minor, answer)
        return answer[()]

    def _get_submatrix(self, major=None, minor=None, copy=False):
        """Return a submatrix of this matrix.
        major, minor: None or slice with step 1
        """
        M, N = self._swap(*self.shape)
        i0, i1 = self._process_slice(major, M)
        j0, j1 = self._process_slice(minor, N)

        is_major_full = i0 == 0 and i1 == M
        is_minor_full = j0 == 0 and j1 == N

        if is_major_full and is_minor_full:
            return self.copy() if copy else self

        indptr, indices, data = self.indptr, self.indices, self.data

        if not is_major_full:
            indptr, indices, data = _index._get_csr_submatrix_major_axis(
                indptr, indices, data, i0, i1)
        if not is_minor_full:
            indptr, indices, data = _index._get_csr_submatrix_minor_axis(
                indptr, indices, data, j0, j1)

        shape = self._swap(i1 - i0, j1 - j0)
        return self.__class__((data, indices, indptr), shape=shape,
                              dtype=self.dtype, copy=False)

    def _major_slice(self, idx, copy=False):
        """Index along the major axis where idx is a slice object.
        """

        if idx == slice(None):
            return self.copy() if copy else self

        M, N = self._swap(*self.shape)
        start, stop, step = idx.indices(M)
        M = len(range(start, stop, step))
        new_shape = self._swap(M, N)
        if M == 0:
            return self.__class__(new_shape)

        row_nnz = cupy.diff(self.indptr)
        idx_dtype = self.indices.dtype
        res_indptr = cupy.zeros(M+1, dtype=idx_dtype)

        cupy.cumsum(row_nnz[idx], out=res_indptr[1:])

        if step == 1:
            idx_start = self.indptr[start]
            idx_stop = self.indptr[stop]
            res_indices = cupy.array(self.indices[idx_start:idx_stop],
                                     copy=copy)
            res_data = cupy.array(self.data[idx_start:idx_stop], copy=copy)
        else:
            res_indices, res_data = _index._csr_row_slice(
                start, step, self.indptr, self.indices, self.data, res_indptr)

        return self.__class__((res_data, res_indices, res_indptr),
                              shape=new_shape, copy=False)

    def __get_has_canonical_format(self):
        """Determine whether the matrix has sorted indices and no duplicates.

        Returns
            bool: ``True`` if the above applies, otherwise ``False``.

        .. note::
            :attr:`has_canonical_format` implies :attr:`has_sorted_indices`, so
            if the latter flag is ``False``, so will the former be; if the
            former is found ``True``, the latter flag is also set.

        .. warning::
            Getting this property might synchronize the device.

        """
        # Modified from the SciPy counterpart.

        # In CuPy the implemented conversions do not exactly match those of
        # SciPy's, so it's hard to put this exactly as where it is in SciPy,
        # but this should do the job.
        if self.data.size == 0:
            self._has_canonical_format = True
        # check to see if result was cached
        elif not getattr(self, '_has_sorted_indices', True):
            # not sorted => not canonical
            self._has_canonical_format = False
        elif not hasattr(self, '_has_canonical_format'):
            is_canonical = self._has_canonical_format_kern(
                self.indptr, self.indices, size=self.indptr.size-1)
            self._has_canonical_format = bool(is_canonical.all())
        return self._has_canonical_format

    def __set_has_canonical_format(self, val):
        """Taken from SciPy as is."""
        self._has_canonical_format = bool(val)
        if val:
            self.has_sorted_indices = True

    has_canonical_format = property(fget=__get_has_canonical_format,
                                    fset=__set_has_canonical_format)

    def __get_sorted(self):
        """Determine whether the matrix has sorted indices.

        Returns
            bool:
                ``True`` if the indices of the matrix are in sorted order,
                otherwise ``False``.

        .. warning::
            Getting this property might synchronize the device.

        """
        # Modified from the SciPy counterpart.

        # In CuPy the implemented conversions do not exactly match those of
        # SciPy's, so it's hard to put this exactly as where it is in SciPy,
        # but this should do the job.
        if self.data.size == 0:
            self._has_sorted_indices = True
        # check to see if result was cached
        elif not hasattr(self, '_has_sorted_indices'):
            is_sorted = self._has_sorted_indices_kern(
                self.indptr, self.indices, size=self.indptr.size-1)
            self._has_sorted_indices = bool(is_sorted.all())
        return self._has_sorted_indices

    def __set_sorted(self, val):
        self._has_sorted_indices = bool(val)

    has_sorted_indices = property(fget=__get_sorted, fset=__set_sorted)

    def get_shape(self):
        """Returns the shape of the matrix.

        Returns:
            tuple: Shape of the matrix.

        """
        return self._shape

    def getnnz(self, axis=None):
        """Returns the number of stored values, including explicit zeros.

        Args:
            axis: Not supported yet.

        Returns:
            int: The number of stored values.

        """
        if axis is None:
            return self.data.size
        else:
            raise ValueError

    def sorted_indices(self):
        """Return a copy of this matrix with sorted indices

        .. warning::
            Calling this function might synchronize the device.

        """
        # Taken from SciPy as is.
        A = self.copy()
        A.sort_indices()
        return A

    def sort_indices(self):
        # Unlike in SciPy, here this is implemented in child classes because
        # each child needs to call its own sort function from cuSPARSE
        raise NotImplementedError

    def sum_duplicates(self):
        """Eliminate duplicate matrix entries by adding them together.

        .. note::
            This is an *in place* operation.

        .. warning::
            Calling this function might synchronize the device.

        .. seealso::
           :meth:`scipy.sparse.csr_matrix.sum_duplicates`,
           :meth:`scipy.sparse.csc_matrix.sum_duplicates`

        """
        if self.has_canonical_format:
            return
        # TODO(leofang): add a kernel for compressed sparse matrices without
        # converting to coo
        coo = self.tocoo()
        coo.sum_duplicates()
        self.__init__(coo.asformat(self.format))
        self.has_canonical_format = True

    #####################
    # Reduce operations #
    #####################

    def _minor_reduce(self, ufunc, axis, nonzero):
        """Reduce nonzeros with a ufunc over the minor axis when non-empty

        Can be applied to a function of self.data by supplying data parameter.
        Warning: this does not call sum_duplicates()

        Args:
            ufunc (object): Function handle giving the operation to be
                conducted.
            axis (int): Matrix over which the reduction should be
                conducted.

        Returns:
            (cupy.ndarray): Reduce result for nonzeros in each
                major_index.

        """

        # Call to the appropriate kernel function
        if axis == 1:
            # Create the vector to hold output
            value = cupy.zeros(self.shape[0]).astype(cupy.float64)

            if nonzero:
                # Perform the calculation
                if ufunc == cupy.amax:
                    self._max_nonzero_reduction_kern(
                        (self.shape[0],), (1,),
                        (self.data.astype(cupy.float64),
                         self.indptr[:len(self.indptr) - 1],
                         self.indptr[1:], cupy.int64(self.shape[1]),
                         value))
                if ufunc == cupy.amin:
                    self._min_nonzero_reduction_kern(
                        (self.shape[0],), (1,),
                        (self.data.astype(cupy.float64),
                         self.indptr[:len(self.indptr) - 1],
                         self.indptr[1:], cupy.int64(self.shape[1]),
                         value))

            else:
                # Perform the calculation
                if ufunc == cupy.amax:
                    self._max_reduction_kern(
                        (self.shape[0],), (1,),
                        (self.data.astype(cupy.float64),
                         self.indptr[:len(self.indptr) - 1],
                         self.indptr[1:], cupy.int64(self.shape[1]),
                         value))
                if ufunc == cupy.amin:
                    self._min_reduction_kern(
                        (self.shape[0],), (1,),
                        (self.data.astype(cupy.float64),
                         self.indptr[:len(self.indptr) - 1],
                         self.indptr[1:], cupy.int64(self.shape[1]),
                         value))

        if axis == 0:
            # Create the vector to hold output
            value = cupy.zeros(self.shape[1]).astype(cupy.float64)

            if nonzero:
                # Perform the calculation
                if ufunc == cupy.amax:
                    self._max_nonzero_reduction_kern(
                        (self.shape[1],), (1,),
                        (self.data.astype(cupy.float64),
                         self.indptr[:len(self.indptr) - 1],
                         self.indptr[1:], cupy.int64(self.shape[0]),
                         value))
                if ufunc == cupy.amin:
                    self._min_nonzero_reduction_kern(
                        (self.shape[1],), (1,),
                        (self.data.astype(cupy.float64),
                         self.indptr[:len(self.indptr) - 1],
                         self.indptr[1:], cupy.int64(self.shape[0]),
                         value))
            else:
                # Perform the calculation
                if ufunc == cupy.amax:
                    self._max_reduction_kern(
                        (self.shape[1],), (1,),
                        (self.data.astype(cupy.float64),
                         self.indptr[:len(self.indptr) - 1],
                         self.indptr[1:], cupy.int64(self.shape[0]),
                         value))
                if ufunc == cupy.amin:
                    self._min_reduction_kern(
                        (self.shape[1],), (1,),
                        (self.data.astype(cupy.float64),
                         self.indptr[:len(self.indptr) - 1],
                         self.indptr[1:], cupy.int64(self.shape[0]),
                         value))

        return value

    def _arg_minor_reduce(self, ufunc, axis):
        """Reduce nonzeros with a ufunc over the minor axis when non-empty

        Can be applied to a function of self.data by supplying data parameter.
        Warning: this does not call sum_duplicates()

        Args:
            ufunc (object): Function handle giving the operation to be
                conducted.
            axis (int): Maxtrix over which the reduction should be conducted

        Returns:
            (cupy.ndarray): Reduce result for nonzeros in each
                major_index

        """

        # Call to the appropriate kernel function
        if axis == 1:
            # Create the vector to hold output
            value = cupy.zeros(self.shape[0]).astype(cupy.int64)

            # Perform the calculation
            if ufunc == cupy.argmax:
                self._max_arg_reduction_kern(
                    (self.shape[0],), (1,),
                    (self.data.astype(cupy.float64), self.indices,
                     self.indptr[:len(self.indptr) - 1],
                     self.indptr[1:], cupy.int64(self.shape[1]),
                     value))
            if ufunc == cupy.argmin:
                self._min_arg_reduction_kern(
                    (self.shape[0],), (1,),
                    (self.data.astype(cupy.float64), self.indices,
                     self.indptr[:len(self.indptr) - 1],
                     self.indptr[1:], cupy.int64(self.shape[1]),
                     value))

        if axis == 0:
            # Create the vector to hold output
            value = cupy.zeros(self.shape[1]).astype(cupy.int64)

            # Perform the calculation
            if ufunc == cupy.argmax:
                self._max_arg_reduction_kern(
                    (self.shape[1],), (1,),
                    (self.data.astype(cupy.float64), self.indices,
                     self.indptr[:len(self.indptr) - 1],
                     self.indptr[1:], cupy.int64(self.shape[0]),
                     value))
            if ufunc == cupy.argmin:
                self._min_arg_reduction_kern(
                    (self.shape[1],), (1,),
                    (self.data.astype(cupy.float64), self.indices,
                     self.indptr[:len(self.indptr) - 1],
                     self.indptr[1:],
                     cupy.int64(self.shape[0]), value))

        return value

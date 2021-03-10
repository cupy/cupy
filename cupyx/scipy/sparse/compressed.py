import string
import warnings

import numpy
try:
    import scipy.sparse
    scipy_available = True
except ImportError:
    scipy_available = False

import cupy
import cupyx

from cupy import core
from cupy.core import _scalar
from cupy._creation import basic
from cupy import cusparse
from cupyx.scipy.sparse import base
from cupyx.scipy.sparse import data as sparse_data
from cupyx.scipy.sparse import sputils
from cupyx.scipy.sparse import _util

from cupyx.scipy.sparse import _index


class _compressed_sparse_matrix(sparse_data._data_matrix,
                                sparse_data._minmax_mixin,
                                _index.IndexMixin):

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

    # For _max_arg_reduction_mod and _min_arg_reduction_mod below, we pick
    # the right template specialization according to input dtypes at runtime.
    # The distinction in int types (T2) is important for portability in OS.

    _argmax_argmin_code = r'''
        template<typename T1, typename T2> __global__ void
        ${func}_arg_reduction(T1* data, int* indices, int* x, int* y,
                              int length, T2* z) {
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
                    if (data[entry] ${op} data_value){
                        data_index = indices[entry];
                        data_value = data[entry];
                    }
                }
            }

            // Store in the return function
            z[tid] = data_index;
        }'''

    _max_arg_reduction_mod = core.RawModule(
        code=string.Template(_argmax_argmin_code).substitute(
            func='max', op='>'),
        options=('-std=c++11',),
        name_expressions=['max_arg_reduction<float, int>',
                          'max_arg_reduction<float, long long>',
                          'max_arg_reduction<double, int>',
                          'max_arg_reduction<double, long long>'])

    _min_arg_reduction_mod = core.RawModule(
        code=string.Template(_argmax_argmin_code).substitute(
            func='min', op='<'),
        options=('-std=c++11',),
        name_expressions=['min_arg_reduction<float, int>',
                          'min_arg_reduction<float, long long>',
                          'min_arg_reduction<double, int>',
                          'min_arg_reduction<double, long long>'])

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
            if not _util.isshape(shape):
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

        elif _util.isshape(arg1):
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

        if dtype.char not in '?fdFD':
            raise ValueError(
                'Only bool, float32, float64, complex64 and complex128 '
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
        data, indices, _ = _index._get_csr_submatrix_major_axis(
            self.data, self.indices, self.indptr, major, major + 1)
        dtype = data.dtype
        res = cupy.zeros((), dtype=dtype)
        if dtype.kind == 'c':
            _index._compress_getitem_complex_kern(
                data.real, data.imag, indices, minor, res.real, res.imag)
        else:
            _index._compress_getitem_kern(data, indices, minor, res)
        return res

    def _get_sliceXslice(self, row, col):
        major, minor = self._swap(row, col)
        copy = major.step in (1, None)
        return self._major_slice(major)._minor_slice(minor, copy=copy)

    def _get_arrayXarray(self, row, col, not_found_val=0):
        # inner indexing
        idx_dtype = self.indices.dtype
        M, N = self._swap(*self.shape)
        major, minor = self._swap(row, col)
        major = major.astype(idx_dtype, copy=False)
        minor = minor.astype(idx_dtype, copy=False)

        val = _index._csr_sample_values(
            M, N, self.indptr, self.indices, self.data,
            major.ravel(), minor.ravel(),
            not_found_val)

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
        M = idx.size
        new_shape = self._swap(M, N)
        if self.nnz == 0 or M == 0:
            return self.__class__(new_shape)

        return self.__class__(
            _index._csr_row_index(self.data, self.indices, self.indptr, idx),
            shape=new_shape, copy=False)

    def _minor_index_fancy(self, idx):
        """Index along the minor axis where idx is an array of ints.
        """
        M, _ = self._swap(*self.shape)
        N = idx.size
        new_shape = self._swap(M, N)
        if self.nnz == 0 or N == 0:
            return self.__class__(new_shape)

        if idx.size * M < self.nnz:
            # TODO (asi1024): Implement faster algorithm.
            pass

        return self._tocsx()._major_index_fancy(idx)._tocsx()

    def _major_slice(self, idx, copy=False):
        """Index along the major axis where idx is a slice object.
        """
        M, N = self._swap(*self.shape)
        start, stop, step = idx.indices(M)

        if start == 0 and stop == M and step == 1:
            return self.copy() if copy else self

        M = len(range(start, stop, step))
        new_shape = self._swap(M, N)

        if step == 1:
            if M == 0 or self.nnz == 0:
                return self.__class__(new_shape, dtype=self.dtype)
            return self.__class__(
                _index._get_csr_submatrix_major_axis(
                    self.data, self.indices, self.indptr, start, stop),
                shape=new_shape, copy=copy)
        rows = cupy.arange(start, stop, step, dtype=self.indptr.dtype)
        return self._major_index_fancy(rows)

    def _minor_slice(self, idx, copy=False):
        """Index along the minor axis where idx is a slice object.
        """
        M, N = self._swap(*self.shape)
        start, stop, step = idx.indices(N)

        if start == 0 and stop == N and step == 1:
            return self.copy() if copy else self

        N = len(range(start, stop, step))
        new_shape = self._swap(M, N)

        if N == 0 or self.nnz == 0:
            return self.__class__(new_shape)
        if step == 1:
            return self.__class__(
                _index._get_csr_submatrix_minor_axis(
                    self.data, self.indices, self.indptr, start, stop),
                shape=new_shape, copy=False)
        cols = cupy.arange(start, stop, step, dtype=self.indices.dtype)
        return self._minor_index_fancy(cols)

    def _set_intXint(self, row, col, x):
        i, j = self._swap(row, col)
        self._set_many(i, j, x)

    def _set_arrayXarray(self, row, col, x):
        i, j = self._swap(row, col)
        self._set_many(i, j, x)

    def _set_arrayXarray_sparse(self, row, col, x):
        # clear entries that will be overwritten
        self._zero_many(*self._swap(row, col))

        M, N = row.shape  # matches col.shape
        broadcast_row = M != 1 and x.shape[0] == 1
        broadcast_col = N != 1 and x.shape[1] == 1
        r, c = x.row, x.col
        x = cupy.asarray(x.data, dtype=self.dtype)
        if broadcast_row:
            r = cupy.repeat(cupy.arange(M), r.size)
            c = cupy.tile(c, M)
            x = cupy.tile(x, M)
        if broadcast_col:
            r = cupy.repeat(r, N)
            c = cupy.tile(cupy.arange(N), c.size)
            x = cupy.repeat(x, N)
        # only assign entries in the new sparsity structure
        i, j = self._swap(row[r, c], col[r, c])
        self._set_many(i, j, x)

    def _prepare_indices(self, i, j):
        M, N = self._swap(*self.shape)

        def check_bounds(indices, bound):
            idx = indices.max()
            if idx >= bound:
                raise IndexError('index (%d) out of range (>= %d)' %
                                 (idx, bound))
            idx = indices.min()
            if idx < -bound:
                raise IndexError('index (%d) out of range (< -%d)' %
                                 (idx, bound))

        i = cupy.array(i, dtype=self.indptr.dtype,
                       copy=True, ndmin=1).ravel()
        j = cupy.array(j, dtype=self.indices.dtype,
                       copy=True, ndmin=1).ravel()
        check_bounds(i, M)
        check_bounds(j, N)
        return i, j, M, N

    def _set_many(self, i, j, x):
        """Sets value at each (i, j) to x
        Here (i,j) index major and minor respectively, and must not contain
        duplicate entries.
        """
        i, j, M, N = self._prepare_indices(i, j)
        x = cupy.array(x, dtype=self.dtype, copy=True, ndmin=1).ravel()

        new_sp = cupyx.scipy.sparse.csr_matrix(
            (cupy.arange(self.nnz, dtype=cupy.float32),
             self.indices, self.indptr), shape=(M, N))

        offsets = new_sp._get_arrayXarray(
            i, j, not_found_val=-1).astype(cupy.int32).ravel()

        if -1 not in offsets:
            # only affects existing non-zero cells
            self.data[offsets] = x
            return

        else:
            warnings.warn('Changing the sparsity structure of a '
                          '{}_matrix is expensive.'
                          ' lil_matrix is more efficient.'.format(self.format))
            # replace where possible
            mask = offsets > -1
            self.data[offsets[mask]] = x[mask]
            # only insertions remain
            mask = ~mask
            i = i[mask]
            i[i < 0] += M
            j = j[mask]
            j[j < 0] += N
            self._insert_many(i, j, x[mask])

    def _zero_many(self, i, j):
        """Sets value at each (i, j) to zero, preserving sparsity structure.
        Here (i,j) index major and minor respectively.
        """
        i, j, M, N = self._prepare_indices(i, j)

        new_sp = cupyx.scipy.sparse.csr_matrix(
            (cupy.arange(self.nnz, dtype=cupy.float32),
             self.indices, self.indptr), shape=(M, N))

        offsets = new_sp._get_arrayXarray(
            i, j, not_found_val=-1).astype(cupy.int32).ravel()

        # only assign zeros to the existing sparsity structure
        self.data[offsets[offsets > -1]] = 0

    def _perform_insert(self, indices_inserts, data_inserts,
                        rows, row_counts, idx_dtype):
        """Insert new elements into current sparse matrix in sorted order"""
        indptr_diff = cupy.diff(self.indptr)
        indptr_diff[rows] += row_counts

        new_indptr = cupy.empty(self.indptr.shape, dtype=idx_dtype)
        new_indptr[0] = idx_dtype(0)
        new_indptr[1:] = indptr_diff

        # Build output arrays
        cupy.cumsum(new_indptr, out=new_indptr)
        out_nnz = int(new_indptr[-1])

        new_indices = cupy.empty(out_nnz, dtype=idx_dtype)
        new_data = cupy.empty(out_nnz, dtype=self.data.dtype)

        # Build an indexed indptr that contains the offsets for each
        # row but only for in i, j, and x.
        new_indptr_lookup = cupy.zeros(new_indptr.size, dtype=idx_dtype)
        new_indptr_lookup[1:][rows] = row_counts
        cupy.cumsum(new_indptr_lookup, out=new_indptr_lookup)

        _index._insert_many_populate_arrays(
            indices_inserts, data_inserts, new_indptr_lookup,
            self.indptr, self.indices, self.data, new_indptr, new_indices,
            new_data, size=self.indptr.size-1)

        self.indptr = new_indptr
        self.indices = new_indices
        self.data = new_data

    def _insert_many(self, i, j, x):
        """Inserts new nonzero at each (i, j) with value x
        Here (i,j) index major and minor respectively.
        i, j and x must be non-empty, 1d arrays.
        Inserts each major group (e.g. all entries per row) at a time.
        Maintains has_sorted_indices property.
        Modifies i, j, x in place.
        """

        order = cupy.argsort(i)  # stable for duplicates
        i = i.take(order)
        j = j.take(order)
        x = x.take(order)

        # Update index data type

        idx_dtype = sputils.get_index_dtype(
            (self.indices, self.indptr), maxval=(
                self.nnz + x.size))

        self.indptr = self.indptr.astype(idx_dtype)
        self.indices = self.indices.astype(idx_dtype)
        self.data = self.data.astype(self.dtype)

        indptr_inserts, indices_inserts, data_inserts = \
            _index._select_last_indices(i, j, x, idx_dtype)

        rows, ui_indptr = cupy.unique(indptr_inserts, return_index=True)

        to_add = cupy.empty(ui_indptr.size+1, ui_indptr.dtype)
        to_add[-1] = j.size
        to_add[:-1] = ui_indptr
        ui_indptr = to_add

        # Compute the counts for each row in the insertion array
        row_counts = cupy.zeros(ui_indptr.size-1, dtype=idx_dtype)
        cupyx.scatter_add(
            row_counts, cupy.searchsorted(rows, indptr_inserts), 1)

        self._perform_insert(indices_inserts, data_inserts,
                             rows, row_counts, idx_dtype)

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
            # Note: it's important to set "int" here, following what SciPy
            # does, as the outcome dtype is platform dependent
            value = cupy.zeros(self.shape[0], dtype=int)

            # Perform the calculation
            ker_name = '_arg_reduction<{}, {}>'.format(
                _scalar.get_typename(self.data.dtype),
                _scalar.get_typename(value.dtype))
            if ufunc == cupy.argmax:
                ker_name = 'max' + ker_name
                ker = self._max_arg_reduction_mod.get_function(ker_name)
            elif ufunc == cupy.argmin:
                ker_name = 'min' + ker_name
                ker = self._min_arg_reduction_mod.get_function(ker_name)
            ker((self.shape[0],), (1,),
                (self.data, self.indices,
                 self.indptr[:len(self.indptr) - 1],
                 self.indptr[1:], cupy.int64(self.shape[1]),
                 value))

        elif axis == 0:
            # Create the vector to hold output
            # Note: it's important to set "int" here, following what SciPy
            # does, as the outcome dtype is platform dependent
            value = cupy.zeros(self.shape[1], dtype=int)

            # Perform the calculation
            ker_name = '_arg_reduction<{}, {}>'.format(
                _scalar.get_typename(self.data.dtype),
                _scalar.get_typename(value.dtype))
            if ufunc == cupy.argmax:
                ker_name = 'max' + ker_name
                ker = self._max_arg_reduction_mod.get_function(ker_name)
            elif ufunc == cupy.argmin:
                ker_name = 'min' + ker_name
                ker = self._min_arg_reduction_mod.get_function(ker_name)
            ker((self.shape[1],), (1,),
                (self.data, self.indices,
                 self.indptr[:len(self.indptr) - 1],
                 self.indptr[1:], cupy.int64(self.shape[0]),
                 value))

        return value

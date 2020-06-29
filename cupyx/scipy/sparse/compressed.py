import numpy
try:
    import scipy.sparse
    scipy_available = True
except ImportError:
    scipy_available = False

import cupy
from cupy import core
from cupy.creation import basic
from cupy import cusparse
from cupyx.scipy.sparse import base
from cupyx.scipy.sparse import data as sparse_data
from cupyx.scipy.sparse import util
from cupyx.scipy.sparse import sputils

from cupyx.scipy.sparse.index import IndexMixin
from cupyx.scipy.sparse.index import get_csr_submatrix

from cupyx.scipy.sparse.index import csr_column_index1
from cupyx.scipy.sparse.index import csr_column_index2

from cupyx.scipy.sparse.index import csr_sample_values


class _compressed_sparse_matrix(sparse_data._data_matrix,
                                sparse_data._minmax_mixin,
                                IndexMixin):

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

            has_canonical_format = x.has_canonical_format
        elif util.isshape(arg1):
            m, n = arg1
            m, n = int(m), int(n)
            data = basic.zeros(0, dtype if dtype else 'd')
            indices = basic.zeros(0, 'i')
            indptr = basic.zeros(self._swap(m, n)[0] + 1, dtype='i')
            # shape and copy argument is ignored
            shape = (m, n)
            copy = False
            has_canonical_format = True

        elif scipy_available and scipy.sparse.issparse(arg1):
            # Convert scipy.sparse to cupyx.scipy.sparse
            x = arg1.asformat(self.format)
            data = cupy.array(x.data)
            indices = cupy.array(x.indices, dtype='i')
            indptr = cupy.array(x.indptr, dtype='i')
            copy = False

            if shape is None:
                shape = arg1.shape
            has_canonical_format = x.has_canonical_format

        elif isinstance(arg1, tuple) and len(arg1) == 3:
            data, indices, indptr = arg1
            if not (base.isdense(data) and data.ndim == 1 and
                    base.isdense(indices) and indices.ndim == 1 and
                    base.isdense(indptr) and indptr.ndim == 1):
                raise ValueError(
                    'data, indices, and indptr should be 1-D')

            if len(data) != len(indices):
                raise ValueError('indices and data should have the same size')

            has_canonical_format = False

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

            has_canonical_format = True

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
        self._has_canonical_format = has_canonical_format

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

    # def __getitem__(self, slices):
    #
    #     if isinstance(slices, tuple):
    #         slices = list(slices)
    #     elif isinstance(slices, list):
    #         slices = list(slices)
    #         if all([isinstance(s, int) for s in slices]):
    #             slices = [slices]
    #     else:
    #         slices = [slices]
    #
    #     ellipsis = -1
    #     n_ellipsis = 0
    #     for i, s in enumerate(slices):
    #         if s is None:
    #             raise IndexError('newaxis is not supported')
    #         elif s is Ellipsis:
    #             ellipsis = i
    #             n_ellipsis += 1
    #     if n_ellipsis > 0:
    #         ellipsis_size = self.ndim - (len(slices) - 1)
    #         slices[ellipsis:ellipsis + 1] = [slice(None)] * ellipsis_size
    #
    #     if len(slices) == 2:
    #         row, col = slices
    #     elif len(slices) == 1:
    #         row, col = slices[0], slice(None)
    #     else:
    #         raise IndexError('invalid number of indices')
    #
    #     major, minor = self._swap(row, col)  # This will fail on COO
    #     major_size, minor_size = self._swap(*self._shape)
    #
    #     if numpy.isscalar(major):
    #         i = int(major)
    #         if i < 0:
    #             i += major_size
    #         if not (0 <= i < major_size):
    #             raise IndexError('index out of bounds')
    #         if numpy.isscalar(minor):
    #             j = int(minor)
    #             if j < 0:
    #                 j += minor_size
    #             if not (0 <= j < minor_size):
    #                 raise IndexError('index out of bounds')
    #
    #             # If i,j are both scalars, just grab the element
    #             return self._get_single(i, j)
    #
    #         elif minor == slice(None):
    #
    #             # Get all columns for single row
    #             # Done.
    #             return self._get_major_slice(slice(i, i + 1))
    #
    #         elif isinstance(minor, (list, tuple, set)):
    #
    #             # Fetch list of columns for single row
    #             # Since we know how many columns we have, it should be straightforward
    #             # to schedule as a kernel
    #             # Pass 1- Compute output degree (indptr)
    #             # Pass 2- Populate columns & values
    #             """
    #             Example of use:
    #
    #             indptr = self._compute_output_degree(major, minor)
    #             indices, data = self._populate_indices_and_data(indptr, major, minor)
    #
    #             The branching conditionals that require both major/minor will be pushed
    #             down to compute_output_degree and populate_indices_and_data
    #
    #             Pass #1:
    #             --------
    #             We really have 3 different types of functions to compute the output
    #             degrees based on the indptr:
    #                - major == scalar: grab index from indptr
    #                - major == list/tuple/set: grab elements from indptr that exist
    #                - major == slice: grab range of elements from indptr
    #                - major == slice(None): noop- don't need to do anything
    #
    #             Then we have 3 different types of functions to compute the output
    #             degrees based on the indices/values:
    #                - minor == scalar: just need to filter indptr for a single column
    #                - minor == list/tuple/set: Just need to filter indptr for elements that exist
    #                - minor == slice: grab range of minor elements from filtered indptr
    #                - minor == slice(None): noop- don't need to do anything
    #
    #
    #             Pass #2:
    #             --------
    #
    #             Then we just need to apply the rules from Pass #1 to populate the data and indices.
    #             """
    #
    #             self._major_scalar_minor_index(major, minor)
    #
    #         else:
    #
    #             # Slice columns for single row
    #             # Pass 1- Compute output degree (indptr)
    #             # Pass 2- Populate columns & values
    #             return self._major_scalar_minor_slice(major, minor)
    #
    #     elif major == slice(None):
    #
    #         if minor == slice(None):
    #
    #             # Just return self.
    #             return self.copy()
    #
    #         elif isinstance(min, (list, tuple, set)):
    #
    #             # Get all rows for a predefined set of columns
    #             # Pass 1- Compute output degree (indptr)
    #             # Pass 2- Populate values
    #             return self._minor_index(minor)
    #
    #         elif isinstance(minor, slice):
    #
    #             # Get range of columns for all rows
    #             # Pass 1- Compute output degree (indptr)
    #             # Pass 2- Populate values in slice
    #             return self._minor_slice(minor)
    #
    #     elif isinstance(major, slice):
    #
    #         # Fails: csr[:, 1:5], csr[1:5, 1:5], Succeeds: csr[1:5, :]
    #         # Fails: csc[1:5, :], csc[1:5, 1:5], Succeeds: csc[:, 1:5]
    #         if minor == slice(None):
    #
    #             # Get all columns for a range of rows
    #             # Done.
    #             return self._get_major_slice(major)
    #
    #         elif isinstance(minor, (list, tuple, set)):
    #
    #             # Get range of rows for a predefined set of columns
    #             # Slice `indptr` and diff to figure out grid size & # threads
    #             # Pass 1- Compute output degree based on non-zero
    #             #         values for indexed cols (indptr)
    #             # Pass 2- Populate output values
    #             return self._major_slice_minor_index(major, minor)
    #
    #         elif isinstance(minor, slice):
    #
    #             # Slice rows and columns
    #             # Two-pass kernel-
    #             # Slice `indptr` and diff to figure out the grid size and
    #             # number of threads.
    #             # Pass 1- Compute output degree (indptr)
    #             # Pass 2- Compute output values
    #             return self._major_slice_minor_slice(major, minor)
    #
    #     elif isinstance(major, (list, tuple, set)):
    #
    #         if minor == slice(None):
    #
    #             # Get all columns for index of rows
    #             # Two-pass kernel-
    #             # Pass 1- Compute output degree (indptr)
    #             # Pass 2- Populate values for rows
    #             self._major_index(major)
    #
    #         elif isinstance(minor, (list, tuple, set)):
    #
    #             # Get predefined set of rows and predefined set of columns
    #             # Two-pass kernel
    #             # Pass 1- Compute output degree (indptr)
    #             # Pass 2- Populate values for rows
    #             self._major_index_minor_index(major, minor)
    #
    #         elif isinstance(minor, slice):
    #
    #             # Get predefined set of rows and a range of columns
    #             # Two-pass kernel
    #             # Pass 1- Compute output degree (indptr)
    #             # Pass 2- Populate values for rows
    #             self._major_index_minor_slice(major, minor)
    #
    #     raise ValueError("unsupported indexing: %s" % slices)


    #######################
    # Getting and Setting #
    #######################

    def _get_intXint(self, row, col):
        M, N = self._swap(self.shape)
        major, minor = self._swap((row, col))

        # @TODO
        indptr, indices, data = get_csr_submatrix(
            M, N, self.indptr, self.indices, self.data,
            major, major + 1, minor, minor + 1)
        return data.sum(dtype=self.dtype)

    def _get_sliceXslice(self, row, col):
        major, minor = self._swap((row, col))
        if major.step in (1, None) and minor.step in (1, None):
            return self._get_submatrix(major, minor, copy=True)
        return self._major_slice(major)._minor_slice(minor)

    def _get_arrayXarray(self, row, col):
        # inner indexing
        idx_dtype = self.indices.dtype
        M, N = self._swap(self.shape)
        major, minor = self._swap((row, col))
        major = cupy.asarray(major, dtype=idx_dtype)
        minor = cupy.asarray(minor, dtype=idx_dtype)

        val = cupy.empty(major.size, dtype=self.dtype)

        # @TODO
        csr_sample_values(M, N, self.indptr, self.indices, self.data,
                          major.size, major.ravel(), minor.ravel(), val)
        if major.ndim == 1:

            # @TODO: Replace w/ 2-d ndarray?
            return asmatrix(val)
        return self.__class__(val.reshape(major.shape))

    def _get_columnXarray(self, row, col):
        # outer indexing
        major, minor = self._swap((row, col))
        return self._major_index_fancy(major)._minor_index_fancy(minor)

    def _major_index_fancy(self, idx):
        """Index along the major axis where idx is an array of ints.
        """
        idx_dtype = self.indices.dtype
        indices = cupy.asarray(idx, dtype=idx_dtype).ravel()

    def _get_minor_index_fancy(self, idx):
        """Index along the minor axis where idx is an array of ints.
        """
        idx_dtype = self.indices.dtype
        idx = cupy.asarray(idx, dtype=idx_dtype).ravel()

        M, N = self._swap(self.shape)
        k = len(idx)
        new_shape = self._swap((M, k))
        if k == 0:
            return self.__class__(new_shape)

        # pass 1: count idx entries and compute new indptr
        col_offsets = cupy.zeros(N, dtype=idx_dtype)
        res_indptr = cupy.empty_like(self.indptr)

        # @TODO
        csr_column_index1(k, idx, M, N, self.indptr, self.indices,
                          col_offsets, res_indptr)

        # pass 2: copy indices/data for selected idxs
        col_order = cupy.argsort(idx).astype(idx_dtype, copy=False)
        nnz = res_indptr[-1]
        res_indices = cupy.empty(nnz, dtype=idx_dtype)
        res_data = cupy.empty(nnz, dtype=self.dtype)

        # @TODO
        csr_column_index2(col_order, col_offsets, len(self.indices),
                          self.indices, self.data, res_indices, res_data)
        return self.__class__((res_data, res_indices, res_indptr),
                              shape=new_shape, copy=False)

    def _minor_slice(self, idx, copy=False):
        """Index along the minor axis where idx is a slice object.
        """
        if idx == slice(None):
            return self.copy() if copy else self

        M, N = self._swap(self.shape)
        start, stop, step = idx.indices(N)
        N = len(range(start, stop, step))
        if N == 0:
            return self.__class__(self._swap((M, N)))
        if step == 1:
            return self._get_submatrix(minor=idx, copy=copy)
        # TODO: don't fall back to fancy indexing here
        return self._get_minor_index_fancy(cupy.arange(start, stop, step))

    @staticmethod
    def _process_slice(sl, num):
        if sl is None:
            i0, i1 = 0, num
        elif isinstance(sl, slice):
            i0, i1, stride = sl.indices(num)
            if stride != 1:
                raise ValueError('slicing with step != 1 not supported')
            i0 = min(i0, i1)  # give an empty slice when i0 > i1
        elif sputils.isintlike(sl):
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
        major, minor: None, int, or slice with step 1
        """
        M, N = self._swap(self.shape)
        i0, i1 = self._process_slice(major, M)
        j0, j1 = self._process_slice(minor, N)

        if i0 == 0 and j0 == 0 and i1 == M and j1 == N:
            return self.copy() if copy else self
                                # @todo
        indptr, indices, data = get_csr_submatrix(
            M, N, self.indptr, self.indices, self.data, i0, i1, j0, j1)

        shape = self._swap((i1 - i0, j1 - j0))
        return self.__class__((data, indices, indptr), shape=shape,
                              dtype=self.dtype, copy=False)

    def _major_slice(self, major):
        major_size, minor_size = self._swap(*self._shape)
        # major.indices cannot be used because scipy.sparse behaves differently
        major_start = major.start
        major_stop = major.stop
        major_step = major.step
        if major_start is None:
            major_start = 0
        if major_stop is None:
            major_stop = major_size
        if major_step is None:
            major_step = 1
        if major_start < 0:
            major_start += major_size
        if major_stop < 0:
            major_stop += major_size
        major_start = max(min(major_start, major_size), 0)
        major_stop = max(min(major_stop, major_size), 0)

        if major_step != 1:
            raise ValueError('slicing with step != 1 not supported')

        if not (major_start <= major_stop):
            # will give an empty slice, but preserve shape on the other axis
            major_start = major_stop

        start = self.indptr[major_start]
        stop = self.indptr[major_stop]
        data = self.data[start:stop]
        indptr = self.indptr[major_start:major_stop + 1] - start
        indices = self.indices[start:stop]

        shape = self._swap(len(indptr) - 1, minor_size)
        return self.__class__(
            (data, indices, indptr), shape=shape, dtype=self.dtype, copy=False)

    @property
    def has_canonical_format(self):
        return self._has_canonical_format

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

    # TODO(unno): Implement sorted_indices

    def sum_duplicates(self):
        if self._has_canonical_format:
            return
        if self.data.size == 0:
            self._has_canonical_format = True
            return
        coo = self.tocoo()
        coo.sum_duplicates()
        self.__init__(coo.asformat(self.format))
        self._has_canonical_format = True

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

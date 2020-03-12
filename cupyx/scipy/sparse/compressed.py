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


class _compressed_sparse_matrix(sparse_data._data_matrix,
                                sparse_data._minmax_mixin):
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

    # https: // www.micc.unifi.it / bertini / download
    # / gpu - programming - basics / 2017 / gpu_cuda_5.pdf
    setitem_preamble = '''
            #if __CUDA_ARCH__ < 600
            __device__ double atomicExch(double* address, double val)
            {
                unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
                unsigned long long int old = *address_as_ull, compare;
                do {
                    compare = old;
                    old = atomicCAS(address_as_ull, compare,
                                            __double_as_longlong(val));
                } while (compare != old);
                return __longlong_as_double(old);
            }
            #endif
            '''

    _compress_setitem_kern = core.ElementwiseKernel(
        'T values, S offsets', 'raw T data',
        'atomicExch(&data[offsets], values);',
        'compress_setitem',
        preamble=setitem_preamble)

    _compress_setitem_complex_kern = core.ElementwiseKernel(
        'T values_real, T values_imag, S offsets',
        'raw T real, raw T imag',
        '''
          atomicExch(&real[offsets], values_real);
          atomicExch(&imag[offsets], values_imag);
        ''',
        'compress_setitem_complex',
        preamble=setitem_preamble)

    _data_offsets_canonical = core.RawKernel(r'''
        extern "C" __global__
        void data_offsets_canonical (int rows, int cols, int* indptr,
        int* indices, int* outptr, int* inrows, int* incols){
            // Get the index of the block
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            int i = inrows[tid] < 0 ? inrows[tid] + rows : inrows[tid];
            int j = incols[tid] < 0 ? incols[tid] + cols : incols[tid];
            int start = indptr[i], stop = indptr[i+1] ,offset = -1;
            atomicExch(&outptr[tid],-1);
            while (start < stop) {
                int mid = start + (stop - start)/2;
                if (indices[mid] == j){
                    offset = mid ;
                    stop = mid - 1 ;
                } else if (indices[mid] < j) {
                    start = mid + 1;
                } else {
                    stop = mid - 1 ;
                }
            }
            if (offset != -1 && indices[offset] == j)
                    atomicExch(&outptr[tid], offset);
            }''', 'data_offsets_canonical')

    _data_offsets = core.RawKernel(r'''
        extern "C" __global__
        void data_offsets (int rows, int cols, int* indptr,
        int* indices, int* outptr, int* inrows, int* incols, int ret){
            // Get the index of the block
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            int i = inrows[tid] < 0 ? inrows[tid] + rows : inrows[tid];
            int j = incols[tid] < 0 ? incols[tid] + cols : incols[tid];
            int start = indptr [i], stop = indptr [i+1], offset = -1;

            for(int col = start; col < stop; col++)
            {
                if (indices[col] == j) {
                    offset = col++;
                    for (; col < stop; col++) {
                        if (indices[col] == j) {
                            ret= 1;
                            return;
                        }
                    }
                }
            }
            atomicExch(&outptr[tid], offset);
            ret = 0;
         }''', 'data_offsets')

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

    def __getitem__(self, slices):
        if isinstance(slices, tuple):
            slices = list(slices)
        elif isinstance(slices, list):
            slices = list(slices)
            if all([isinstance(s, int) for s in slices]):
                slices = [slices]
        else:
            slices = [slices]

        ellipsis = -1
        n_ellipsis = 0
        for i, s in enumerate(slices):
            if s is None:
                raise IndexError('newaxis is not supported')
            elif s is Ellipsis:
                ellipsis = i
                n_ellipsis += 1
        if n_ellipsis > 0:
            ellipsis_size = self.ndim - (len(slices) - 1)
            slices[ellipsis:ellipsis + 1] = [slice(None)] * ellipsis_size

        if len(slices) == 2:
            row, col = slices
        elif len(slices) == 1:
            row, col = slices[0], slice(None)
        else:
            raise IndexError('invalid number of indices')

        major, minor = self._swap(row, col)
        major_size, minor_size = self._swap(*self._shape)
        if numpy.isscalar(major):
            i = int(major)
            if i < 0:
                i += major_size
            if not (0 <= i < major_size):
                raise IndexError('index out of bounds')
            if numpy.isscalar(minor):
                j = int(minor)
                if j < 0:
                    j += minor_size
                if not (0 <= j < minor_size):
                    raise IndexError('index out of bounds')
                return self._get_single(i, j)
            elif minor == slice(None):
                return self._get_major_slice(slice(i, i + 1))
        elif isinstance(major, slice):
            if minor == slice(None):
                return self._get_major_slice(major)

        raise ValueError('unsupported indexing')

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

    def _get_major_slice(self, major):
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

    def __setitem__(self, slices, value):
        if isinstance(slices, tuple):
            slices = list(slices)
        elif isinstance(slices, list):
            slices = list(slices)
            if all([isinstance(s, int) for s in slices]):
                slices = [slices]
        else:
            slices = [slices]

        ellipsis = -1
        n_ellipsis = 0
        for i, s in enumerate(slices):
            if s is None:
                raise IndexError('newaxis is not supported')
            elif s is Ellipsis:
                ellipsis = i
                n_ellipsis += 1
        if n_ellipsis > 0:
            ellipsis_size = self.ndim - (len(slices) - 1)
            slices[ellipsis:ellipsis + 1] = [slice(None)] * ellipsis_size

        if len(slices) == 2:
            row, col = slices
        elif len(slices) == 1:
            row, col = slices[0], slice(None)
        else:
            raise IndexError('invalid number of indices')

        major, minor = self._swap(row, col)
        major_size, minor_size = self._swap(*self._shape)
        # flatten rows, cols, values lists
        major = cupy.array(major, dtype=self.indices.dtype,
                           copy=False, ndmin=1).ravel()
        minor = cupy.array(minor, dtype=self.indices.dtype,
                           copy=False, ndmin=1).ravel()
        value = cupy.array(value, dtype=self.dtype,
                           copy=False, ndmin=1).ravel()
        # check boundaries
        major_max = cupy.amax(major)
        major_min = cupy.amin(major)
        minor_max = cupy.amax(minor)
        minor_min = cupy.amin(minor)
        if major_max >= major_size or major_min < - major_size \
                or minor_max >= minor_size or minor_min < - minor_size:
            raise IndexError('index out of bounds')
        self._set_items(major, minor, value, major_size, minor_size)

    def _set_items(self, major, minor, values, major_size, minor_size):
        offsets = cupy.empty(values.size, dtype=self.indices.dtype)
        repeat = self.sample_offsets(major_size, minor_size,
                                     values.size, major, minor, offsets)
        if repeat == 1:
            self.sum_duplicates()
            self.sample_offsets(major_size, minor_size,
                                values.size, major, minor, offsets)
        mask = offsets > -1
        self.update_values(offsets[mask], values[mask])
        if -1 in offsets:
            major = major[~mask]
            minor = minor[~mask]
            values = values[~mask]
            major[major < 0] += major_size
            minor[minor < 0] += minor_size
            self.insert_values(major, minor, values)
        self.eliminate_zeros()

    def eliminate_zeros(self):
        pass

    def update_values(self, offsets, values):
        if self.dtype.kind == 'c':
            self._compress_setitem_complex_kern(
                values.real, values.imag, offsets,
                self.data.real, self.data.imag)
        else:
            self._compress_setitem_kern(values, offsets, self.data)

    def insert_values(self, major, minor, values):
        order = cupy.argsort(major)
        major = major.take(order)
        minor = minor.take(order)
        values = values.take(order)

        uniq_rows, rows_indptr = cupy.unique(major, return_index=True)
        rows_indptr = cupy.append(rows_indptr, len(minor))
        new_nnz = cupy.diff(rows_indptr)
        metadata = zip(uniq_rows, rows_indptr, rows_indptr[1:])

        row = 0
        data = []
        indices = []
        for col, (next, col_start, col_stop) in enumerate(metadata):
            start = self.indptr[row]
            stop = self.indptr[next]
            row = next
            data.append(self.data[start:stop])
            indices.append(self.indices[start:stop])

            uniq_cols, cols_indptr = cupy.unique(
                minor[col_start:col_stop][::-1], return_index=True)
            if len(uniq_cols) == col_stop - col_start:
                data.append(values[col_start:col_stop])
                indices.append(minor[col_start:col_stop])
            else:
                new_nnz[col] = len(uniq_cols)
                data.append(
                    values[col_start:col_stop][::-1][cols_indptr])
                indices.append(
                    minor[col_start:col_stop][::-1][cols_indptr])

        # remaining part
        start = self.indptr[next]
        data.append(self.data[start:])
        indices.append(self.indices[start:])

        nnzs = cupy.empty(self.indptr.shape, dtype=self.indices.dtype)
        indptr_diff = cupy.diff(self.indptr)
        indptr_diff[uniq_rows] += new_nnz
        nnzs[0] = 0
        nnzs[1:] = indptr_diff
        self.data = cupy.concatenate(data)
        self.indices = cupy.concatenate(indices)
        self.indptr = cupy.cumsum(nnzs, out=nnzs)
        self.sort_indices()

    def sample_offsets(self, rows, cols, samples, major, minor, offsets):
        repeat = 0
        if self._has_canonical_format and samples > self.indptr[rows] / 10:
            self._data_offsets_canonical(
                (samples,), (1,),
                (rows, cols, self.indptr, self.indices,
                 offsets, major, minor))
        else:
            self._data_offsets(
                (samples,), (1,),
                (rows, cols, self.indptr, self.indices,
                 offsets, major, minor, repeat))
        return repeat

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
    def sort_indices(self):
        pass

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

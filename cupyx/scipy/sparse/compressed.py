import numpy
from cupy.core.raw import RawModule

try:
    import scipy.sparse

    scipy_available = True
except ImportError:
    scipy_available = False

import cupy
from cupy import core
from cupy.creation import basic
from cupy import cusparse
from cupyx.scipy.sparse import base, coo
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
      __device__
      double atomicExch(double* address, double val)
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
        'S offsets', 'raw T data',
        '''
        if (offsets != -1)
          atomicExch(&data[offsets], T(0));
        ''',
        'compress_setitem',
        preamble=setitem_preamble)

    _compress_setitem_complex_kern = core.ElementwiseKernel(
        'S offsets',
        'raw T real, raw T imag',
        '''
          if (offsets != -1){
              atomicExch(&real[offsets], T(0));
              atomicExch(&imag[offsets], T(0));
          }
        ''',
        'compress_setitem_complex',
        preamble=setitem_preamble)

    mask_module = r'''
       extern "C" __global__
       void data_offsets_canonical (int rows, int cols, int* indptr,
       int* indices, int* major, int* minor, int csc, int * optr){
           // Get the index of the block
           int tid = blockIdx.x * blockDim.x + threadIdx.x;
           int i = major[tid] < 0 ? major[tid] + rows : major[tid];
           int j = minor[tid] < 0 ? minor[tid] + cols : minor[tid];
           major[tid] = i, minor[tid] = j;
           int start = indptr[i], stop = indptr[i+1], offset = -1;
           while (start < stop) {
              int mid = start + (stop - start)/2;
              if (indices[mid] == j){
                    offset = mid ;
                    stop = mid;
              } else if (indices[mid] < j) {
                   if (csc) stop = mid;
                   else start = mid + 1;
               } else {
                   if (csc) start = mid + 1;
                   else stop = mid;
               }
           }
           optr[tid] = offset;
        }
       extern "C" __global__
       void data_offsets (int rows, int cols, int* indptr,
       int* indices, int* major, int* minor, int* optr){
           // Get the index of the block
           int tid = blockIdx.x * blockDim.x + threadIdx.x;
           int i = major[tid] < 0 ? major[tid] + rows : major[tid];
           int j = minor[tid] < 0 ? minor[tid] + cols : minor[tid];
           major[tid] = i, minor[tid] = j;
           int start = indptr [i], stop = indptr [i+1], offset=-1;
           for(int col = start; col < stop; col++)
           {
               if (indices[col] == j) {
                    offset = col;
                    break;
               }
           }
            optr[tid] = offset;
        }
        '''
    module = RawModule(code=mask_module)
    _data_offsets_canonical = module.get_function('data_offsets_canonical')
    _data_offsets = module.get_function('data_offsets')

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

    def __setitem__(self, slices, values, format):
        """
                 Assigns a value or a list of values to particular indices

                Args:
                    slices: tuple or list of rows and columns indices
                    values: single or list of values

                Returns: void

                .. warning::
                     Currently doesn't support slice assignment

                """
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

        if type(major) is slice or type(minor) is slice:
            raise ValueError("Unsupported assignment")
        if type(major) is int:
            major = [major]
        if type(minor) is int:
            minor = [minor]
        # validate boundaries
        if max(major) >= major_size or min(major) < - major_size \
                or max(minor) >= minor_size or min(minor) < - minor_size:
            raise IndexError('index out of bounds')
        # flatten rows, cols, values lists
        major = cupy.array(major, dtype=self.indices.dtype,
                           copy=False, ndmin=1).ravel()
        minor = cupy.array(minor, dtype=self.indices.dtype,
                           copy=False, ndmin=1).ravel()
        values = cupy.array(values, dtype=self.dtype,
                            copy=False, ndmin=1).ravel()
        major, minor, values = self.remove_duplicates(major, minor, values)
        self.mask_existing_entries(major_size, minor_size,
                                   values.size, major, minor, format)
        if format == 'csc':
            compress_new = coo.coo_matrix((values, (minor, major)),
                                          shape=self.shape,
                                          dtype=self.dtype).tocsc()

        elif format == 'csr':
            compress_new = coo.coo_matrix((values, (major, minor)),
                                          shape=self.shape,
                                          dtype=self.dtype).tocsr()
        else:
            raise ValueError("Unsupported format")
        compress_new += self
        compress_new.eliminate_zeros()
        self.data = compress_new.data
        self.indptr = compress_new.indptr
        self.indices = compress_new.indices

    def eliminate_zeros(self):
        pass

    def remove_duplicates(self, major, minor, values):
        """
        Eliminate duplicate matrix entries by keeping the last value.
        Args:
            major: rows array
            minor: columns array
            values: array

        Returns:
            major: array of unique rows indices
            minor: array of unique columns indices
            values: array of latest values
        """
        keys = cupy.stack([major, minor])
        order = cupy.lexsort(keys)
        values = values[order]
        major = major[order]
        minor = minor[order]
        diff = cupy.ElementwiseKernel(
            'raw int32 row, raw int32 col',
            'int32 diff',
            '''
            int index;
            if (i == 0 || row[i - 1] == row[i] && col[i - 1] == col[i]) {
              diff = 0;
            } else {
              diff = 1;
            }
            ''',
            'remove_duplicates_diff'
        )(major, minor, size=major.size)

        if not diff[1:].all():
            index = cupy.cumsum(diff, dtype='i')
            size = int(index[-1]) + 1
            data = cupy.zeros(size, dtype=values.dtype)
            row = cupy.empty(size, dtype='i')
            col = cupy.empty(size, dtype='i')
            if values.dtype.kind == 'f':
                cupy.ElementwiseKernel(
                    'T src_data, int32 src_row, int32 src_col, int32 index',
                    'raw T data, raw int32 row, raw int32 col',
                    '''
                    data[index] = src_data;
                    row[index] = src_row;
                    col[index] = src_col;
                    ''',
                    'remove_duplicates_assign'
                )(values, major, minor, index, data, row, col)
            elif values.dtype.kind == 'c':
                cupy.ElementwiseKernel(
                    'T src_real, T src_imag, int32 src_row, int32 src_col, '
                    'int32 index',
                    'raw T real, raw T imag, raw int32 row, raw int32 col',
                    '''
                    real[index] = src_real;
                    imag[index] = src_imag;
                    row[index] = src_row;
                    col[index] = src_col;
                    ''',
                    'remove_duplicates_assign_complex'
                )(values.real, values.imag, major, minor, index,
                  data.real, data.imag, row, col)
            values = data
            major = row
            minor = col
        return major, minor, values

    def mask_existing_entries(self, major_size, minor_size,
                              samples, major, minor, format):
        """
         - set entries of values to be updated to zero
         - If number of samples to be assigned exceeds an arbitrary threshold,
           binary search's lower bound is applied else linear search is used
        Args:
            major_size: number of rows
            minor_size: number of columns
            samples: number of sample values
            major: flattened cupy array of rows indices
            minor: flattened cupy array of columns indices
        """
        threads = 1024 if samples > 1024 else samples
        offsets = cupy.zeros(samples, dtype='i')
        if self._has_canonical_format and \
                samples > self.indptr[major_size] / 10:
            self._data_offsets_canonical(
                (int(cupy.ceil(samples / threads)),), (threads,),
                (major_size, minor_size, self.indptr, self.indices,
                 major, minor, format == "csc", offsets))
        else:
            self._data_offsets(
                (int(cupy.ceil(samples / threads)),), (threads,),
                (major_size, minor_size, self.indptr, self.indices,
                 major, minor, offsets))
        if self.dtype.kind == 'c':
            self._compress_setitem_complex_kern(offsets,
                                                self.data.real, self.data.imag)
        else:
            self._compress_setitem_kern(offsets, self.data)

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

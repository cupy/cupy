# distutils: language = c++
import sys

import numpy

import cupy
from cupy.core import _errors
from cupy.core._kernel import ElementwiseKernel
from cupy.core._ufuncs import elementwise_copy

from libcpp cimport vector

from cupy.core cimport core
from cupy.core cimport _routines_math as _math
from cupy.core cimport _routines_manipulation as _manipulation
from cupy.core.core cimport ndarray
from cupy.core cimport internal


# ndarray members


cdef ndarray _ndarray_getitem(ndarray self, slices):
    # supports basic indexing (by slices, ints or Ellipsis) and
    # some parts of advanced indexing by integer or boolean arrays.
    # TODO(beam2d): Support the advanced indexing of NumPy.
    cdef Py_ssize_t mask_i
    cdef list slice_list, adv_mask, adv_slices
    cdef bint advanced, mask_exists

    slice_list, advanced, mask_exists = _prepare_slice_list(
        slices, self._shape.size())

    if mask_exists:
        mask_i = _get_mask_index(slice_list)
        return _getitem_mask_single(self, slice_list[mask_i], mask_i)
    if advanced:
        a, adv_slices, adv_mask = _prepare_advanced_indexing(
            self, slice_list)
        if sum(adv_mask) == 1:
            axis = adv_mask.index(True)
            return a.take(adv_slices[axis], axis)
        return _getitem_multiple(a, adv_slices)

    return _simple_getitem(self, slice_list)


cdef _ndarray_setitem(ndarray self, slices, value):
    _scatter_op(self, slices, value, 'update')


cdef tuple _ndarray_nonzero(ndarray self):
    cdef Py_ssize_t count_nonzero
    cdef int ndim
    dtype = numpy.int64
    if self.size == 0:
        count_nonzero = 0
    else:
        r = self.ravel()
        nonzero = cupy.core.not_equal(r, 0, ndarray(r.shape, dtype))
        del r
        scan_index = _math.scan(nonzero)
        count_nonzero = int(scan_index[-1])  # synchronize!
    ndim = max(<int>self._shape.size(), 1)
    if count_nonzero == 0:
        return (ndarray((0,), dtype=dtype),) * ndim

    if ndim <= 1:
        dst = ndarray((count_nonzero,), dtype=dtype)
        _nonzero_kernel_1d(nonzero, scan_index, dst)
        return dst,
    else:
        nonzero.shape = self.shape
        scan_index.shape = self.shape
        dst = ndarray((ndim, count_nonzero), dtype=dtype)
        _nonzero_kernel(nonzero, scan_index, dst)
        return tuple([dst[i] for i in range(ndim)])


cdef _ndarray_scatter_add(ndarray self, slices, value):
    _scatter_op(self, slices, value, 'add')


cdef _ndarray_scatter_max(ndarray self, slices, value):
    _scatter_op(self, slices, value, 'max')


cdef _ndarray_scatter_min(ndarray self, slices, value):
    _scatter_op(self, slices, value, 'min')


cdef ndarray _ndarray_take(ndarray self, indices, axis, out):
    if axis is None:
        return _take(self, indices, 0, self._shape.size() - 1, out)
    else:
        return _take(self, indices, axis, axis, out)


cdef ndarray _ndarray_put(ndarray self, indices, values, mode):
    if mode not in ('raise', 'wrap', 'clip'):
        raise TypeError('clipmode not understood')

    n = self.size
    if not isinstance(indices, ndarray):
        indices = ndarray(indices)
    indices = indices.ravel()

    if not isinstance(values, ndarray):
        values = ndarray(values, dtype=self.dtype)
    if values.size == 0:
        return

    if mode == 'raise':
        err = cupy.zeros((), dtype=numpy.bool_)
        _put_raise_kernel(indices, values, values.size, n, self, err)
        if err:
            raise IndexError('invalid entry in indices array')
    elif mode == 'wrap':
        _put_wrap_kernel(indices, values, values.size, n, self)
    elif mode == 'clip':
        _put_clip_kernel(indices, values, values.size, n, self)


cdef ndarray _ndarray_choose(ndarray self, choices, out, mode):
    a = self
    n = choices.shape[0]

    # broadcast `a` and `choices[i]` for all i
    if a.ndim < choices.ndim - 1:
        for i in range(choices.ndim - 1 - a.ndim):
            a = a[None, ...]
    elif a.ndim > choices.ndim - 1:
        for i in range(a.ndim + 1 - choices.ndim):
            choices = choices[:, None, ...]
    ba, bcs = _manipulation.broadcast(a, choices).values

    if out is None:
        out = ndarray(ba.shape[1:], choices.dtype)

    n_channel = numpy.prod(bcs[0].shape)
    if mode == 'raise':
        if not ((a < n).all() and (0 <= a).all()):
            raise ValueError('invalid entry in choice array')
        _choose_kernel(ba[0], bcs, n_channel, out)
    elif mode == 'wrap':
        ba = ba[0] % n
        _choose_kernel(ba, bcs, n_channel, out)
    elif mode == 'clip':
        _choose_clip_kernel(ba[0], bcs, n_channel, n, out)
    else:
        raise TypeError('clipmode not understood')

    return out


cdef ndarray _ndarray_diagonal(ndarray self, offset, axis1, axis2):
    return _diagonal(self, offset, axis1, axis2)


# private/internal


cpdef tuple _prepare_slice_list(slices, Py_ssize_t ndim):
    cdef Py_ssize_t i, n_newaxes, axis
    cdef list slice_list
    cdef char kind
    cdef bint advanced, mask_exists

    if isinstance(slices, tuple):
        slice_list = list(slices)
    elif isinstance(slices, list):
        slice_list = list(slices)  # copy list
        for s in slice_list:
            if not isinstance(s, int):
                break
        else:
            slice_list = [slice_list]
    else:
        slice_list = [slices]

    slice_list, n_newaxes = internal.complete_slice_list(slice_list, ndim)

    # Check if advanced is true,
    # and convert list/NumPy arrays to cupy.ndarray
    advanced = False
    mask_exists = False
    for i, s in enumerate(slice_list):
        to_gpu = True
        if isinstance(s, list):
            # handle the case when s is an empty list
            s = numpy.array(s)
            if s.size == 0:
                s = s.astype(numpy.int32)
        elif isinstance(s, bool):
            s = numpy.array(s)
        elif isinstance(s, ndarray):
            to_gpu = False
        elif not isinstance(s, numpy.ndarray):
            continue
        kind = ord(s.dtype.kind)
        if kind == b'i' or kind == b'u':
            advanced = True
        elif kind == b'b':
            mask_exists = True
        else:
            raise IndexError(
                'arrays used as indices must be of integer or boolean '
                'type. (actual: {})'.format(s.dtype.type))
        if to_gpu:
            slice_list[i] = core.array(s)

    if not mask_exists and len(slice_list) > ndim + n_newaxes:
        raise IndexError('too many indices for array')
    return slice_list, advanced, mask_exists


cdef Py_ssize_t _get_mask_index(list slice_list) except *:
    cdef Py_ssize_t i, n_not_slice_none, mask_i
    cdef slice none_slice = slice(None)
    n_not_slice_none = 0
    mask_i = -1
    for i, s in enumerate(slice_list):
        if not isinstance(s, slice) or s != none_slice:
            n_not_slice_none += 1
            if isinstance(s, ndarray) and s.dtype == numpy.bool_:
                mask_i = i
    if n_not_slice_none != 1 or mask_i == -1:
        raise ValueError('currently, CuPy only supports slices that '
                         'consist of one boolean array.')
    return mask_i


cdef tuple _prepare_advanced_indexing(ndarray a, list slice_list):
    cdef slice none_slice = slice(None)

    # split slices that can be handled by basic-indexing
    cdef list basic_slices = []
    cdef list adv_slices = []
    cdef list adv_mask = []
    cdef bint use_basic_indexing = False
    for i, s in enumerate(slice_list):
        if s is None:
            basic_slices.append(None)
            adv_slices.append(none_slice)
            adv_mask.append(False)
            use_basic_indexing = True
        elif isinstance(s, slice):
            basic_slices.append(s)
            adv_slices.append(none_slice)
            adv_mask.append(False)
            use_basic_indexing |= s != none_slice
        elif isinstance(s, ndarray):
            kind = s.dtype.kind
            assert kind == 'i' or kind == 'u'
            basic_slices.append(none_slice)
            adv_slices.append(s)
            adv_mask.append(True)
        elif isinstance(s, int):
            basic_slices.append(none_slice)
            scalar_array = ndarray((), dtype=numpy.int64)
            scalar_array.fill(s)
            adv_slices.append(scalar_array)
            adv_mask.append(True)
        else:
            raise IndexError(
                'only integers, slices (`:`), ellipsis (`...`),'
                'numpy.newaxis (`None`) and integer or '
                'boolean arrays are valid indices')

    # check if this is a combination of basic and advanced indexing
    if use_basic_indexing:
        a = _simple_getitem(a, basic_slices)

    return a, adv_slices, adv_mask

cdef ndarray _simple_getitem(ndarray a, list slice_list):
    cdef vector.vector[Py_ssize_t] shape, strides
    cdef ndarray v
    cdef Py_ssize_t i, j, offset, ndim
    cdef Py_ssize_t s_start, s_stop, s_step, dim, ind
    cdef slice ss

    # Create new shape and stride
    j = 0
    offset = 0
    ndim = a._shape.size()
    for i, s in enumerate(slice_list):
        if s is None:
            shape.push_back(1)
            if j < ndim:
                strides.push_back(a._strides[j])
            elif ndim > 0:
                strides.push_back(a._strides[ndim - 1])
            else:
                strides.push_back(a.itemsize)
        elif ndim <= j:
            raise IndexError('too many indices for array')
        elif isinstance(s, slice):
            ss = internal.complete_slice(s, a._shape[j])
            s_start = ss.start
            s_stop = ss.stop
            s_step = ss.step
            if s_step > 0:
                dim = (s_stop - s_start - 1) // s_step + 1
            else:
                dim = (s_stop - s_start + 1) // s_step + 1

            if dim == 0:
                strides.push_back(a._strides[j])
            else:
                strides.push_back(a._strides[j] * s_step)

            if s_start > 0:
                offset += a._strides[j] * s_start
            shape.push_back(dim)
            j += 1
        elif numpy.isscalar(s):
            ind = int(s)
            if ind < 0:
                ind += a._shape[j]
            if not (0 <= ind < a._shape[j]):
                msg = ('Index %s is out of bounds for axis %s with '
                       'size %s' % (s, j, a._shape[j]))
                raise IndexError(msg)
            offset += ind * a._strides[j]
            j += 1
        else:
            raise TypeError('Invalid index type: %s' % type(slice_list[i]))

    v = a.view()
    if a.size != 0:
        v.data = a.data + offset
    # TODO(niboshi): Confirm update_x_contiguity flags
    v._set_shape_and_strides(shape, strides, True, True)
    return v


_nonzero_kernel_1d = ElementwiseKernel(
    'T src, S index', 'raw S dst',
    'if (src != 0) dst[index - 1] = i',
    'nonzero_kernel_1d')


_nonzero_kernel = ElementwiseKernel(
    'T src, S index', 'raw S dst',
    '''
    if (src != 0){
        for(int j = 0; j < _ind.ndim; j++){
            ptrdiff_t ind[] = {j, index - 1};
            dst[ind] = _ind.get()[j];
        }
    }''',
    'nonzero_kernel',
    reduce_dims=False)


_take_kernel_core = '''
ptrdiff_t out_i = indices % index_range;
if (out_i < 0) out_i += index_range;
if (ldim != 1) out_i += (i / (cdim * rdim)) * index_range;
if (rdim != 1) out_i = out_i * rdim + i % rdim;
out = a[out_i];
'''


_take_kernel = ElementwiseKernel(
    'raw T a, S indices, uint32 ldim, uint32 cdim, uint32 rdim, S index_range',
    'T out', _take_kernel_core, 'cupy_take')


_take_kernel_scalar = ElementwiseKernel(
    'raw T a, int64 indices, uint32 ldim, uint32 cdim, uint32 rdim, '
    'int64 index_range',
    'T out', _take_kernel_core, 'cupy_take_scalar')


_choose_kernel = ElementwiseKernel(
    'S a, raw T choices, int32 n_channel',
    'T y',
    'y = choices[i + n_channel * a]',
    'cupy_choose')


_choose_clip_kernel = ElementwiseKernel(
    'S a, raw T choices, int32 n_channel, int32 n',
    'T y',
    '''
      S x = a;
      if (a < 0) {
        x = 0;
      } else if (a >= n) {
        x = n - 1;
      }
      y = choices[i + n_channel * x];
    ''',
    'cupy_choose_clip')


cdef _put_raise_kernel = ElementwiseKernel(
    'S ind, raw T vals, int64 n_vals, int64 n',
    'raw U data, raw bool err',
    '''
      ptrdiff_t ind_ = ind;
      if (!(-n <= ind_ && ind_ < n)) {
        err[0] = 1;
      } else {
        if (ind_ < 0) ind_ += n;
        data[ind_] = (U)(vals[i % n_vals]);
      }
    ''',
    'cupy_put_raise')


cdef _put_wrap_kernel = ElementwiseKernel(
    'S ind, raw T vals, int64 n_vals, int64 n',
    'raw U data',
    '''
      ptrdiff_t ind_ = ind;
      ind_ %= n;
      if (ind_ < 0) ind_ += n;
      data[ind_] = (U)(vals[i % n_vals]);
    ''',
    'cupy_put_wrap')


cdef _put_clip_kernel = ElementwiseKernel(
    'S ind, raw T vals, int64 n_vals, int64 n',
    'raw U data',
    '''
      ptrdiff_t ind_ = ind;
      if (ind_ < 0) {
        ind_ = 0;
      } else if (ind_ >= n) {
        ind_ = n - 1;
      }
      data[ind_] = (U)(vals[i % n_vals]);
    ''',
    'cupy_put_clip')


_scatter_update_kernel = ElementwiseKernel(
    'T v, S indices, int32 cdim, int32 rdim, int32 adim',
    'raw T a',
    '''
      S wrap_indices = indices % adim;
      if (wrap_indices < 0) wrap_indices += adim;
      ptrdiff_t li = i / (rdim * cdim);
      ptrdiff_t ri = i % rdim;
      a[(li * adim + wrap_indices) * rdim + ri] = v;
    ''',
    'cupy_scatter_update')


_scatter_add_kernel = ElementwiseKernel(
    'raw T v, S indices, int32 cdim, int32 rdim, int32 adim',
    'raw T a',
    '''
      S wrap_indices = indices % adim;
      if (wrap_indices < 0) wrap_indices += adim;
      ptrdiff_t li = i / (rdim * cdim);
      ptrdiff_t ri = i % rdim;
      atomicAdd(&a[(li * adim + wrap_indices) * rdim + ri], v[i]);
    ''',
    'cupy_scatter_add')


_scatter_max_kernel = ElementwiseKernel(
    'raw T v, S indices, int32 cdim, int32 rdim, int32 adim',
    'raw T a',
    '''
      S wrap_indices = indices % adim;
      if (wrap_indices < 0) wrap_indices += adim;
      ptrdiff_t li = i / (rdim * cdim);
      ptrdiff_t ri = i % rdim;
      atomicMax(&a[(li * adim + wrap_indices) * rdim + ri], v[i]);
    ''',
    'cupy_scatter_max')


_scatter_min_kernel = ElementwiseKernel(
    'raw T v, S indices, int32 cdim, int32 rdim, int32 adim',
    'raw T a',
    '''
      S wrap_indices = indices % adim;
      if (wrap_indices < 0) wrap_indices += adim;
      ptrdiff_t li = i / (rdim * cdim);
      ptrdiff_t ri = i % rdim;
      atomicMin(&a[(li * adim + wrap_indices) * rdim + ri], v[i]);
    ''',
    'cupy_scatter_min')


_scatter_update_mask_kernel = ElementwiseKernel(
    'raw T v, bool mask, S mask_scanned',
    'T a',
    'if (mask) a = v[mask_scanned - 1]',
    'cupy_scatter_update_mask')


_scatter_add_mask_kernel = ElementwiseKernel(
    'raw T v, bool mask, S mask_scanned',
    'T a',
    'if (mask) a = a + v[mask_scanned - 1]',
    'cupy_scatter_add_mask')


_scatter_max_mask_kernel = ElementwiseKernel(
    'raw T v, bool mask, S mask_scanned',
    'T a',
    'if (mask) a = max(a, v[mask_scanned - 1])',
    'cupy_scatter_max_mask')


_scatter_min_mask_kernel = ElementwiseKernel(
    'raw T v, bool mask, S mask_scanned',
    'T a',
    'if (mask) a = min(a, v[mask_scanned - 1])',
    'cupy_scatter_min_mask')


_getitem_mask_kernel = ElementwiseKernel(
    'T a, bool mask, S mask_scanned',
    'raw T out',
    'if (mask) out[mask_scanned - 1] = a',
    'cupy_getitem_mask')


cpdef _prepare_mask_indexing_single(ndarray a, ndarray mask, Py_ssize_t axis):
    cdef ndarray mask_scanned, mask_br, mask_br_scanned
    cdef int n_true
    cdef tuple lshape, rshape, out_shape

    lshape = a.shape[:axis]
    rshape = a.shape[axis + mask._shape.size():]

    if mask.size == 0:
        masked_shape = lshape + (0,) + rshape
        mask_br = _manipulation._reshape(mask, masked_shape)
        return mask_br, mask_br, masked_shape

    for i, s in enumerate(mask._shape):
        if a.shape[axis + i] != s:
            raise IndexError('boolean index did not match')

    # Get number of True in the mask to determine the shape of the array
    # after masking.
    if mask.size <= 2 ** 31 - 1:
        mask_type = numpy.int32
    else:
        mask_type = numpy.int64
    mask_scanned = _math.scan(mask.astype(mask_type).ravel())  # starts with 1
    n_true = int(mask_scanned[-1])
    masked_shape = lshape + (n_true,) + rshape

    # When mask covers the entire array, broadcasting is not necessary.
    if mask._shape.size() == a._shape.size() and axis == 0:
        return (
            mask,
            _manipulation._reshape(mask_scanned, mask._shape),
            masked_shape)
    mask_scanned = None

    # The scan of the broadcasted array is used to index on kernel.
    mask = _manipulation._reshape(
        mask,
        axis * (1,) + mask.shape + (a.ndim - axis - mask.ndim) * (1,))
    if mask._shape.size() > a._shape.size():
        raise IndexError('too many indices for array')

    mask = _manipulation.broadcast_to(mask, a.shape)
    if mask.size <= 2 ** 31 - 1:
        mask_type = numpy.int32
    else:
        mask_type = numpy.int64
    mask_scanned = _manipulation._reshape(
        _math.scan(mask.astype(mask_type).ravel()),
        mask._shape)
    return mask, mask_scanned, masked_shape


cpdef ndarray _getitem_mask_single(ndarray a, ndarray mask, int axis):
    cdef ndarray mask_scanned
    cdef tuple masked_shape

    mask, mask_scanned, masked_shape = _prepare_mask_indexing_single(
        a, mask, axis)
    out = ndarray(masked_shape, dtype=a.dtype)
    if out.size == 0:
        return out
    return _getitem_mask_kernel(a, mask, mask_scanned, out)


cdef ndarray _take(ndarray a, indices, int li, int ri, ndarray out=None):
    # When li + 1 == ri this function behaves similarly to np.take
    cdef tuple out_shape, ind_shape, indices_shape
    cdef int i, ndim = a._shape.size()
    cdef Py_ssize_t ldim, cdim, rdim, index_range
    if ndim == 0:
        a = a.ravel()
        ndim = 1

    if not (-ndim <= li < ndim and -ndim <= ri < ndim):
        raise _errors._AxisError('Axis overrun')

    if ndim == 1:
        li = ri = 0
    else:
        li %= ndim
        ri %= ndim
        assert 0 <= li <= ri

    if numpy.isscalar(indices):
        indices_shape = ()
        cdim = 1
    else:
        if not isinstance(indices, ndarray):
            indices = core.array(indices, dtype=int)
        indices_shape = indices.shape
        cdim = indices.size

    ldim = rdim = 1
    if ndim == 1:
        out_shape = indices_shape
        index_range = a.size
    else:
        a_shape = a.shape
        out_shape = a_shape[:li] + indices_shape + a_shape[ri + 1:]
        if len(indices_shape) != 0:
            indices = _manipulation._reshape(
                indices,
                (1,) * li + indices_shape + (1,) * (ndim - (ri + 1)))
        for i in range(li):
            ldim *= a._shape[i]
        for i in range(ri + 1, ndim):
            rdim *= a._shape[i]
        index_range = a.size // (ldim * rdim)

    if out is None:
        out = ndarray(out_shape, dtype=a.dtype)
    else:
        if out.dtype != a.dtype:
            raise TypeError('Output dtype mismatch')
        if out.shape != out_shape:
            raise ValueError('Output shape mismatch')
    if a.size == 0 and out.size != 0:
        raise IndexError('cannot do a non-empty take from an empty axes.')

    if isinstance(indices, ndarray):
        return _take_kernel(
            a.reduced_view(), indices, ldim, cdim, rdim, index_range, out)
    else:
        return _take_kernel_scalar(
            a.reduced_view(), indices, ldim, cdim, rdim, index_range, out)


cdef _scatter_op_single(
        ndarray a, ndarray indices, v, Py_ssize_t li=0, Py_ssize_t ri=0,
        op=''):
    # When op == 'update', this function behaves similarly to
    # a code below using NumPy under the condition that a = a._reshape(shape)
    # does not invoke copy.
    #
    # shape = a[:li] +\
    #     (numpy.prod(a[li:ri+1]),) + a[ri+1:]
    # a = a._reshape(shape)
    # slices = (slice(None),) * li + indices +\
    #     (slice(None),) * (a.ndim - indices.ndim - ri)
    # a[slices] = v
    cdef Py_ssize_t ndim, adim, cdim, rdim
    cdef tuple a_shape, indices_shape, lshape, rshape, v_shape

    ndim = a._shape.size()

    if ndim == 0:
        raise ValueError('requires a.ndim >= 1')
    if not (-ndim <= li < ndim and -ndim <= ri < ndim):
        raise ValueError('Axis overrun')

    if not isinstance(v, ndarray):
        v = core.array(v, dtype=a.dtype)
    else:
        v = v.astype(a.dtype, copy=False)

    a_shape = a.shape
    li %= ndim
    ri %= ndim

    lshape = a_shape[:li]
    rshape = a_shape[ri + 1:]
    adim = internal.prod_sequence(a_shape[li:ri + 1])

    indices_shape = indices.shape
    v_shape = lshape + indices_shape + rshape
    v = _manipulation.broadcast_to(v, v_shape)

    cdim = indices.size
    rdim = internal.prod_sequence(rshape)
    indices = _manipulation._reshape(
        indices,
        (1,) * len(lshape) + indices_shape + (1,) * len(rshape))
    indices = _manipulation.broadcast_to(indices, v_shape)

    if op == 'update':
        _scatter_update_kernel(
            v, indices, cdim, rdim, adim, a.reduced_view())
    elif op == 'add':
        # There is constraints on types because atomicAdd() in CUDA 7.5
        # only supports int32, uint32, uint64, and float32.
        if not issubclass(v.dtype.type,
                          (numpy.int32, numpy.float16, numpy.float32,
                           numpy.float64, numpy.uint32, numpy.uint64,
                           numpy.ulonglong)):
            raise TypeError(
                'scatter_add only supports int32, float16, float32, float64, '
                'uint32, uint64, as data type')
        _scatter_add_kernel(
            v, indices, cdim, rdim, adim, a.reduced_view())
    elif op == 'max':
        if not issubclass(v.dtype.type,
                          (numpy.int32, numpy.float32, numpy.float64,
                           numpy.uint32, numpy.uint64, numpy.ulonglong)):
            raise TypeError(
                'scatter_max only supports int32, float32, float64, '
                'uint32, uint64 as data type')
        _scatter_max_kernel(
            v, indices, cdim, rdim, adim, a.reduced_view())
    elif op == 'min':
        if not issubclass(v.dtype.type,
                          (numpy.int32, numpy.float32, numpy.float64,
                           numpy.uint32, numpy.uint64, numpy.ulonglong)):
            raise TypeError(
                'scatter_min only supports int32, float32, float64, '
                'uint32, uint64 as data type')
        _scatter_min_kernel(
            v, indices, cdim, rdim, adim, a.reduced_view())
    else:
        raise ValueError('provided op is not supported')


cdef _scatter_op_mask_single(ndarray a, ndarray mask, v, Py_ssize_t axis, op):
    cdef ndarray mask_scanned, src
    cdef tuple masked_shape

    mask, mask_scanned, masked_shape = _prepare_mask_indexing_single(
        a, mask, axis)
    if internal.prod(masked_shape) == 0:
        return

    if not isinstance(v, ndarray):
        src = core.array(v, dtype=a.dtype)
    else:
        src = v
        # Cython's static resolution does not work because of omitted arguments
        src = (<object>src).astype(a.dtype, copy=False)
    # broadcast src to shape determined by the mask
    src = _manipulation.broadcast_to(src, masked_shape)

    if op == 'update':
        _scatter_update_mask_kernel(src, mask, mask_scanned, a)
    elif op == 'add':
        _scatter_add_mask_kernel(src, mask, mask_scanned, a)
    elif op == 'max':
        _scatter_max_mask_kernel(src, mask, mask_scanned, a)
    elif op == 'min':
        _scatter_min_mask_kernel(src, mask, mask_scanned, a)
    else:
        raise ValueError('provided op is not supported')


cdef _scatter_op(ndarray a, slices, value, op):
    cdef Py_ssize_t i, li, ri
    cdef ndarray v, x, y, a_interm, reduced_idx
    cdef list slice_list, adv_mask, adv_slices
    cdef bint advanced, mask_exists

    slice_list, advanced, mask_exists = _prepare_slice_list(
        slices, a._shape.size())

    if mask_exists:
        mask_i = _get_mask_index(slice_list)
        _scatter_op_mask_single(a, slice_list[mask_i], value, mask_i, op)
        return

    if advanced:
        a, adv_slices, adv_mask = _prepare_advanced_indexing(a, slice_list)
        if sum(adv_mask) == 1:
            axis = adv_mask.index(True)
            _scatter_op_single(a, adv_slices[axis], value, axis, axis, op)
            return

        # scatter_op with multiple integer arrays
        a_interm, reduced_idx, li, ri =\
            _prepare_multiple_array_indexing(a, adv_slices)
        _scatter_op_single(a_interm, reduced_idx, value, li, ri, op)
        return

    y = _simple_getitem(a, slice_list)
    if op == 'update':
        if not isinstance(value, ndarray):
            y.fill(value)
            return
        x = value
        if (internal.vector_equal(y._shape, x._shape) and
                internal.vector_equal(y._strides, x._strides)):
            if y.data.ptr == x.data.ptr:
                return  # Skip since x and y are the same array
            elif y._c_contiguous and x.dtype == y.dtype:
                y.data.copy_from_device_async(x.data, x.nbytes)
                return
        elementwise_copy(x, y)
        return
    if op == 'add':
        _math._add(y, value, y)
        return
    if op == 'max':
        cupy.maximum(y, value, y)
        return
    if op == 'min':
        cupy.minimum(y, value, y)
        return
    raise ValueError('this op is not supported')


cdef ndarray _diagonal(
        ndarray a, Py_ssize_t offset=0, Py_ssize_t axis1=0,
        Py_ssize_t axis2=1):
    cdef Py_ssize_t ndim = a.ndim
    if not (-ndim <= axis1 < ndim and -ndim <= axis2 < ndim):
        raise _errors._AxisError(
            'axis1(={0}) and axis2(={1}) must be within range '
            '(ndim={2})'.format(axis1, axis2, ndim))

    axis1 %= ndim
    axis2 %= ndim
    if axis1 < axis2:
        min_axis, max_axis = axis1, axis2
    else:
        min_axis, max_axis = axis2, axis1

    tr = list(range(ndim))
    del tr[max_axis]
    del tr[min_axis]
    if offset >= 0:
        a = _manipulation._transpose(a, tr + [axis1, axis2])
    else:
        a = _manipulation._transpose(a, tr + [axis2, axis1])
        offset = -offset

    diag_size = max(0, min(a.shape[-2], a.shape[-1] - offset))
    ret_shape = a.shape[:-2] + (diag_size,)
    if diag_size == 0:
        return ndarray(ret_shape, dtype=a.dtype)

    a = a[..., :diag_size, offset:offset + diag_size]

    ret = a.view()
    # TODO(niboshi): Confirm update_x_contiguity flags
    ret._set_shape_and_strides(
        a.shape[:-2] + (diag_size,),
        a.strides[:-2] + (a.strides[-1] + a.strides[-2],),
        True, True)
    return ret


_prepare_array_indexing = ElementwiseKernel(
    'T s, S len, S stride',
    'S out',
    'S in0 = s, in1 = len;'
    'out += stride * (in0 - _floor_divide(in0, in1) * in1)',
    'cupy_prepare_array_indexing')


cdef tuple _prepare_multiple_array_indexing(ndarray a, list slices):
    # slices consist of either slice(None) or ndarray
    cdef Py_ssize_t i, p, li, ri, stride, prev_arr_i
    cdef ndarray reduced_idx
    cdef bint do_transpose

    br = _manipulation.broadcast(*slices)
    slices = list(br.values)

    # check if transpose is necessasry
    # li:  index of the leftmost array in slices
    # ri:  index of the rightmost array in slices
    do_transpose = False
    prev_arr_i = -1
    li = 0
    ri = 0
    for i, s in enumerate(slices):
        if isinstance(s, ndarray):
            if prev_arr_i == -1:
                prev_arr_i = i
                li = i
            elif i - prev_arr_i > 1:
                do_transpose = True
            else:
                prev_arr_i = i
                ri = i

    if do_transpose:
        transp_a = []
        transp_b = []
        slices_a = []
        slices_b = []

        for i, s in enumerate(slices):
            if isinstance(s, ndarray):
                transp_a.append(i)
                slices_a.append(s)
            else:
                transp_b.append(i)
                slices_b.append(s)
        a = _manipulation._transpose(a, transp_a + transp_b)
        slices = slices_a + slices_b
        li = 0
        ri = len(transp_a) - 1

    reduced_idx = ndarray(br.shape, dtype=numpy.int64)
    reduced_idx.fill(0)
    stride = 1
    for i in range(ri, li - 1, -1):
        s = slices[i]
        a_shape_i = a._shape[i]
        # wrap all out-of-bound indices
        if a_shape_i != 0:
            _prepare_array_indexing(s, a_shape_i, stride, reduced_idx)
        stride *= a_shape_i

    return a, reduced_idx, li, ri


cdef ndarray _getitem_multiple(ndarray a, list slices):
    a, reduced_idx, li, ri = _prepare_multiple_array_indexing(a, slices)
    return _take(a, reduced_idx, li, ri)

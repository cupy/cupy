# distutils: language = c++
import warnings
import string

import numpy

import cupy
import cupy._core.core as core
from cupy.exceptions import AxisError
from cupy._core._kernel import ElementwiseKernel, _get_warpsize
from cupy._core._ufuncs import elementwise_copy

from libcpp cimport vector

from cupy._core._carray cimport shape_t
from cupy._core._carray cimport strides_t
from cupy._core cimport core
from cupy._core cimport _routines_math as _math
from cupy._core cimport _routines_manipulation as _manipulation
from cupy._core.core cimport _ndarray_base
from cupy._core cimport internal


# _ndarray_base members


cdef _ndarray_base _ndarray_getitem(_ndarray_base self, slices):
    cdef Py_ssize_t axis
    cdef list slice_list
    cdef _ndarray_base a

    slice_list = _prepare_slice_list(slices)
    a, adv = _view_getitem(self, slice_list)
    if adv is None:
        return a

    axis = adv
    if len(slice_list) == 1:
        s = slice_list[0]
        if s.dtype.kind == 'b':
            return _getitem_mask_single(a, s, axis)
        else:
            return a.take(s, axis)

    return _getitem_multiple(a, axis, slice_list)


cdef _ndarray_setitem(_ndarray_base self, slices, value):
    if isinstance(value, _ndarray_base):
        value = _squeeze_leading_unit_dims(value)
    _scatter_op(self, slices, value, 'update')


cdef tuple _ndarray_nonzero(_ndarray_base self):
    cdef int ndim
    cdef _ndarray_base dst = _ndarray_argwhere(self)
    ndim = self.ndim
    if ndim >= 1:
        return tuple([dst[:, i] for i in range(ndim)])
    else:
        warnings.warn(
            'calling nonzero on 0d arrays is deprecated',
            DeprecationWarning)
        return cupy.zeros(dst.shape[0], numpy.int64),


# TODO(kataoka): Rename the function because `_ndarray_base` does not have
# `argwhere` method
cpdef _ndarray_base _ndarray_argwhere(_ndarray_base self):
    cdef Py_ssize_t count_nonzero
    cdef int ndim
    cdef _ndarray_base nonzero
    numpy_int64 = numpy.int64
    if self.size == 0:
        count_nonzero = 0
    else:
        if self.dtype == numpy.bool_:
            nonzero = self.ravel()
        else:
            nonzero = cupy._core.not_equal(self, 0)
            nonzero = nonzero.ravel()

        # Get number of True in the mask to determine the shape of the array
        # after masking.
        if nonzero.size <= 2 ** 31 - 1:
            scan_dtype = numpy.int32
        else:
            scan_dtype = numpy_int64

        chunk_size = 512

        # TODO(anaruse): Use Optuna to automatically tune the threshold
        # that determines whether "incomplete scan" is enabled or not.
        # Basically, "incomplete scan" is fast when the array size is large,
        # but for small arrays, it is better to use the normal method.
        incomplete_scan = nonzero.size > chunk_size

        scan_index = _math.scan(
            nonzero, op=_math.scan_op.SCAN_SUM, dtype=scan_dtype, out=None,
            incomplete=incomplete_scan, chunk_size=chunk_size)
        count_nonzero = int(scan_index[-1])  # synchronize!

    ndim = self._shape.size()
    dst = core.ndarray((count_nonzero, ndim), dtype=numpy_int64)
    if dst.size == 0:
        return dst

    nonzero.shape = self.shape
    if incomplete_scan:
        warp_size = _get_warpsize()
        size = scan_index.size * chunk_size
        _nonzero_kernel_incomplete_scan(chunk_size, warp_size)(
            nonzero, scan_index, dst,
            size=size, block_size=chunk_size)
    else:
        scan_index.shape = self.shape
        _nonzero_kernel(nonzero, scan_index, dst)

    return dst


cdef _ndarray_base _ndarray_take(_ndarray_base self, indices, axis, out):
    cdef Py_ssize_t ndim = self._shape.size()
    if axis is None:
        return _take(self, indices, 0, ndim, out)
    elif ndim == 0:
        # check axis after atleast_1d
        internal._normalize_axis_index(axis, 1)
        return _take(self, indices, 0, 0, out)
    else:
        axis = internal._normalize_axis_index(axis, ndim)
        return _take(self, indices, axis, axis + 1, out)


cdef _ndarray_base _ndarray_put(_ndarray_base self, indices, values, mode):
    if mode not in ('raise', 'wrap', 'clip'):
        raise ValueError('clipmode not understood')

    n = self.size
    if not isinstance(indices, _ndarray_base):
        indices = core.array(indices)
    indices = indices.ravel()

    if not isinstance(values, _ndarray_base):
        values = core.array(values, dtype=self.dtype)
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


cdef _ndarray_base _ndarray_choose(_ndarray_base self, choices, out, mode):
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
        out = core.ndarray(ba.shape[1:], choices.dtype)

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
        raise ValueError('clipmode not understood')

    return out


cdef _ndarray_base _ndarray_compress(_ndarray_base self, condition, axis, out):
    a = self

    if numpy.isscalar(condition):
        raise ValueError('condition must be a 1-d array')

    if not isinstance(condition, _ndarray_base):
        condition = core.array(condition, dtype=int)
        if condition.ndim != 1:
            raise ValueError('condition must be a 1-d array')

    # do not test condition.shape
    res = _ndarray_nonzero(condition)  # synchronize

    # the `take` method/function also make the input atleast_1d
    return _ndarray_take(a, res[0], axis, out)


cdef _ndarray_base _ndarray_diagonal(_ndarray_base self, offset, axis1, axis2):
    return _diagonal(self, offset, axis1, axis2)


# private/internal


cdef _ndarray_base _squeeze_leading_unit_dims(_ndarray_base src):
    # remove leading 1s from the shape greedily.
    # TODO(kataoka): compute requested ndim and do not remove too much for
    # printing correct shape in error message.
    cdef Py_ssize_t i
    for i in range(src.ndim):
        if src._shape[i] != 1:
            break
    else:
        i = src.ndim

    if i == 0:
        return src

    src = src.view()
    # del src._shape[:i]
    # del src._strides[:i]
    src._shape.erase(src._shape.begin(), src._shape.begin()+i)
    src._strides.erase(src._strides.begin(), src._strides.begin()+i)
    return src


cpdef list _prepare_slice_list(slices):
    cdef Py_ssize_t i
    cdef list slice_list
    cdef bint fix_empty_dtype

    if isinstance(slices, tuple):
        slice_list = list(slices)
    else:
        slice_list = [slices]

    # Convert list/NumPy/CUDA-Array-Interface arrays to cupy.ndarray.
    # - Scalar int in indices returns a view.
    # - Other array-like (including ()-shaped array) in indices forces to
    #   return a new array.
    for i, s in enumerate(slice_list):
        if s is None or s is Ellipsis or isinstance(s, (slice, _ndarray_base)):
            continue

        fix_empty_dtype = False
        if isinstance(s, (list, tuple)):
            # This condition looks inaccurate, but so is NumPy.
            # a[1, [np.empty(0, float)]] is allowed, while
            # a[1, np.empty((1, 0), float)] raises IndexError.
            fix_empty_dtype = True
        elif numpy.isscalar(s):
            if not isinstance(s, (bool, numpy.bool_)):
                # keep scalar int
                continue

        if cupy.min_scalar_type(s).char == 'O':
            raise IndexError(
                'arrays used as indices must be of integer (or boolean) type')
        try:
            s = core.array(s, dtype=None, copy=False)
        except ValueError:
            # "Unsupported dtype"
            raise IndexError(
                'only integers, slices (`:`), ellipsis (`...`),'
                'numpy.newaxis (`None`) and integer or '
                'boolean arrays are valid indices')
        if fix_empty_dtype and s.size == 0:
            # An empty list means empty indices, not empty mask.
            # Fix default dtype (float64).
            s = s.astype(numpy.int32)
        slice_list[i] = s

    return slice_list


cdef tuple _view_getitem(_ndarray_base a, list slice_list):
    # Process scalar/slice/ellipsis indices
    # Returns a 2-tuple
    # - [0] (ndarray): view of a
    # - [1] (int or None): start axis for remaining indices
    # slice_list will be overwritten.
    #     input should contain:
    #         None, Ellipsis, slice (start:stop:step), scalar int, or
    #         cupy.ndarray
    #     output will contain:
    #         cupy.ndarray
    cdef shape_t shape
    cdef strides_t strides
    cdef _ndarray_base v
    cdef Py_ssize_t ndim_a, axis_a, ndim_v, axis_v, ndim_ellipsis
    cdef Py_ssize_t i, k, offset
    cdef Py_ssize_t s_start, s_stop, s_step, dim, ind
    cdef slice ss
    cdef list index_list, axes
    cdef vector.vector[bint] array_like_flags
    cdef vector.vector[Py_ssize_t] array_ndims
    cdef bint has_ellipsis, flag
    cdef char kind

    axis_a = 0
    has_ellipsis = False
    for s in slice_list:
        if s is None:
            continue
        elif s is Ellipsis:
            if has_ellipsis:
                raise IndexError(
                    "an index can only have a single ellipsis ('...')")
            has_ellipsis = True
        elif isinstance(s, _ndarray_base):
            kind = ord(s.dtype.kind)
            if kind == b'b':
                k = s.ndim
            elif kind == b'i' or kind == b'u':
                k = 1
            else:
                raise IndexError(
                    'arrays used as indices must be of integer or boolean '
                    'type. (actual: {})'.format(s.dtype.type))
            array_ndims.push_back(k)
            axis_a += k
        else:
            # isinstance(s, slice) or numpy.isscalar(s)
            axis_a += 1
    if not has_ellipsis:
        slice_list.append(Ellipsis)

    ndim_a = a._shape.size()
    if axis_a > ndim_a:
        raise IndexError(
            'too many indices for array: '
            f'array is {ndim_a}-dimensional, but {axis_a} were indexed')
    ndim_ellipsis = ndim_a - axis_a

    # Create new shape and stride
    i = 0
    axis_a = 0
    axis_v = 0
    offset = 0
    # index_list: remaining indices to be processed.
    # Each elem is a 3-tuple (array, axis_start, axis_count)
    index_list = []
    for s in slice_list:
        if s is None:
            shape.push_back(1)
            strides.push_back(0)
            axis_v += 1
            array_like_flags.push_back(False)
        elif isinstance(s, _ndarray_base):
            k = array_ndims[i]
            index_list.append((s, axis_v, k))
            i += 1
            kind = ord(s.dtype.kind)
            if kind == b'b':
                _check_mask_shape(a, s, axis_a)
            for _ in range(k):
                shape.push_back(a._shape[axis_a])
                strides.push_back(a._strides[axis_a])
                axis_a += 1
            axis_v += k
            array_like_flags.push_back(True)
        elif s is Ellipsis:
            for _ in range(ndim_ellipsis):
                shape.push_back(a._shape[axis_a])
                strides.push_back(a._strides[axis_a])
                axis_a += 1
            axis_v += ndim_ellipsis
            array_like_flags.push_back(False)
        elif isinstance(s, slice):
            ss = internal.complete_slice(s, a._shape[axis_a])
            s_start = ss.start
            s_stop = ss.stop
            s_step = ss.step
            if s_step > 0:
                dim = (s_stop - s_start - 1) // s_step + 1
            else:
                dim = (s_stop - s_start + 1) // s_step + 1

            if dim == 0:
                strides.push_back(a._strides[axis_a])
            else:
                strides.push_back(a._strides[axis_a] * s_step)

            if s_start > 0:
                offset += a._strides[axis_a] * s_start
            shape.push_back(dim)
            axis_a += 1
            axis_v += 1
            array_like_flags.push_back(False)
        else:
            # numpy.isscalar(s)
            ind = int(s)
            if ind < 0:
                ind += a._shape[axis_a]
            if not (0 <= ind < a._shape[axis_a]):
                msg = ('Index %s is out of bounds for axis %s with '
                       'size %s' % (s, axis_a, a._shape[axis_a]))
                raise IndexError(msg)
            offset += ind * a._strides[axis_a]
            axis_a += 1
            # array-like but not array
            array_like_flags.push_back(True)

    ndim_v = axis_v
    v = a.view()
    if a.size != 0:
        v.data = a.data + offset
    v._set_shape_and_strides(shape, strides, True, True)

    if array_ndims.empty():
        # no advanced indexing. no mask.
        del slice_list[:]
        return v, None

    slice_list[:] = [s for s, _, _ in index_list]

    # non-consecutive array-like indices => batch dims go first in output
    # consecutive array-like indices => start batch dims there
    k = 0
    for i, flag in enumerate(array_like_flags):
        if k == 0:
            if flag:
                k = 1
        elif k == 1:
            if not flag:
                k = 2
        else:  # k == 2
            if flag:
                break
    else:
        return v, index_list[0][1]

    # compute transpose arg
    axes = []
    for _, axis_v, k in index_list:
        for _ in range(k):
            axes.append(axis_v)
            axis_v += 1
    axes.extend([dim for dim in range(ndim_v) if dim not in axes])
    v = _manipulation._transpose(v, axes)
    return v, 0


@cupy._util.memoize(for_each_device=True)
def _nonzero_kernel_incomplete_scan(block_size, warp_size=32):
    in_params = 'raw T a, raw S b'
    out_params = 'raw O dst'
    loop_prep = string.Template("""
        __shared__ S smem[${warp_size}];
        const int n_warp = ${block_size} / ${warp_size};
        const int warp_id = threadIdx.x / ${warp_size};
        const int lane_id = threadIdx.x % ${warp_size};
    """).substitute(block_size=block_size, warp_size=warp_size)
    loop_body = string.Template("""
        S x = 0;
        if (i < a.size()) x = a[i];
        for (int j = 1; j < ${warp_size}; j *= 2) {
            S tmp = __shfl_up_sync(0xffffffff, x, j, ${warp_size});
            if (lane_id - j >= 0) x += tmp;
        }
        if (lane_id == ${warp_size} - 1) smem[warp_id] = x;
        __syncthreads();
        if (warp_id == 0) {
            S y = 0;
            if (lane_id < n_warp) y = smem[lane_id];
            for (int j = 1; j < n_warp; j *= 2) {
                S tmp = __shfl_up_sync(0xffffffff, y, j, ${warp_size});
                if (lane_id - j >= 0) y += tmp;
            }
            int block_id = i / ${block_size};
            S base = 0;
            if (block_id > 0) base = b[block_id - 1];
            if (lane_id == ${warp_size} - 1) y = 0;
            smem[(lane_id + 1) % ${warp_size}] = y + base;
        }
        __syncthreads();
        x += smem[warp_id];
        S x0 = __shfl_up_sync(0xffffffff, x, 1, ${warp_size});
        if (lane_id == 0) {
            x0 = smem[warp_id];
        }
        if (x0 < x && i < a.size()) {
            O j = i;
            for (int d = a.ndim - 1; d >= 0; d--) {
                ptrdiff_t ind[] = {x0, d};
                O j_next = j / a.shape()[d];
                dst[ind] = j - j_next * a.shape()[d];
                j = j_next;
            }
        }
    """).substitute(block_size=block_size, warp_size=warp_size)
    return cupy.ElementwiseKernel(in_params, out_params, loop_body,
                                  'cupy_nonzero_kernel_incomplete_scan',
                                  loop_prep=loop_prep)


_nonzero_kernel = ElementwiseKernel(
    'T src, S index', 'raw U dst',
    '''
    if (src != 0){
        for(int j = 0; j < _ind.ndim; j++){
            ptrdiff_t ind[] = {index - 1, j};
            dst[ind] = _ind.get()[j];
        }
    }''',
    'cupy_nonzero_kernel',
    reduce_dims=False)


_take_kernel_core = '''
ptrdiff_t out_i = indices % index_range;
if (out_i < 0) out_i += index_range;
if (ldim != 1) out_i += (i / (cdim * rdim)) * index_range;
if (rdim != 1) out_i = out_i * rdim + i % rdim;
out = a[out_i];
'''


_take_kernel = ElementwiseKernel(
    'raw T a, S indices, uint32 ldim, uint32 cdim, uint32 rdim, '
    'int64 index_range',
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


cdef _create_scatter_kernel(name, code):
    return ElementwiseKernel(
        'T v, S indices, int32 cdim, int32 rdim, int32 adim',
        'raw T a',
        string.Template('''
            S wrap_indices = indices % adim;
            if (wrap_indices < 0) wrap_indices += adim;
            ptrdiff_t li = i / (rdim * cdim);
            ptrdiff_t ri = i % rdim;
            T &out0 = a[(li * adim + wrap_indices) * rdim + ri];
            T &in0 = out0;
            const T &in1 = v;
            ${code};
        ''').substitute(code=code),
        name,
    )


cdef _scatter_update_kernel = _create_scatter_kernel(
    'cupy_scatter_update', 'out0 = in1')

cdef _scatter_add_kernel = _create_scatter_kernel(
    'cupy_scatter_add', 'atomicAdd(&out0, in1)')

cdef _scatter_sub_kernel = _create_scatter_kernel(
    'cupy_scatter_sub', 'atomicSub(&out0, in1)')

cdef _scatter_max_kernel = _create_scatter_kernel(
    'cupy_scatter_max', 'atomicMax(&out0, in1)')

cdef _scatter_min_kernel = _create_scatter_kernel(
    'cupy_scatter_min', 'atomicMin(&out0, in1)')

cdef _scatter_and_kernel = _create_scatter_kernel(
    'cupy_scatter_and', 'atomicAnd(&out0, in1)')

cdef _scatter_or_kernel = _create_scatter_kernel(
    'cupy_scatter_or', 'atomicOr(&out0, in1)')

cdef _scatter_xor_kernel = _create_scatter_kernel(
    'cupy_scatter_xor', 'atomicXor(&out0, in1)')


cdef _create_scatter_mask_kernel(name, code):
    return ElementwiseKernel(
        'raw T v, bool mask, S mask_scanned',
        'T a',
        string.Template('''
            T &out0 = a;
            T &in0 = a;
            const T &in1 = v[mask_scanned - 1];
            if (mask) ${code};
        ''').substitute(code=code),
        name,
    )


cdef _scatter_update_mask_kernel = _create_scatter_mask_kernel(
    'cupy_scatter_update_mask', 'out0 = in1')

cdef _scatter_add_mask_kernel = _create_scatter_mask_kernel(
    'cupy_scatter_add_mask', 'out0 = in0 + in1')

cdef _scatter_sub_mask_kernel = _create_scatter_mask_kernel(
    'cupy_scatter_add_mask', 'out0 = in0 - in1')

cdef _scatter_max_mask_kernel = _create_scatter_mask_kernel(
    'cupy_scatter_max_mask', 'out0 = max(in0, in1)')

cdef _scatter_min_mask_kernel = _create_scatter_mask_kernel(
    'cupy_scatter_min_mask', 'out0 = min(in0, in1)')

cdef _scatter_and_mask_kernel = _create_scatter_mask_kernel(
    'cupy_scatter_and_mask', 'out0 = (in0 & in1)')

cdef _scatter_or_mask_kernel = _create_scatter_mask_kernel(
    'cupy_scatter_or_mask', 'out0 = (in0 | in1)')

cdef _scatter_xor_mask_kernel = _create_scatter_mask_kernel(
    'cupy_scatter_xor_mask', 'out0 = (in0 ^ in1)')


_getitem_mask_kernel = ElementwiseKernel(
    'T a, bool mask, S mask_scanned',
    'raw T out',
    'if (mask) out[mask_scanned - 1] = a',
    'cupy_getitem_mask')


cdef _check_mask_shape(_ndarray_base a, _ndarray_base mask, Py_ssize_t axis):
    cdef Py_ssize_t i, a_sh, m_sh
    for i, m_sh in enumerate(mask._shape):
        a_sh = a._shape[axis + i]
        if m_sh not in (0, a_sh):
            raise IndexError(
                'boolean index did not match indexed array along dimension '
                f'{axis + i}; dimension is {a_sh} '
                f'but corresponding boolean dimension is {m_sh}'
            )


cpdef _prepare_mask_indexing_single(
        _ndarray_base a, _ndarray_base mask, Py_ssize_t axis):
    cdef _ndarray_base mask_scanned, mask_br
    cdef int n_true
    cdef tuple lshape, rshape, a_shape
    cdef Py_ssize_t a_ndim, mask_ndim

    a_ndim = a._shape.size()
    mask_ndim = mask._shape.size()
    a_shape = a.shape
    lshape = a_shape[:axis]
    rshape = a_shape[axis + mask._shape.size():]

    if mask.size == 0:
        masked_shape = lshape + (0,) + rshape
        mask_br = _manipulation._reshape(mask, masked_shape)
        return mask_br, mask_br, masked_shape

    # Get number of True in the mask to determine the shape of the array
    # after masking.
    if mask.size <= 2 ** 31 - 1:
        mask_type = numpy.int32
    else:
        mask_type = numpy.int64
    op = _math.scan_op.SCAN_SUM

    # starts with 1
    mask_scanned = _math.scan(mask.ravel(), op=op, dtype=mask_type)
    n_true = int(mask_scanned[-1])
    masked_shape = lshape + (n_true,) + rshape
    # When mask covers the entire array, broadcasting is not necessary.
    if mask_ndim == a_ndim and axis == 0:
        return (
            mask,
            _manipulation._reshape(mask_scanned, mask._shape),
            masked_shape)
    mask_scanned = None

    # The scan of the broadcasted array is used to index on kernel.
    mask = _manipulation._reshape(
        mask,
        axis * (1,) + mask.shape + (a_ndim - axis - mask_ndim) * (1,))
    if <Py_ssize_t>mask._shape.size() > a_ndim:
        raise IndexError('too many indices for array')

    mask = _manipulation.broadcast_to(mask, a_shape)
    if mask.size <= 2 ** 31 - 1:
        mask_type = numpy.int32
    else:
        mask_type = numpy.int64
    mask_scanned = _manipulation._reshape(
        _math.scan(mask.ravel(), op=_math.scan_op.SCAN_SUM, dtype=mask_type),
        mask._shape)
    return mask, mask_scanned, masked_shape


cpdef _ndarray_base _getitem_mask_single(
        _ndarray_base a, _ndarray_base mask, int axis):
    cdef _ndarray_base mask_scanned
    cdef tuple masked_shape

    mask, mask_scanned, masked_shape = _prepare_mask_indexing_single(
        a, mask, axis)
    out = core.ndarray(masked_shape, dtype=a.dtype)
    if out.size == 0:
        return out
    return _getitem_mask_kernel(a, mask, mask_scanned, out)


cdef _ndarray_base _take(
        _ndarray_base a, indices, int start, int stop, _ndarray_base out=None):
    # Take along (flattened) axes from start to stop.
    # When start + 1 == stop this function behaves similarly to np.take
    cdef tuple out_shape, indices_shape
    cdef int i, ndim = a._shape.size()
    cdef Py_ssize_t ldim, cdim, rdim, index_range

    assert start <= stop

    if numpy.isscalar(indices):
        indices_shape = ()
        cdim = 1
    else:
        if not isinstance(indices, _ndarray_base):
            indices = core.array(indices, dtype=int)
        indices_shape = indices.shape
        cdim = indices.size

    ldim = rdim = 1
    if start == 0 and stop == ndim:
        out_shape = indices_shape
        index_range = a.size
    else:
        a_shape = a.shape
        out_shape = a_shape[:start] + indices_shape + a_shape[stop:]
        if len(indices_shape) != 0:
            indices = _manipulation._reshape(
                indices,
                (1,) * start + indices_shape + (1,) * (ndim - stop))
        for i in range(start):
            ldim *= a._shape[i]
        for i in range(stop, ndim):
            rdim *= a._shape[i]
        index_range = 1
        for i in range(start, stop):
            index_range *= a._shape[i]

    if out is None:
        out = core.ndarray(out_shape, dtype=a.dtype)
    else:
        if out.dtype != a.dtype:
            raise TypeError('Output dtype mismatch')
        if out.shape != out_shape:
            raise ValueError('Output shape mismatch')
    if a.size == 0 and out.size != 0:
        raise IndexError('cannot do a non-empty take from an empty axes.')

    if isinstance(indices, _ndarray_base):
        return _take_kernel(
            a.reduced_view(), indices, ldim, cdim, rdim, index_range, out)
    else:
        return _take_kernel_scalar(
            a.reduced_view(), indices, ldim, cdim, rdim, index_range, out)


cdef _scatter_op_single(
        _ndarray_base a, _ndarray_base indices, value, Py_ssize_t start,
        Py_ssize_t stop, op=''):
    # When op == 'update', this function behaves similarly to
    # a code below using NumPy under the condition that a = a._reshape(shape)
    # does not invoke copy.
    #
    # shape = a[:start] +\
    #     (numpy.prod(a[start:stop]),) + a[stop:]
    # a = a._reshape(shape)
    # slices = (slice(None),) * start + indices +\
    #     (slice(None),) * (a.ndim - stop)
    # a[slices] = value
    cdef Py_ssize_t adim, cdim, rdim
    cdef tuple a_shape, indices_shape, lshape, rshape, v_shape
    cdef _ndarray_base v

    if not isinstance(value, _ndarray_base):
        v = core.array(value, dtype=a.dtype)
    else:
        v = value.astype(a.dtype, copy=False)

    a_shape = a.shape

    lshape = a_shape[:start]
    rshape = a_shape[stop:]
    adim = internal.prod_sequence(a_shape[start:stop])

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
                           numpy.intc, numpy.uintc, numpy.ulonglong)):
            raise TypeError(
                'cupy.add.at only supports int32, float16, float32, float64, '
                'uint32, uint64, as data type')
        _scatter_add_kernel(
            v, indices, cdim, rdim, adim, a.reduced_view())
    elif op == 'sub':
        if not issubclass(v.dtype.type,
                          (numpy.int32, numpy.uint32,
                           numpy.intc, numpy.uintc)):
            raise TypeError(
                'cupy.subtract.at only supports int32, uint32, as data type')
        _scatter_sub_kernel(
            v, indices, cdim, rdim, adim, a.reduced_view())
    elif op == 'max':
        if not issubclass(v.dtype.type,
                          (numpy.int32, numpy.float32, numpy.float64,
                           numpy.uint32, numpy.uint64,
                           numpy.intc, numpy.uintc, numpy.ulonglong)):
            raise TypeError(
                'cupy.maximum.at only supports int32, float32, float64, '
                'uint32, uint64 as data type')
        _scatter_max_kernel(
            v, indices, cdim, rdim, adim, a.reduced_view())
    elif op == 'min':
        if not issubclass(v.dtype.type,
                          (numpy.int32, numpy.float32, numpy.float64,
                           numpy.uint32, numpy.uint64,
                           numpy.intc, numpy.uintc, numpy.ulonglong)):
            raise TypeError(
                'cupy.minimum.at only supports int32, float32, float64, '
                'uint32, uint64 as data type')
        _scatter_min_kernel(
            v, indices, cdim, rdim, adim, a.reduced_view())
    elif op == 'and':
        if not issubclass(v.dtype.type,
                          (numpy.int32, numpy.int64,
                           numpy.uint32, numpy.uint64,
                           numpy.intc, numpy.uintc,
                           numpy.longlong, numpy.ulonglong)):
            raise TypeError(
                'cupy.bitwise_and.at only supports int32, int64, '
                'uint32, uint64 as data type')
        _scatter_and_kernel(
            v, indices, cdim, rdim, adim, a.reduced_view())
    elif op == 'or':
        if not issubclass(v.dtype.type,
                          (numpy.int32, numpy.int64,
                           numpy.uint32, numpy.uint64,
                           numpy.intc, numpy.uintc,
                           numpy.longlong, numpy.ulonglong)):
            raise TypeError(
                'cupy.bitwise_or.at only supports int32, int64, '
                'uint32, uint64 as data type')
        _scatter_or_kernel(
            v, indices, cdim, rdim, adim, a.reduced_view())
    elif op == 'xor':
        if not issubclass(v.dtype.type,
                          (numpy.int32, numpy.int64,
                           numpy.uint32, numpy.uint64,
                           numpy.intc, numpy.uintc,
                           numpy.longlong, numpy.ulonglong)):
            raise TypeError(
                'cupy.bitwise_xor.at only supports int32, int64, '
                'uint32, uint64 as data type')
        _scatter_xor_kernel(
            v, indices, cdim, rdim, adim, a.reduced_view())
    else:
        raise ValueError('provided op is not supported')


cdef _scatter_op_mask_single(
        _ndarray_base a, _ndarray_base mask, v, Py_ssize_t axis, op):
    cdef _ndarray_base mask_scanned, src
    cdef tuple masked_shape

    mask, mask_scanned, masked_shape = _prepare_mask_indexing_single(
        a, mask, axis)
    if internal.prod(masked_shape) == 0:
        return

    if not isinstance(v, _ndarray_base):
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
    elif op == 'sub':
        _scatter_sub_mask_kernel(src, mask, mask_scanned, a)
    elif op == 'max':
        _scatter_max_mask_kernel(src, mask, mask_scanned, a)
    elif op == 'min':
        _scatter_min_mask_kernel(src, mask, mask_scanned, a)
    elif op == 'and':
        _scatter_and_mask_kernel(src, mask, mask_scanned, a)
    elif op == 'or':
        _scatter_or_mask_kernel(src, mask, mask_scanned, a)
    elif op == 'xor':
        _scatter_xor_mask_kernel(src, mask, mask_scanned, a)
    else:
        raise ValueError('provided op is not supported')


cdef _scatter_op(_ndarray_base a, slices, value, op):
    cdef Py_ssize_t start, stop, axis
    cdef _ndarray_base x, y, reduced_idx
    cdef list slice_list

    slice_list = _prepare_slice_list(slices)
    a, adv = _view_getitem(a, slice_list)
    if adv is not None:
        axis = adv
        if len(slice_list) == 1:
            s = slice_list[0]
            if s.dtype.kind == 'b':
                _scatter_op_mask_single(a, s, value, axis, op)
            else:
                _scatter_op_single(a, s, value, axis, axis + 1, op)
        else:
            # scatter_op with multiple integer arrays
            reduced_idx, start, stop = _prepare_multiple_array_indexing(
                a, axis, slice_list)
            _scatter_op_single(a, reduced_idx, value, start, stop, op)
        return

    y = a

    if op == 'update':
        if not isinstance(value, _ndarray_base):
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
    if op == 'sub':
        _math._subtract(y, value, y)
        return
    if op == 'max':
        cupy.maximum(y, value, y)
        return
    if op == 'min':
        cupy.minimum(y, value, y)
        return
    if op == 'and':
        cupy.bitwise_and(y, value, y)
        return
    if op == 'or':
        cupy.bitwise_or(y, value, y)
        return
    if op == 'xor':
        cupy.bitwise_xor(y, value, y)
        return
    raise ValueError('this op is not supported')


cdef _ndarray_base _diagonal(
        _ndarray_base a, Py_ssize_t offset=0, Py_ssize_t axis1=0,
        Py_ssize_t axis2=1):
    cdef Py_ssize_t ndim = a.ndim
    if not (-ndim <= axis1 < ndim and -ndim <= axis2 < ndim):
        raise AxisError(
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
        return core.ndarray(ret_shape, dtype=a.dtype)

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


cdef tuple _prepare_multiple_array_indexing(
    _ndarray_base a, Py_ssize_t start, list slices
):
    # slices consist of ndarray
    cdef list indices = [], shapes = []  # int ndarrays
    cdef Py_ssize_t i, stop, stride
    cdef _ndarray_base reduced_idx, s

    for s in slices:
        if s.dtype.kind == 'b':
            s = _ndarray_argwhere(s).T
            indices.extend(s)
            shapes.append(s.shape[1:])
        else:
            indices.append(s)
            shapes.append(s.shape)

    stop = start + len(indices)

    # br = _manipulation.broadcast(*indices)
    # indices = list(br.values)

    reduced_idx = core.ndarray(
        internal._broadcast_shapes(shapes), dtype=numpy.int64)
    reduced_idx.fill(0)
    stride = 1
    i = stop
    for s in reversed(indices):
        i -= 1
        a_shape_i = a._shape[i]
        # wrap all out-of-bound indices
        if a_shape_i != 0:
            _prepare_array_indexing(s, a_shape_i, stride, reduced_idx)
        stride *= a_shape_i

    return reduced_idx, start, stop


cdef _ndarray_base _getitem_multiple(
        _ndarray_base a, Py_ssize_t start, list slices):
    reduced_idx, start, stop = _prepare_multiple_array_indexing(
        a, start, slices)
    return _take(a, reduced_idx, start, stop)


cdef _ndarray_base _add_reduceat(
        _ndarray_base array, indices, axis, dtype, out):
    from cupy._sorting import search
    axis = internal._normalize_axis_index(axis, array.ndim)
    indices = cupy.append(indices, array.shape[axis])
    shape = [1 if i == axis else dim for i, dim in enumerate(array.shape)]
    acc = array.cumsum(axis, dtype)
    acc = cupy.append(cupy.zeros(shape, acc.dtype), acc, axis)
    mask = indices[:-1] >= indices[1:]
    mask = mask.reshape(-1, *([1] * (array.ndim - axis - 1)))
    return search._where_ufunc(
        mask,
        array.take(indices[:-1], axis),
        acc.take(indices[1:], axis) - acc.take(indices[:-1], axis),
        out
    )

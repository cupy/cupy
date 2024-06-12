# distutils: language = c++
import functools

import numpy

from cupy._core._kernel import ElementwiseKernel
from cupy._core._ufuncs import elementwise_copy
import cupy._core.core as core
from cupy.exceptions import AxisError

cimport cpython  # NOQA
cimport cython  # NOQA
from libcpp cimport vector

from cupy._core._dtype cimport get_dtype, _raise_if_invalid_cast
from cupy._core cimport core
from cupy._core.core cimport _ndarray_base
from cupy._core cimport internal
from cupy._core._kernel cimport _check_peer_access, _preprocess_args

from cupy.cuda import device


@cython.final
cdef class broadcast:
    """Object that performs broadcasting.

    CuPy actually uses this class to support broadcasting in various
    operations. Note that this class does not provide an iterator.

    Args:
        arrays (tuple of arrays): Arrays to be broadcasted.

    Attributes:
        ~broadcast.shape (tuple of ints): The broadcasted shape.
        nd (int): Number of dimensions of the broadcasted shape.
        ~broadcast.size (int): Total size of the broadcasted shape.
        values (list of arrays): The broadcasted arrays.

    .. seealso:: :class:`numpy.broadcast`

    """

    def __init__(self, *arrays):
        cdef shape_t shape
        cdef list val = list(arrays)
        internal._broadcast_core(val, shape)
        self.values = tuple(val)
        self.shape = tuple(shape)
        self.nd = <Py_ssize_t>shape.size()
        self.size = internal.prod(shape)


# _ndarray_base members


cdef _ndarray_shape_setter(_ndarray_base self, newshape):
    cdef shape_t shape, strides
    if not cpython.PySequence_Check(newshape):
        newshape = (newshape,)
    shape = internal.infer_unknown_dimension(newshape, self.size)
    _get_strides_for_nocopy_reshape(self, shape, strides)
    if strides.size() != shape.size():
        raise AttributeError(
            'Incompatible shape for in-place modification. Use `.reshape()` '
            'to make a copy with the desired shape.')
    self._set_shape_and_strides(shape, strides, False, True)


cdef _ndarray_base _ndarray_reshape(_ndarray_base self, tuple shape, order):
    cdef int order_char = internal._normalize_order(order, False)

    if len(shape) == 1 and cpython.PySequence_Check(shape[0]):
        shape = tuple(shape[0])

    if order_char == b'A':
        if self._f_contiguous and not self._c_contiguous:
            order_char = b'F'
        else:
            order_char = b'C'
    if order_char == b'C':
        return _reshape(self, shape)
    else:
        # TODO(grlee77): Support order within _reshape instead

        # The Fortran-ordered case is equivalent to:
        #     1.) reverse the axes via transpose
        #     2.) C-ordered reshape using reversed shape
        #     3.) reverse the axes via transpose
        return _T(_reshape(_T(self), shape[::-1]))


cdef _ndarray_base _ndarray_transpose(_ndarray_base self, tuple axes):
    if len(axes) == 0:
        return _T(self)
    if len(axes) == 1:
        a = axes[0]
        if a is None:
            return _T(self)
        elif cpython.PySequence_Check(a):
            axes = tuple(a)
    return _transpose(self, axes)


cdef _ndarray_base _ndarray_swapaxes(
        _ndarray_base self, Py_ssize_t axis1, Py_ssize_t axis2):
    cdef Py_ssize_t ndim = self.ndim
    cdef vector.vector[Py_ssize_t] axes
    if axis1 < -ndim or axis1 >= ndim or axis2 < -ndim or axis2 >= ndim:
        raise ValueError('Axis out of range')
    axis1 %= ndim
    axis2 %= ndim
    for i in range(ndim):
        axes.push_back(i)
    axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
    return _transpose(self, axes)


cdef _ndarray_base _ndarray_flatten(_ndarray_base self, order):
    cdef int order_char
    cdef vector.vector[Py_ssize_t] axes

    order_char = internal._normalize_order(order, True)
    if order_char == b'A':
        if self._f_contiguous and not self._c_contiguous:
            order_char = b'F'
        else:
            order_char = b'C'
    if order_char == b'C':
        return _ndarray_flatten_order_c(self)
    elif order_char == b'F':
        return _ndarray_flatten_order_c(_T(self))
    elif order_char == b'K':
        axes = _npyiter_k_order_axes(self.strides)
        return _ndarray_flatten_order_c(_transpose(self, axes))


cdef _ndarray_base _ndarray_flatten_order_c(_ndarray_base self):
    newarray = self.copy(order='C')
    newarray._shape.assign(<Py_ssize_t>1, self.size)
    newarray._strides.assign(<Py_ssize_t>1,
                             <Py_ssize_t>self.itemsize)
    newarray._c_contiguous = True
    newarray._f_contiguous = True
    return newarray


cdef vector.vector[Py_ssize_t] _npyiter_k_order_axes(strides_t& strides):
    # output transpose axes such that
    # x.flatten(order="K") == x.transpose(axes).flatten(order="C")
    # by reproducing `npyiter_find_best_axis_ordering`
    # in numpy/core/src/multiarray/nditer_constr.c

    # Note that `flatten` and `ravel` should use this function for order="K",
    # while `copy(order="K")` should use `internal._get_strides_for_order_K`.
    cdef vector.vector[Py_ssize_t] axes
    cdef Py_ssize_t stride0, stride1
    cdef int ndim, i0, i1, ipos, k
    ndim = strides.size()
    for i0 in reversed(range(ndim)):
        stride0 = abs(strides[i0])
        if stride0 == 0:  # ambiguous
            axes.insert(axes.begin(), i0)
            continue
        ipos = 0
        for k, i1 in enumerate(axes):
            stride1 = abs(strides[i1])
            if stride1 == 0:  # ambiguous
                continue
            elif stride1 <= stride0:  # shouldswap = false
                break
            else:  # shouldswap = true
                ipos = k + 1
        axes.insert(axes.begin() + ipos, i0)
    return axes


cdef _ndarray_base _ndarray_ravel(_ndarray_base self, order):
    cdef int order_char
    cdef shape_t shape
    cdef vector.vector[Py_ssize_t] axes
    shape.push_back(self.size)

    order_char = internal._normalize_order(order, True)
    if order_char == b'A':
        if self._f_contiguous and not self._c_contiguous:
            order_char = b'F'
        else:
            order_char = b'C'
    if order_char == b'C':
        return _reshape(self, shape)
    elif order_char == b'F':
        return _reshape(_T(self), shape)
    elif order_char == b'K':
        axes = _npyiter_k_order_axes(self.strides)
        return _reshape(_transpose(self, axes), shape)


cdef _ndarray_base _ndarray_squeeze(_ndarray_base self, axis):
    cdef vector.vector[char] axis_flags
    cdef shape_t newshape
    cdef strides_t newstrides
    cdef Py_ssize_t ndim, naxes, _axis

    ndim = self._shape.size()
    axis_flags = vector.vector[char](ndim, 0)

    # Convert axis to boolean flag.
    if axis is None:
        for idim in range(ndim):
            if self._shape[idim] == 1:
                axis_flags[idim] = 1
    elif isinstance(axis, tuple):
        naxes = <Py_ssize_t>len(axis)
        for i in range(naxes):
            _axis = internal._normalize_axis_index(<Py_ssize_t>axis[i], ndim)
            if axis_flags[_axis] == 1:
                raise ValueError('duplicate value in \'axis\'')
            axis_flags[_axis] = 1
    else:
        _axis = <Py_ssize_t>axis
        if ndim == 0 and (_axis == 0 or _axis == -1):
            # Special case letting axis={-1,0} slip through for scalars,
            # for backwards compatibility reasons.
            pass
        else:
            _axis = internal._normalize_axis_index(_axis, ndim)
            axis_flags[_axis] = 1

    # Verify that the axes requested are all of size one
    any_ones = 0
    for idim in range(ndim):
        if axis_flags[idim] != 0:
            if self._shape[idim] == 1:
                any_ones = 1
            else:
                raise ValueError('cannot select an axis to squeeze out '
                                 'which has size not equal to one')

    # If there were no axes to squeeze out, return the same array
    if any_ones == 0:
        return self

    for i in range(ndim):
        if axis_flags[i] == 0:
            newshape.push_back(self._shape[i])
            newstrides.push_back(self._strides[i])

    v = self.view()
    # TODO(niboshi): Confirm update_x_contiguity flags
    v._set_shape_and_strides(newshape, newstrides, False, True)
    return v


cdef _ndarray_base _ndarray_repeat(_ndarray_base self, repeats, axis):
    return _repeat(self, repeats, axis)


# exposed


cpdef _ndarray_base _expand_dims(_ndarray_base a, tuple axis):
    cdef vector.vector[Py_ssize_t] normalized_axis
    cdef out_ndim = a.ndim + len(axis)
    cdef shape_t a_shape = a.shape, out_shape
    _normalize_axis_tuple(axis, out_ndim, normalized_axis)
    out_shape.assign(out_ndim, 0)
    cdef Py_ssize_t i, j
    for i in normalized_axis:
        out_shape[i] = 1
    j = 0
    for i in range(out_ndim):
        if out_shape[i] == 1:
            continue
        out_shape[i] = a_shape[j]
        j += 1
    return _reshape(a, out_shape)


cpdef _ndarray_base moveaxis(_ndarray_base a, source, destination):
    cdef shape_t src, dest
    cdef Py_ssize_t ndim = a.ndim
    _normalize_axis_tuple(source, ndim, src)
    _normalize_axis_tuple(destination, ndim, dest)

    if src.size() != dest.size():
        raise ValueError('`source` and `destination` arguments must have '
                         'the same number of elements')

    cdef vector.vector[Py_ssize_t] order
    cdef Py_ssize_t i
    for i in range(ndim):
        if not _has_element(src, i):
            order.push_back(i)

    cdef Py_ssize_t d, s
    for d, s in sorted(zip(dest, src)):
        order.insert(order.begin() + d, s)

    return _transpose(a, order)


cpdef _ndarray_base _move_single_axis(
        _ndarray_base a, Py_ssize_t source, Py_ssize_t destination):
    """Like moveaxis, but supporting only integer source and destination."""
    cdef Py_ssize_t ndim = a.ndim
    source = internal._normalize_axis_index(source, ndim)
    destination = internal._normalize_axis_index(destination, ndim)

    if source == destination:
        return a

    cdef vector.vector[Py_ssize_t] order
    cdef Py_ssize_t i
    for i in range(ndim):
        if i != source:
            order.push_back(i)

    order.insert(order.begin() + destination, source)
    return _transpose(a, order)


cpdef _ndarray_base rollaxis(
        _ndarray_base a, Py_ssize_t axis, Py_ssize_t start=0):
    cdef Py_ssize_t i, ndim = a.ndim
    cdef vector.vector[Py_ssize_t] axes
    if axis < 0:
        axis += ndim
    if start < 0:
        start += ndim
    if not (0 <= axis < ndim and 0 <= start <= ndim):
        raise ValueError('Axis out of range')
    if axis < start:
        start -= 1
    if axis == start:
        return a
    if ndim == 2:
        return _transpose(a, axes)

    for i in range(ndim):
        axes.push_back(i)
    axes.erase(axes.begin() + axis)
    axes.insert(axes.begin() + start, axis)
    return _transpose(a, axes)


cpdef _ndarray_base _reshape(_ndarray_base self, const shape_t &shape_spec):
    cdef shape_t shape
    cdef strides_t strides
    cdef _ndarray_base newarray
    shape = internal.infer_unknown_dimension(shape_spec, self.size)
    if internal.vector_equal(shape, self._shape):
        return self.view()

    _get_strides_for_nocopy_reshape(self, shape, strides)
    if strides.size() == shape.size():
        return self._view(type(self), shape, strides, False, True, self)
    newarray = self.copy()
    _get_strides_for_nocopy_reshape(newarray, shape, strides)

    # TODO(niboshi): Confirm update_x_contiguity flags
    newarray._set_shape_and_strides(shape, strides, False, True)
    return newarray


cpdef _ndarray_base _T(_ndarray_base self):
    ret = self.view()
    ret._shape.assign(self._shape.rbegin(), self._shape.rend())
    ret._strides.assign(self._strides.rbegin(), self._strides.rend())
    ret._c_contiguous = self._f_contiguous
    ret._f_contiguous = self._c_contiguous
    return ret


cpdef _ndarray_base _transpose(
        _ndarray_base self, const vector.vector[Py_ssize_t] &axes):
    cdef vector.vector[Py_ssize_t] a_axes
    cdef vector.vector[char] axis_flags
    cdef Py_ssize_t i, ndim, axis, axes_size
    cdef bint is_normal = True, is_trans = True

    axes_size = axes.size()
    if axes_size == 0:
        return _T(self)

    ndim = self._shape.size()
    if axes_size != ndim:
        raise ValueError("axes don't match array")

    axis_flags.resize(ndim, 0)
    for i in range(axes_size):
        axis = axes[i]
        if axis < -ndim or axis >= ndim:
            raise AxisError(axis, ndim)
        axis %= ndim
        a_axes.push_back(axis)
        if axis_flags[axis]:
            raise ValueError('repeated axis in transpose')
        axis_flags[axis] = 1
        is_normal &= i == axis
        is_trans &= ndim - 1 - i == axis

    if is_normal:
        return self.view()
    if is_trans:
        return _T(self)

    ret = self.view()
    ret._shape.clear()
    ret._strides.clear()
    for axis in a_axes:
        ret._shape.push_back(self._shape[axis])
        ret._strides.push_back(self._strides[axis])
    ret._update_contiguity()
    return ret


cpdef array_split(_ndarray_base ary, indices_or_sections, Py_ssize_t axis):
    cdef Py_ssize_t i, ndim, size, each_size, index, prev, stride
    cdef Py_ssize_t num_large
    cdef shape_t shape

    ndim = ary.ndim
    if -ndim > axis or ndim <= axis:
        raise IndexError('Axis exceeds ndim')
    if axis < 0:
        axis += ndim
    size = ary._shape[axis]

    if numpy.isscalar(indices_or_sections):
        each_size = (size - 1) // indices_or_sections
        num_large = (size - 1) % indices_or_sections + 1
        indices = [i * each_size + min(i, num_large)
                   for i in range(1, indices_or_sections)]
    else:
        indices = [i if i >= 0 else size + i for i in indices_or_sections]

    if len(indices) == 0:
        return [ary]

    # Make a copy of shape for each view
    shape = ary._shape

    prev = 0
    ret = []
    stride = ary._strides[axis]
    if ary.size == 0:
        stride = 0
    for index in indices:
        index = min(index, size)
        shape[axis] = max(index - prev, 0)
        v = ary.view()
        v.data = ary.data + prev * stride
        # TODO(niboshi): Confirm update_x_contiguity flags
        v._set_shape_and_strides(shape, ary._strides, True, True)
        ret.append(v)

        prev = index

    shape[axis] = size - prev
    v = ary.view()
    v.data = ary.data + prev * stride
    # TODO(niboshi): Confirm update_x_contiguity flags
    v._set_shape_and_strides(shape, ary._strides, True, True)
    ret.append(v)

    return ret


cpdef _ndarray_base broadcast_to(_ndarray_base array, shape):
    """Broadcast an array to a given shape.

    .. seealso::
        :func:`cupy.broadcast_to` for full documentation,
        :meth:`numpy.broadcast_to`

    """
    shape = tuple(shape) if numpy.iterable(shape) else (shape,)
    cdef int i, j, ndim = array._shape.size(), length = len(shape)
    cdef Py_ssize_t sh, a_sh
    if ndim > length:
        raise ValueError(
            'input operand has more dimensions than allowed by the axis '
            'remapping')
    cdef shape_t _shape = shape
    cdef strides_t strides
    strides.assign(length, 0)
    for i in range(ndim):
        j = i + length - ndim
        sh = _shape[j]
        a_sh = array._shape[i]
        if sh == a_sh:
            strides[j] = array._strides[i]
        elif a_sh != 1:
            raise ValueError(
                'operands could not be broadcast together with shape {} and '
                'requested shape {}'.format(array.shape, shape))

    view = array.view()
    # TODO(niboshi): Confirm update_x_contiguity flags
    view._set_shape_and_strides(_shape, strides, True, True)
    return view


cpdef _ndarray_base _repeat(_ndarray_base a, repeats, axis=None):
    """Repeat arrays along an axis.

    Args:
        a (cupy.ndarray): Array to transform.
        repeats (int, list or tuple): The number of repeats.
        axis (int): The axis to repeat.

    Returns:
        cupy.ndarray: Transformed array with repeats.

    .. seealso:: :func:`numpy.repeat`

    """
    cdef _ndarray_base ret

    if isinstance(repeats, _ndarray_base):
        raise ValueError(
            'cupy.ndaray cannot be specified as `repeats` argument.')

    # Scalar and size 1 'repeat' arrays broadcast to any shape, for all
    # other inputs the dimension must match exactly.
    cdef bint broadcast = False
    # numpy.issubdtype(1, numpy.integer) fails with old numpy like 1.13.3.
    if (isinstance(repeats, int) or
            (hasattr(repeats, 'dtype') and
             numpy.issubdtype(repeats, numpy.integer))):
        if repeats < 0:
            raise ValueError(
                '\'repeats\' should not be negative: {}'.format(repeats))
        broadcast = True
        repeats = [repeats]
    elif cpython.PySequence_Check(repeats):
        for rep in repeats:
            if rep < 0:
                raise ValueError(
                    'all elements of \'repeats\' should not be negative: {}'
                    .format(repeats))
        if len(repeats) == 1:
            broadcast = True
    else:
        raise ValueError(
            '\'repeats\' should be int or sequence: {}'.format(repeats))

    if axis is None:
        if broadcast:
            a = _reshape(a, (-1, 1))
            ret = core.ndarray((a.size, repeats[0]), dtype=a.dtype)
            if ret.size:
                elementwise_copy(a, ret)
            return ret.ravel()
        else:
            a = a.ravel()
            axis = 0
    else:
        axis = internal._normalize_axis_index(axis, a.ndim)

    if broadcast:
        repeats = repeats * a._shape[axis]
    elif a.shape[axis] != len(repeats):
        raise ValueError(
            '\'repeats\' and \'axis\' of \'a\' should be same length: {} != {}'
            .format(a.shape[axis], len(repeats)))

    ret_shape = list(a.shape)
    ret_shape[axis] = sum(repeats)
    ret = core.ndarray(ret_shape, dtype=a.dtype)
    a_index = [slice(None)] * len(ret_shape)
    ret_index = list(a_index)
    offset = 0
    for i in range(a._shape[axis]):
        if repeats[i] == 0:
            continue
        a_index[axis] = slice(i, i + 1)
        ret_index[axis] = slice(offset, offset + repeats[i])
        # convert to tuple because cupy has a indexing bug
        ret[tuple(ret_index)] = a[tuple(a_index)]
        offset += repeats[i]
    return ret


cpdef _ndarray_base concatenate_method(
        tup, int axis, _ndarray_base out=None, dtype=None,
        casting='same_kind'):
    cdef int ndim0
    cdef int i
    cdef _ndarray_base a, a0

    if dtype is not None:
        dtype = get_dtype(dtype)

    dev_id = device.get_device_id()
    arrays = _preprocess_args(dev_id, tup, False)

    # Check if the input is not an empty sequence
    if len(arrays) == 0:
        raise ValueError('Cannot concatenate from empty tuple')

    # Check types of the input arrays
    for o in arrays:
        if not isinstance(o, _ndarray_base):
            raise TypeError('Only cupy arrays can be concatenated')

    # Check ndim > 0 for the input arrays
    for o in arrays:
        a = o
        if a._shape.size() == 0:
            raise TypeError('zero-dimensional arrays cannot be concatenated')

    # Check ndim consistency of the input arrays
    a0 = arrays[0]
    ndim0 = a0._shape.size()
    for o in arrays[1:]:
        a = o
        if a._shape.size() != ndim0:
            raise ValueError(
                'All arrays to concatenate must have the same ndim')

    # Check shape consistency of the input arrays, and compute the output shape
    shape0 = a0._shape
    axis = internal._normalize_axis_index(axis, ndim0)
    for o in arrays[1:]:
        a = o
        for i in range(ndim0):
            if i != axis and shape0[i] != a._shape[i]:
                raise ValueError(
                    'All arrays must have same shape except the axis to '
                    'concatenate')
        shape0[axis] += a._shape[axis]

    # Compute the output dtype
    if out is None:
        if dtype is None:
            dtype = a0.dtype
            have_same_types = True
            for o in arrays[1:]:
                have_same_types = have_same_types and (o.dtype == dtype)
            if not have_same_types:
                dtype = functools.reduce(
                    numpy.promote_types, set([a.dtype for a in arrays]))
    else:
        if dtype is not None:
            raise TypeError('concatenate() only takes `out` or `dtype` as an '
                            'argument, but both were provided.')
        dtype = out.dtype

    # Check casting rule
    for o in arrays:
        _raise_if_invalid_cast(o.dtype, dtype, casting)

    # Prpare the output array
    shape_t = tuple(shape0)
    if out is None:
        out = core.ndarray(shape_t, dtype=dtype)
    else:
        if len(out.shape) != len(shape_t):
            raise ValueError('Output array has wrong dimensionality')
        if out.shape != shape_t:
            raise ValueError('Output array is the wrong shape')

    return _concatenate(arrays, axis, shape_t, out, casting)


cpdef _ndarray_base _concatenate(
        list arrays, Py_ssize_t axis, tuple shape, _ndarray_base out,
        str casting):
    cdef _ndarray_base a, b
    cdef Py_ssize_t i, aw, itemsize, axis_size
    cdef bint all_same_type, same_shape_and_contiguous
    # If arrays are large, Issuing each copy method is efficient.
    cdef Py_ssize_t threshold_size = 2 * 1024 * 1024

    dtype = out.dtype

    if len(arrays) > 8:
        all_same_type = True
        same_shape_and_contiguous = True
        axis_size = shape[axis] // len(arrays)
        total_bytes = 0
        itemsize = dtype.itemsize
        for a in arrays:
            if a.dtype != dtype:
                all_same_type = False
                break
            if same_shape_and_contiguous:
                same_shape_and_contiguous = (
                    a._c_contiguous and a._shape[axis] == axis_size)
            total_bytes += a.size * itemsize

        if all_same_type and total_bytes < threshold_size * len(arrays):
            return _concatenate_single_kernel(
                arrays, axis, shape, dtype, same_shape_and_contiguous, out)

    i = 0
    slice_list = [slice(None)] * len(shape)
    for a in arrays:
        aw = a._shape[axis]
        slice_list[axis] = slice(i, i + aw)
        b = out[tuple(slice_list)]
        elementwise_copy(a, b, casting=casting)
        i += aw
    return out


cpdef Py_ssize_t size(_ndarray_base a, axis=None) except? -1:
    """Returns the number of elements along a given axis.

    Args:
        a (ndarray): Input data.
        axis (int or None): Axis along which the elements are counted.
            When it is ``None``, it returns the total number of elements.

    Returns:
        int: Number of elements along the given axis.

    """
    cdef int index, ndim
    if axis is None:
        return a.size
    else:
        index = axis
        ndim = a._shape.size()
        if index < 0:
            index += ndim
        if not 0 <= index < ndim:
            raise IndexError('index out of range')
        return a._shape[index]


# private


cdef bint _has_element(const shape_t &source, Py_ssize_t n):
    for i in range(source.size()):
        if source[i] == n:
            return True
    return False


cdef _get_strides_for_nocopy_reshape(
        _ndarray_base a, const shape_t &newshape, strides_t &newstrides):
    cdef Py_ssize_t size, itemsize, ndim, dim, last_stride
    size = a.size
    newstrides.clear()

    itemsize = a.itemsize
    if size == 1:
        newstrides.assign(<Py_ssize_t>newshape.size(), itemsize)
        return
    if size == 0:
        internal.get_contiguous_strides_inplace(
            newshape, newstrides, itemsize, True, False)
        return

    cdef shape_t shape
    cdef strides_t strides
    internal.get_reduced_dims(a._shape, a._strides, itemsize, shape, strides)

    ndim = shape.size()
    dim = 0
    last_stride = shape[0] * strides[0]
    for i in range(newshape.size()):
        size = newshape[i]
        if size <= 1:
            newstrides.push_back(last_stride)
            continue
        if dim >= ndim or shape[dim] % size != 0:
            newstrides.clear()
            break
        shape[dim] //= size
        last_stride = shape[dim] * strides[dim]
        newstrides.push_back(last_stride)
        if shape[dim] == 1:
            dim += 1


cdef _normalize_axis_tuple(axis, Py_ssize_t ndim, shape_t &ret):
    """Normalizes an axis argument into a tuple of non-negative integer axes.

    Arguments `argname` and `allow_duplicate` are not supported.

    """
    if numpy.isscalar(axis):
        axis = (axis,)

    for ax in axis:
        ax = internal._normalize_axis_index(ax, ndim)
        if _has_element(ret, ax):
            # the message in `numpy.core.numeric.normalize_axis_tuple`
            raise ValueError('repeated axis')
        ret.push_back(ax)


cdef _ndarray_base _concatenate_single_kernel(
        list arrays, Py_ssize_t axis, tuple shape, dtype,
        bint same_shape_and_contiguous, _ndarray_base out):
    cdef _ndarray_base a, x
    cdef Py_ssize_t base, cum, ndim
    cdef int i, j
    cdef Py_ssize_t[:] ptrs
    cdef Py_ssize_t[:] cum_sizes
    cdef Py_ssize_t[:, :] x_strides
    cdef int device_id = device.get_device_id()

    assert out is not None

    ptrs = numpy.ndarray(len(arrays), numpy.int64)
    for i, a in enumerate(arrays):
        _check_peer_access(a, device_id)
        ptrs[i] = a.data.ptr
    x = core.array(ptrs)

    if same_shape_and_contiguous:
        base = internal.prod_sequence(shape[axis:]) // len(arrays)
        _concatenate_kernel_same_size(x, base, out)
        return out

    ndim = len(shape)
    x_strides = numpy.ndarray((len(arrays), ndim), numpy.int64)
    cum_sizes = numpy.ndarray(len(arrays), numpy.int64)
    cum = 0
    for i, a in enumerate(arrays):
        for j in range(ndim):
            x_strides[i, j] = <int>a._strides[j]
        cum_sizes[i] = cum
        cum += <int>a._shape[axis]

    _concatenate_kernel(
        x, axis, core.array(cum_sizes), core.array(x_strides), out)
    return out


cdef _concatenate_kernel_same_size = ElementwiseKernel(
    'raw P x, int64 base',
    'T y',
    '''
    ptrdiff_t middle = i / base;
    ptrdiff_t top = middle / x.size();
    ptrdiff_t array_ind = middle - top * x.size();
    ptrdiff_t offset = i + (top - middle) * base;
    y = reinterpret_cast<T*>(x[array_ind])[offset];
    ''',
    'cupy_concatenate_same_size'
)


cdef _concatenate_kernel = ElementwiseKernel(
    '''raw P x, int32 axis, raw int64 cum_sizes, raw int64 x_strides''',
    'T y',
    '''
    ptrdiff_t axis_ind = _ind.get()[axis];
    ptrdiff_t left = 0;
    ptrdiff_t right = cum_sizes.size();

    while (left < right - 1) {
      ptrdiff_t m = (left + right) / 2;
      if (axis_ind < cum_sizes[m]) {
        right = m;
      } else {
        left = m;
      }
    }

    ptrdiff_t array_ind = left;
    axis_ind -= cum_sizes[left];
    char* ptr = reinterpret_cast<char*>(x[array_ind]);
    for (int j = _ind.ndim - 1; j >= 0; --j) {
      ptrdiff_t ind[] = {array_ind, j};
      ptrdiff_t offset;
      if (j == axis) {
        offset = axis_ind;
      } else {
        offset = _ind.get()[j];
      }
      ptr += x_strides[ind] * offset;
    }

    y = *reinterpret_cast<T*>(ptr);
    ''',
    'cupy_concatenate',
    reduce_dims=False
)

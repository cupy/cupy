# distutils: language = c++
import sys

import numpy

from cupy.core import _errors
from cupy.core._kernel import ElementwiseKernel
from cupy.core._ufuncs import elementwise_copy

cimport cython  # NOQA
cimport cpython  # NOQA
from libcpp cimport vector

from cupy.core cimport core
from cupy.core.core cimport ndarray
from cupy.core cimport internal


cdef Py_ssize_t PY_SSIZE_T_MAX = sys.maxsize


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
        cdef Py_ssize_t i, j, s, smin, smax, a_ndim, a_sh
        cdef vector.vector[Py_ssize_t] shape, strides, r_shape, r_strides
        cdef vector.vector[vector.vector[Py_ssize_t]] shape_arr
        cdef ndarray a, view
        cdef slice rev = slice(None, None, -1)

        self.nd = 0
        for x in arrays:
            if not isinstance(x, ndarray):
                continue
            a = x
            self.nd = max(self.nd, <Py_ssize_t>a._shape.size())
            r_shape.assign(a._shape.rbegin(), a._shape.rend())
            shape_arr.push_back(r_shape)

        r_shape.clear()
        for i in range(self.nd):
            smin = PY_SSIZE_T_MAX
            smax = 0
            for j in range(<Py_ssize_t>shape_arr.size()):
                if i < <Py_ssize_t>shape_arr[j].size():
                    s = shape_arr[j][i]
                    smin = min(smin, s)
                    smax = max(smax, s)
            if smin == 0 and smax > 1:
                raise ValueError(
                    'shape mismatch: objects cannot be broadcast to a '
                    'single shape')
            r_shape.push_back(0 if smin == 0 else smax)

        shape.assign(r_shape.rbegin(), r_shape.rend())
        self.shape = tuple(shape)
        self.size = internal.prod(shape)

        broadcasted = []
        for x in arrays:
            if not isinstance(x, ndarray):
                broadcasted.append(x)
                continue
            a = x
            if internal.vector_equal(a._shape, shape):
                broadcasted.append(a)
                continue

            r_strides.assign(self.nd, <Py_ssize_t>0)
            a_ndim = a._shape.size()
            for i in range(a_ndim):
                a_sh = a._shape[a_ndim - i - 1]
                if a_sh == r_shape[i]:
                    r_strides[i] = a._strides[a_ndim - i - 1]
                elif a_sh != 1:
                    raise ValueError(
                        'operands could not be broadcast together with shapes '
                        '{}'.format(
                            ', '.join([str(x.shape) if isinstance(x, ndarray)
                                       else '()' for x in arrays])))

            strides.assign(r_strides.rbegin(), r_strides.rend())
            view = a.view()
            # TODO(niboshi): Confirm update_x_contiguity flags
            view._set_shape_and_strides(shape, strides, True, True)
            broadcasted.append(view)

        self.values = tuple(broadcasted)


# ndarray members


cdef _ndarray_shape_setter(ndarray self, newshape):
    cdef vector.vector[Py_ssize_t] shape, strides
    if not cpython.PySequence_Check(newshape):
        newshape = (newshape,)
    shape = internal.infer_unknown_dimension(newshape, self.size)
    strides = _get_strides_for_nocopy_reshape(self, shape)
    if strides.size() != shape.size():
        raise AttributeError('incompatible shape')
    self._shape = shape
    self._strides = strides
    self._update_f_contiguity()


cdef ndarray _ndarray_reshape(ndarray self, tuple shape, order):
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
        return _reshape(self.transpose(), shape[::-1]).transpose()


cdef ndarray _ndarray_transpose(ndarray self, tuple axes):
    cdef ndarray ret
    cdef vector.vector[Py_ssize_t] vec_axes, a_axes, temp_axes
    cdef Py_ssize_t ndim, axis
    if len(axes) == 1:
        a = axes[0]
        if a is None:
            axes = ()
        elif cpython.PySequence_Check(a):
            axes = tuple(a)
    return _transpose(self, axes)


cdef ndarray _ndarray_swapaxes(
        ndarray self, Py_ssize_t axis1, Py_ssize_t axis2):
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


cdef ndarray _ndarray_flatten(ndarray self):
    newarray = self.copy(order='C')
    newarray._shape.assign(<Py_ssize_t>1, self.size)
    newarray._strides.assign(<Py_ssize_t>1,
                             <Py_ssize_t>self.itemsize)
    newarray._c_contiguous = True
    newarray._f_contiguous = True
    return newarray


cdef ndarray _ndarray_ravel(ndarray self, order):
    # TODO(beam2d, grlee77): Support K ordering option
    cdef int order_char
    cdef vector.vector[Py_ssize_t] shape
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
        return _reshape(self.transpose(), shape)
    elif order_char == b'K':
        raise NotImplementedError(
            "ravel with order='K' not yet implemented.")


cdef ndarray _ndarray_squeeze(ndarray self, axis):
    cdef vector.vector[char] axis_flags
    cdef vector.vector[Py_ssize_t] newshape, newstrides
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
            _axis = <Py_ssize_t>axis[i]
            axis_orig = _axis
            if _axis < 0:
                _axis += ndim
            if _axis < 0 or _axis >= ndim:
                raise _errors._AxisError(
                    "'axis' entry %d is out of bounds [-%d, %d)" %
                    (axis_orig, ndim, ndim))
            if axis_flags[_axis] == 1:
                raise ValueError("duplicate value in 'axis'")
            axis_flags[_axis] = 1
    else:
        _axis = <Py_ssize_t>axis
        axis_orig = _axis
        if _axis < 0:
            _axis += ndim
        if ndim == 0 and (_axis == 0 or _axis == -1):
            # Special case letting axis={-1,0} slip through for scalars,
            # for backwards compatibility reasons.
            pass
        else:
            if _axis < 0 or _axis >= ndim:
                raise _errors._AxisError(
                    "'axis' entry %d is out of bounds [-%d, %d)" %
                    (axis_orig, ndim, ndim))
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


cdef ndarray _ndarray_repeat(ndarray self, repeats, axis):
    return _repeat(self, repeats, axis)


# exposed


cpdef ndarray moveaxis(ndarray a, source, destination):
    cdef vector.vector[Py_ssize_t] src = _normalize_axis_tuple(source, a.ndim)
    cdef vector.vector[Py_ssize_t] dest = (
        _normalize_axis_tuple(destination, a.ndim))

    if len(src) != len(dest):
        raise ValueError('`source` and `destination` arguments must have '
                         'the same number of elements')

    cdef vector.vector[Py_ssize_t] order
    cdef Py_ssize_t n = 0
    for i in range(a.ndim):
        n = <Py_ssize_t>i
        if not _has_element(src, n):
            order.push_back(n)

    cdef Py_ssize_t d, s
    for d, s in sorted(zip(dest, src)):
        order.insert(order.begin() + d, s)

    return a.transpose(order)


cpdef ndarray rollaxis(ndarray a, Py_ssize_t axis, Py_ssize_t start=0):
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


cpdef ndarray _reshape(ndarray self, vector.vector[Py_ssize_t] shape):
    cdef vector.vector[Py_ssize_t] strides
    cdef ndarray newarray
    shape = internal.infer_unknown_dimension(shape, self.size)
    if internal.vector_equal(shape, self._shape):
        return self.view()

    strides = _get_strides_for_nocopy_reshape(self, shape)
    if strides.size() == shape.size():
        newarray = self.view()
    else:
        newarray = self.copy()
        strides = _get_strides_for_nocopy_reshape(newarray, shape)

    if shape.size() != strides.size():
        raise ValueError('total size of new array must be unchanged')
    # TODO(niboshi): Confirm update_x_contiguity flags
    newarray._set_shape_and_strides(shape, strides, False, True)
    return newarray


cpdef ndarray _transpose(ndarray self, vector.vector[Py_ssize_t] axes):
    cdef ndarray ret
    cdef vector.vector[Py_ssize_t] a_axes, rev_axes
    cdef Py_ssize_t ndim, axis

    ndim = self._shape.size()
    ret = self.view()
    if axes.size() == 0:
        ret._shape.assign(self._shape.rbegin(), self._shape.rend())
        ret._strides.assign(self._strides.rbegin(), self._strides.rend())
        ret._c_contiguous = self._f_contiguous
        ret._f_contiguous = self._c_contiguous
        return ret

    if <Py_ssize_t>axes.size() != ndim:
        raise ValueError('Invalid axes value: %s' % str(axes))

    for i in range(ndim):
        a_axes.push_back(i)
        axis = axes[i]
        if axis < -ndim or axis >= ndim:
            raise IndexError('Axes overrun')
        axes[i] = axis % ndim

    if internal.vector_equal(a_axes, axes):
        return ret
    rev_axes.assign(axes.rbegin(), axes.rend())
    if internal.vector_equal(a_axes, rev_axes):
        ret._shape.assign(self._shape.rbegin(), self._shape.rend())
        ret._strides.assign(self._strides.rbegin(), self._strides.rend())
        ret._c_contiguous = self._f_contiguous
        ret._f_contiguous = self._c_contiguous
        return ret

    if ndim != len({i for i in axes}):
        raise ValueError('Invalid axes value: %s' % str(axes))

    ret._shape.clear()
    ret._strides.clear()
    for axis in axes:
        ret._shape.push_back(self._shape[axis])
        ret._strides.push_back(self._strides[axis])
    ret._update_contiguity()
    return ret


cpdef array_split(ndarray ary, indices_or_sections, Py_ssize_t axis):

    cdef Py_ssize_t i, ndim, size, each_size, index, prev, offset, stride
    cdef vector.vector[Py_ssize_t] shape

    ndim = ary.ndim
    if -ndim > axis or ndim <= axis:
        raise IndexError('Axis exceeds ndim')
    if axis < 0:
        axis += ndim
    size = ary._shape[axis]

    if numpy.isscalar(indices_or_sections):
        each_size = (size - 1) // indices_or_sections + 1
        indices = [i * each_size
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
        shape[axis] = index - prev
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


cpdef ndarray broadcast_to(ndarray array, shape):
    """Broadcast an array to a given shape.

    .. seealso::
        :func:`cupy.broadcast_to` for full documentation,
        :meth:`numpy.broadcast_to`

    """
    cdef int i, j, ndim = array._shape.size(), length = len(shape)
    cdef Py_ssize_t sh, a_sh
    if ndim > length:
        raise ValueError(
            'input operand has more dimensions than allowed by the axis '
            'remapping')
    cdef vector.vector[Py_ssize_t] strides, _shape = shape
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


cpdef ndarray _repeat(ndarray a, repeats, axis=None):
    """Repeat arrays along an axis.

    Args:
        a (cupy.ndarray): Array to transform.
        repeats (int, list or tuple): The number of repeats.
        axis (int): The axis to repeat.

    Returns:
        cupy.ndarray: Transformed array with repeats.

    .. seealso:: :func:`numpy.repeat`

    """
    cdef ndarray ret

    # Scalar and size 1 'repeat' arrays broadcast to any shape, for all
    # other inputs the dimension must match exactly.
    cdef bint broadcast = False
    # numpy.issubdtype(1, numpy.integer) fails with old numpy like 1.13.3.
    if (isinstance(repeats, int) or
            (hasattr(repeats, 'dtype') and
             numpy.issubdtype(repeats, numpy.integer))):
        if repeats < 0:
            raise ValueError(
                "'repeats' should not be negative: {}".format(repeats))
        broadcast = True
        repeats = [repeats]
    elif cpython.PySequence_Check(repeats):
        for rep in repeats:
            if rep < 0:
                raise ValueError(
                    "all elements of 'repeats' should not be negative: {}"
                    .format(repeats))
        if len(repeats) == 1:
            broadcast = True
    else:
        raise ValueError(
            "'repeats' should be int or sequence: {}".format(repeats))

    if axis is None:
        if broadcast:
            a = _reshape(a, (-1, 1))
            ret = ndarray((a.size, repeats[0]), dtype=a.dtype)
            if ret.size:
                ret[...] = a
            return ret.ravel()
        else:
            a = a.ravel()
            axis = 0
    elif not (-a.ndim <= axis < a.ndim):
        raise _errors._AxisError(
            'axis {} is out of bounds for array of dimension {}'.format(
                axis, a.ndim))

    if broadcast:
        repeats = repeats * a._shape[axis % a._shape.size()]
    elif a.shape[axis] != len(repeats):
        raise ValueError(
            "'repeats' and 'axis' of 'a' should be same length: {} != {}"
            .format(a.shape[axis], len(repeats)))

    if axis < 0:
        axis += a.ndim

    ret_shape = list(a.shape)
    ret_shape[axis] = sum(repeats)
    ret = ndarray(ret_shape, dtype=a.dtype)
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


cpdef ndarray concatenate_method(tup, int axis):
    cdef int ndim, a_ndim
    cdef int i
    cdef ndarray a
    cdef bint have_same_types
    cdef vector.vector[Py_ssize_t] shape

    ndim = -1
    dtype = None
    have_same_types = True
    arrays = list(tup)
    for o in arrays:
        if not isinstance(o, ndarray):
            raise TypeError('Only cupy arrays can be concatenated')
        a = o
        a_ndim = a._shape.size()
        if a_ndim == 0:
            raise TypeError('zero-dimensional arrays cannot be concatenated')
        if ndim == -1:
            ndim = a_ndim
            shape = a._shape
            if axis < 0:
                axis += ndim
            if axis < 0 or axis >= ndim:
                raise _errors._AxisError(
                    'axis {} out of bounds [0, {})'.format(axis, ndim))
            dtype = a.dtype
            continue

        have_same_types = have_same_types and (a.dtype == dtype)
        if a_ndim != ndim:
            raise ValueError(
                'All arrays to concatenate must have the same ndim')
        for i in range(ndim):
            if i != axis and shape[i] != a._shape[i]:
                raise ValueError(
                    'All arrays must have same shape except the axis to '
                    'concatenate')
        shape[axis] += a._shape[axis]

    if ndim == -1:
        raise ValueError('Cannot concatenate from empty tuple')

    if not have_same_types:
        dtype = numpy.find_common_type([a.dtype for a in arrays], [])
    return _concatenate(arrays, axis, tuple(shape), dtype)


cpdef ndarray _concatenate(list arrays, Py_ssize_t axis, tuple shape, dtype):
    cdef ndarray a, ret
    cdef Py_ssize_t i, aw, itemsize, axis_size
    cdef bint all_same_type, same_shape_and_contiguous
    # If arrays are large, Issuing each copy method is efficient.
    cdef Py_ssize_t threshold_size = 2 * 1024 * 1024

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
                arrays, axis, shape, dtype, same_shape_and_contiguous)

    ret = ndarray(shape, dtype=dtype)
    i = 0
    slice_list = [slice(None)] * len(shape)
    for a in arrays:
        aw = a._shape[axis]
        slice_list[axis] = slice(i, i + aw)
        elementwise_copy(a, core._simple_getitem(ret, slice_list))
        i += aw
    return ret


cpdef Py_ssize_t size(ndarray a, axis=None) except? -1:
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


cdef bint _has_element(vector.vector[Py_ssize_t] source, Py_ssize_t n):
    for elem in source:
        if elem == n:
            return True
    return False


cdef vector.vector[Py_ssize_t] _get_strides_for_nocopy_reshape(
        ndarray a, vector.vector[Py_ssize_t] & newshape) except *:
    cdef vector.vector[Py_ssize_t] newstrides
    cdef Py_ssize_t size, itemsize, ndim, dim, last_stride
    size = a.size
    if size != internal.prod(newshape):
        return newstrides

    itemsize = a.itemsize
    if size == 1:
        newstrides.assign(<Py_ssize_t>newshape.size(), itemsize)
        return newstrides

    cdef vector.vector[Py_ssize_t] shape, strides
    internal.get_reduced_dims(a._shape, a._strides, itemsize, shape, strides)

    ndim = shape.size()
    dim = 0
    sh = shape[0]
    st = strides[0]
    last_stride = shape[0] * strides[0]
    for size in newshape:
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
    return newstrides


cdef vector.vector[Py_ssize_t] _normalize_axis_tuple(
        axis, Py_ssize_t ndim) except *:
    """Normalizes an axis argument into a tuple of non-negative integer axes.

    Arguments `allow_duplicate` and `axis_name` are not supported.

    """
    if numpy.isscalar(axis):
        axis = (axis,)

    cdef vector.vector[Py_ssize_t] ret
    for ax in axis:
        if ax >= ndim or ax < -ndim:
            raise _errors._AxisError(
                'axis {} is out of bounds for array of '
                'dimension {}'.format(ax, ndim))
        if _has_element(ret, ax):
            raise _errors._AxisError('repeated axis')
        ret.push_back(ax % ndim)

    return ret


cdef ndarray _concatenate_single_kernel(
        list arrays, Py_ssize_t axis, tuple shape, dtype,
        bint same_shape_and_contiguous):
    cdef ndarray a, x, ret
    cdef Py_ssize_t base, cum, ndim
    cdef int i, j
    cdef Py_ssize_t[:] ptrs
    cdef Py_ssize_t[:] cum_sizes
    cdef Py_ssize_t[:, :] x_strides

    ptrs = numpy.ndarray(len(arrays), numpy.int64)
    for i, a in enumerate(arrays):
        ptrs[i] = a.data.ptr
    x = core.array(ptrs)

    ret = core.ndarray(shape, dtype=dtype)
    if same_shape_and_contiguous:
        base = internal.prod(shape[axis:]) // len(arrays)
        _concatenate_kernel_same_size(x, base, ret)
        return ret

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
        x, axis, core.array(cum_sizes), core.array(x_strides), ret)
    return ret


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

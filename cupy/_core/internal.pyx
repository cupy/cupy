# distutils: language = c++

cimport cpython  # NOQA
cimport cython  # NOQA
from libcpp cimport bool as cpp_bool
from libc.stdint cimport uint32_t

import warnings

import numpy
from cupy.exceptions import AxisError

from cupy._core.core cimport _ndarray_base


cdef extern from 'halffloat.h':
    uint16_t npy_floatbits_to_halfbits(uint32_t f)
    uint32_t npy_halfbits_to_floatbits(uint16_t h)


@cython.profile(False)
cpdef inline Py_ssize_t prod(const vector.vector[Py_ssize_t]& args):
    cdef Py_ssize_t n = 1
    for i in range(args.size()):
        n *= args[i]
    return n


@cython.profile(False)
cpdef inline Py_ssize_t prod_sequence(object args):
    cdef Py_ssize_t i, n = 1
    for i in args:
        n *= i
    return n


@cython.profile(False)
cpdef inline bint is_in(const vector.vector[Py_ssize_t]& args, Py_ssize_t x):
    cdef Py_ssize_t i
    for i in range(<Py_ssize_t>args.size()):
        if args[i] == x:
            return True
    return False


@cython.profile(False)
cpdef inline void _check_not_bool(object x) except *:
    if isinstance(x, (bool, numpy.bool_)):
        raise TypeError('an integer is required')
    if (
        isinstance(x, numpy.ndarray)
        and x.shape == ()
        and x.dtype == numpy.bool_
    ):
        raise TypeError('an integer is required')


@cython.profile(False)
cpdef inline tuple get_size(object size):
    if size is None:
        warnings.warn(
            'Passing None into shape arguments as an alias for () is '
            'deprecated.',
            DeprecationWarning,
        )
        return ()
    if cpython.PySequence_Check(size):
        # A numpy.ndarray unconditionally succeeds in PySequence_Check as
        # it implements __getitem__, but zero-dim one is an unsized object
        # and fails to make a tuple from it.
        try:
            ret = tuple(size)
        except TypeError:
            _check_not_bool(size)
            return size,
        for x in ret:
            _check_not_bool(x)
        return ret
    _check_not_bool(size)
    return size,


@cython.profile(False)
cpdef inline bint vector_equal(
        const vector.vector[Py_ssize_t]& x,
        const vector.vector[Py_ssize_t]& y):
    cdef Py_ssize_t n = x.size()
    if n != <Py_ssize_t>y.size():
        return False
    for i in range(n):
        if x[i] != y[i]:
            return False
    return True


@cython.profile(False)
cdef void get_reduced_dims(
        shape_t& shape, strides_t& strides, Py_ssize_t itemsize,
        shape_t& reduced_shape, strides_t& reduced_strides):
    cdef Py_ssize_t i, ndim, sh, st, prev_st, index
    ndim = shape.size()
    reduced_shape.clear()
    reduced_strides.clear()
    if ndim == 0:
        return
    reduced_shape.reserve(ndim)
    reduced_strides.reserve(ndim)

    prev_st = 0
    index = -1
    for i in range(ndim):
        sh = shape[i]
        if sh == 0:
            reduced_shape.assign(1, 0)
            reduced_strides.assign(1, itemsize)
            return
        if sh == 1:
            continue
        st = strides[i]
        if index == -1 or prev_st != sh * st:
            reduced_shape.push_back(sh)
            reduced_strides.push_back(st)
            index += 1
        else:
            reduced_shape[index] *= sh
            reduced_strides[index] = st
        prev_st = st


@cython.profile(False)
cdef inline Py_ssize_t get_contiguous_strides_inplace(
        const shape_t& shape, strides_t& strides,
        Py_ssize_t itemsize, bint is_c_contiguous, bint zeros_for_zerosize):
    cdef Py_ssize_t st, sh
    cdef Py_ssize_t is_nonzero_size = 1
    cdef int i, ndim = shape.size()
    cdef Py_ssize_t idx
    strides.resize(ndim, 0)
    st = 1

    # For some cases, the strides are all set to zero if the shape is
    # zero-sized since NumPy 1.23.
    if zeros_for_zerosize:
        for i in range(ndim):
            sh = shape[i]
            if sh == 0:
                is_nonzero_size = 0
                break

        if is_nonzero_size == 0:
            for i in range(ndim):
                strides[i] = 0
            return 0

    for i in range(ndim):
        if is_c_contiguous:
            idx = ndim - 1 - i
        else:
            idx = i
        strides[idx] = st * itemsize
        sh = shape[idx]
        if sh > 1:
            st *= sh
        elif sh == 0:
            is_nonzero_size = 0
    return st * is_nonzero_size


@cython.profile(False)
cpdef inline bint get_c_contiguity(
        shape_t& shape, strides_t& strides, Py_ssize_t itemsize):
    cdef Py_ssize_t i, prev_i, ndim, sh, st, index
    ndim = strides.size()
    if ndim == 0 or (ndim == 1 and strides[0] == itemsize):
        return True
    prev_i = -1
    index = st = 0
    for i in range(ndim):
        sh = shape[i]
        if sh == 0:
            return True
        if sh == 1:
            continue
        st = strides[i]
        if prev_i == -1 or strides[prev_i] != sh * st:
            index += 1
        prev_i = i
    return index == 0 or (index == 1 and st == itemsize)


@cython.profile(False)
cpdef shape_t infer_unknown_dimension(
        const shape_t& shape, Py_ssize_t size) except *:
    cdef shape_t ret = shape
    cdef Py_ssize_t cnt=0, index=-1, new_size=1
    for i in range(shape.size()):
        if shape[i] < 0:
            cnt += 1
            index = i
        else:
            new_size *= shape[i]
    if cnt == 0 and new_size == size:
        return ret
    if cnt > 1:
        raise ValueError('can only specify one unknown dimension')
    if cnt == 0 or new_size == 0 or size % new_size != 0:
        # TODO(kataoka): print "newaxis" for unknown
        raise ValueError(
            f'cannot reshape array of size {size} into shape {tuple(shape)}')
    ret[index] = size // new_size
    return ret


@cython.profile(False)
cpdef inline Py_ssize_t _extract_slice_element(x) except? 0:
    try:
        return x.__index__()
    except AttributeError:
        return int(x)


@cython.profile(False)
cpdef slice complete_slice(slice slc, Py_ssize_t dim):
    cdef Py_ssize_t start=0, stop=0, step=0
    cdef bint start_none, stop_none
    if slc.step is None:
        step = 1
    else:
        try:
            step = _extract_slice_element(slc.step)
        except TypeError:
            raise TypeError(
                'slice.step must be int or None or have __index__ method: '
                '{}'.format(slc))
        if step == 0:
            raise ValueError('Slice step must be nonzero.')

    start_none = slc.start is None
    if not start_none:
        try:
            start = _extract_slice_element(slc.start)
        except TypeError:
            raise TypeError(
                'slice.start must be int or None or have __index__ method: '
                '{}'.format(slc))

        if start < 0:
            start += dim

    stop_none = slc.stop is None
    if not stop_none:
        try:
            stop = _extract_slice_element(slc.stop)
        except TypeError:
            raise TypeError(
                'slice.stop must be int or None or have __index__ method: '
                '{}'.format(slc))

        if stop < 0:
            stop += dim

    if step > 0:
        start = 0 if start_none else max(0, min(dim, start))
        stop = dim if stop_none else max(start, min(dim, stop))
    else:
        start = dim - 1 if start_none else max(-1, min(dim - 1, start))
        stop = -1 if stop_none else max(-1, min(start, stop))

    return slice(start, stop, step)


@cython.profile(False)
cpdef tuple complete_slice_list(list slice_list, Py_ssize_t ndim):
    warnings.warn(
        'complete_slice_list is deprecated because the function sometimes '
        'returns wrong result (#1512, #4799).',
        DeprecationWarning)
    cdef Py_ssize_t i, n_newaxes, n_ellipses, ellipsis, n
    slice_list = list(slice_list)  # copy list
    # Expand ellipsis into empty slices
    ellipsis = -1
    n_newaxes = n_ellipses = 0
    for i, s in enumerate(slice_list):
        if s is None:
            n_newaxes += 1
        elif s is Ellipsis:
            n_ellipses += 1
            ellipsis = i
    if n_ellipses > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")

    n = ndim - <Py_ssize_t>len(slice_list) + n_newaxes
    if n_ellipses > 0:
        slice_list[ellipsis:ellipsis + 1] = [slice(None)] * (n + 1)
    elif n > 0:
        slice_list += [slice(None)] * n
    return slice_list, n_newaxes


@cython.profile(False)
cpdef size_t clp2(size_t x):
    x -= 1
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    x |= x >> 32
    return x + 1


cdef union float32_int:
    uint32_t n
    float f


cpdef uint16_t to_float16(float f):
    cdef float32_int c
    c.f = f
    return npy_floatbits_to_halfbits(c.n)


cpdef float from_float16(uint16_t v):
    cdef float32_int c
    c.n = npy_halfbits_to_floatbits(v)
    return c.f


@cython.profile(False)
cdef inline int _normalize_order(order, cpp_bool allow_k=True) except? 0:
    cdef int order_char
    order_char = b'C' if len(order) == 0 else ord(order[0])
    if order_char == b'K' or order_char == b'k':
        if not allow_k:
            raise ValueError('order \'K\' is not permitted')
        order_char = b'K'
    elif order_char == b'A' or order_char == b'a':
        order_char = b'A'
    elif order_char == b'C' or order_char == b'c':
        order_char = b'C'
    elif order_char == b'F' or order_char == b'f':
        order_char = b'F'
    else:
        raise ValueError('order not understood')
    return order_char


cdef _broadcast_core(list arrays, shape_t& shape):
    cdef Py_ssize_t i, j, s, a_ndim, a_sh, nd
    cdef strides_t strides
    cdef vector.vector[int] index
    cdef _ndarray_base a

    shape.clear()
    index.reserve(len(arrays))
    nd = 0
    for i, x in enumerate(arrays):
        if not isinstance(x, _ndarray_base):
            continue
        a = x
        index.push_back(i)
        nd = max(nd, <Py_ssize_t>a._shape.size())

    if index.size() == 0:
        return

    shape.reserve(nd)
    for i in range(nd):
        s = 1
        for j in index:
            a = arrays[j]
            a_ndim = <Py_ssize_t>a._shape.size()
            if i < nd - a_ndim:
                continue
            a_sh = a._shape[i - (nd - a_ndim)]
            if a_sh == s or a_sh == 1:
                continue
            if s == 1:
                s = a_sh
                continue
            raise ValueError(
                'operands could not be broadcast together with shapes {}'
                .format(
                    ' '.join([str(x.shape) if isinstance(x, _ndarray_base)
                              else '()' for x in arrays])))
        shape.push_back(s)

    for i in index:
        a = arrays[i]
        if vector_equal(a._shape, shape):
            continue

        strides.assign(nd, <Py_ssize_t>0)
        a_ndim = <Py_ssize_t>a._shape.size()
        for j in range(a_ndim):
            a_sh = a._shape[j]
            if a_sh == shape[j + nd - a_ndim]:
                strides[j + nd - a_ndim] = a._strides[j]

        # TODO(niboshi): Confirm update_x_contiguity flags
        arrays[i] = a._view(type(a), shape, strides, True, True, a)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bint _contig_axes(tuple axes):
    # Indicate if the specified axes are in ascending order without gaps.
    cdef Py_ssize_t n
    cdef int n_ax = len(axes)
    cdef bint contig = n_ax > 0
    for n in range(1, n_ax):
        contig = (axes[n] - axes[n - 1]) == 1
        if not contig:
            break
    return contig


cpdef Py_ssize_t _normalize_axis_index(
        Py_ssize_t axis, Py_ssize_t ndim) except -1:
    """
    Normalizes an axis index, ``axis``, such that is a valid positive index
    into the shape of array with ``ndim`` dimensions. Raises a ValueError
    with an appropriate message if this is not possible.

    Args:
        axis (int):
            The un-normalized index of the axis. Can be negative
        ndim (int):
            The number of dimensions of the array that ``axis`` should be
            normalized against

    Returns:
        int:
            The normalized axis index, such that `0 <= normalized_axis < ndim`

    """
    if not (-ndim <= axis < ndim):
        raise AxisError(axis, ndim)
    if axis < 0:
        axis += ndim
    return axis


cpdef tuple _normalize_axis_indices(
        axes, Py_ssize_t ndim, cpp_bool sort_axes=True):
    """Normalize axis indices.

    Args:
        axis (int, tuple of int or None):
            The un-normalized indices of the axis. Can be negative.
        ndim (int):
            The number of dimensions of the array that ``axis`` should be
            normalized against
        sort_axes (bool):
            If provided as False will not sort the axes, default is to return
            the sorted values.

    Returns:
        tuple of int:
            The tuple of normalized axis indices.
    """
    if axes is None:
        axes = tuple(range(ndim))
    elif not isinstance(axes, tuple):
        axes = axes,

    res = []
    for axis in axes:
        axis = _normalize_axis_index(axis, ndim)
        if axis in res:
            # the message in `numpy/core/src/multiarray/conversion_utils.c`
            raise ValueError('duplicate value in \'axis\'')
        res.append(axis)

    return tuple(sorted(res) if sort_axes else res)


cpdef strides_t _get_strides_for_order_K(x, dtype, shape=None):
    # x here can be either numpy.ndarray or cupy.ndarray
    cdef strides_t strides
    # strides used when order='K' for astype, empty_like, etc.

    # Note that there is different semantics of order='K'.
    # See also `_routines_manipulation._npyiter_k_order_axes`.
    stride_and_index = [
        (abs(s), -i) for i, s in enumerate(x.strides)]
    stride_and_index.sort()
    strides.resize(x.ndim)
    stride = dtype.itemsize
    for s, i in stride_and_index:
        strides[-i] = stride
        stride *= shape[-i] if shape else x.shape[-i]
    return strides


cpdef int _update_order_char(
        bint is_c_contiguous, bint is_f_contiguous, int order_char):
    # update order_char based on array contiguity
    if order_char == b'A':
        if is_f_contiguous and not is_c_contiguous:
            order_char = b'F'
        else:
            order_char = b'C'
    elif order_char == b'K':
        if is_c_contiguous:
            order_char = b'C'
        elif is_f_contiguous:
            order_char = b'F'
    return order_char


cpdef tuple _broadcast_shapes(shapes):
    """Broadcast shapes together.

    Args:
        shapes (list of tuples of int):
            shapes that will be broadcasted together.

    Returns:
        tuple of int:
            Resulting shape of broadcasting shapes together.
    """
    out_ndim = max([len(shape) for shape in shapes])
    padded_shapes = [
        (1,) * (out_ndim - len(shape)) + shape for shape in shapes]

    result_shape = []
    for dims in zip(*padded_shapes):
        dims = [dim for dim in dims if dim != 1]
        out_dim = 1 if len(dims) == 0 else dims[0]
        if any([dim != out_dim for dim in dims]):
            raise ValueError(
                'operands could not be broadcast together with shapes' +
                ' '.join([str(shape) for shape in shapes]))
        result_shape.append(out_dim)

    return tuple(result_shape)

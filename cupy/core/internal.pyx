# distutils: language = c++

cimport cpython  # NOQA
cimport cython  # NOQA
from libcpp cimport bool as cpp_bool
from libc.stdint cimport uint32_t

from cupy.core.core cimport ndarray

import sys


cdef extern from 'halffloat.h':
    uint16_t npy_floatbits_to_halfbits(uint32_t f)
    uint32_t npy_halfbits_to_floatbits(uint16_t h)


cdef Py_ssize_t PY_SSIZE_T_MAX = sys.maxsize


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
cpdef inline tuple get_size(object size):
    if size is None:
        return ()
    if cpython.PySequence_Check(size):
        return tuple(size)
    if isinstance(size, int):
        return size,
    raise ValueError('size should be None, collections.abc.Sequence, or int')


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
        vector.vector[Py_ssize_t]& shape, vector.vector[Py_ssize_t]& strides,
        Py_ssize_t itemsize, vector.vector[Py_ssize_t]& reduced_shape,
        vector.vector[Py_ssize_t]& reduced_strides):
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
cpdef vector.vector[Py_ssize_t] get_contiguous_strides(
        const vector.vector[Py_ssize_t]& shape, Py_ssize_t itemsize,
        bint is_c_contiguous):
    cdef vector.vector[Py_ssize_t] strides
    set_contiguous_strides(shape, strides, itemsize, is_c_contiguous)
    return strides


@cython.profile(False)
cdef inline Py_ssize_t set_contiguous_strides(
        const vector.vector[Py_ssize_t]& shape,
        vector.vector[Py_ssize_t]& strides,
        Py_ssize_t itemsize, bint is_c_contiguous):
    cdef Py_ssize_t st, sh
    cdef Py_ssize_t is_nonzero_size = 1
    cdef int i, ndim = shape.size()
    cdef Py_ssize_t idx
    strides.resize(ndim, 0)
    st = 1

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
        vector.vector[Py_ssize_t]& shape, vector.vector[Py_ssize_t]& strides,
        Py_ssize_t itemsize):
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
cpdef vector.vector[Py_ssize_t] infer_unknown_dimension(
        const vector.vector[Py_ssize_t]& shape, Py_ssize_t size) except *:
    cdef vector.vector[Py_ssize_t] ret = shape
    cdef Py_ssize_t cnt=0, index=-1, new_size=1
    for i in range(shape.size()):
        if shape[i] < 0:
            cnt += 1
            index = i
        else:
            new_size *= shape[i]
    if cnt == 0:
        return ret
    if cnt > 1:
        raise ValueError('can only specify only one unknown dimension')
    if (size != 0 and new_size == 0) or size % new_size != 0:
        raise ValueError('total size of new array must be unchanged')
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
    cpdef Py_ssize_t start=0, stop=0, step=0
    cpdef bint start_none, stop_none
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
        raise ValueError('Only one Ellipsis is allowed in index')

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
    if allow_k and (order_char == b'K' or order_char == b'k'):
        order_char = b'K'
    elif order_char == b'A' or order_char == b'a':
        order_char = b'A'
    elif order_char == b'C' or order_char == b'c':
        order_char = b'C'
    elif order_char == b'F' or order_char == b'f':
        order_char = b'F'
    else:
        raise TypeError('order not understood')
    return order_char


cdef _broadcast_core(list arrays, vector.vector[Py_ssize_t]& shape):
    cdef Py_ssize_t i, j, s, smin, smax, a_ndim, a_sh, nd
    cdef vector.vector[Py_ssize_t] strides
    cdef vector.vector[int] index
    cdef ndarray a
    cdef list ret

    shape.clear()
    index.reserve(len(arrays))
    nd = 0
    for i, x in enumerate(arrays):
        if not isinstance(x, ndarray):
            continue
        a = x
        index.push_back(i)
        nd = max(nd, <Py_ssize_t>a._shape.size())

    if index.size() == 0:
        return

    shape.reserve(nd)
    for i in range(nd):
        smin = PY_SSIZE_T_MAX
        smax = 0
        for j in index:
            a = arrays[j]
            a_ndim = <Py_ssize_t>a._shape.size()
            if i >= nd - a_ndim:
                s = a._shape[i - (nd - a_ndim)]
                smin = min(smin, s)
                smax = max(smax, s)
        if smin == 0 and smax > 1:
            raise ValueError(
                'shape mismatch: objects cannot be broadcast to a '
                'single shape')
        shape.push_back(0 if smin == 0 else smax)

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
            elif a_sh != 1:
                raise ValueError(
                    'operands could not be broadcast together with shapes '
                    '{}'.format(
                        ', '.join([str(x.shape) if isinstance(x, ndarray)
                                   else '()' for x in arrays])))

        # TODO(niboshi): Confirm update_x_contiguity flags
        arrays[i] = a._view(shape, strides, True, True)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bint _contig_axes(tuple axes):
    # Indicate if the specified axes are in ascending order without gaps.
    cdef Py_ssize_t n
    cdef bint contig = True
    for n in range(1, len(axes)):
        contig = (axes[n] - axes[n - 1]) == 1
        if not contig:
            break
    return contig

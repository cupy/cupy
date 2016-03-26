cimport cpython
cimport cython

@cython.profile(False)
cpdef inline Py_ssize_t prod(args, Py_ssize_t init=1) except *:
    cdef Py_ssize_t arg
    for arg in args:
        init *= arg
    return init


@cython.profile(False)
cpdef inline Py_ssize_t prod_ssize_t(
        vector.vector[Py_ssize_t]& arr, Py_ssize_t init=1):
    cdef Py_ssize_t a
    for a in arr:
        init *= a
    return init


@cython.profile(False)
cpdef inline tuple get_size(object size):
    if size is None:
        return ()
    if cpython.PySequence_Check(size):
        return tuple(size)
    if isinstance(size, int):
        return size,
    raise ValueError('size should be None, collections.Sequence, or int')


@cython.profile(False)
cpdef inline bint vector_equal(
        vector.vector[Py_ssize_t]& x, vector.vector[Py_ssize_t]& y):
    cdef Py_ssize_t n = x.size()
    if n != y.size():
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
    cdef vector.vector[Py_ssize_t] tmp_shape, tmp_strides
    cdef Py_ssize_t i, ndim, sh, st, prev_st, index
    ndim = shape.size()
    reduced_shape.clear()
    reduced_strides.clear()
    if ndim == 0:
        return

    for i in range(ndim):
        sh = shape[i]
        if sh == 0:
            reduced_shape.push_back(0)
            reduced_strides.push_back(itemsize)
            return
        if sh != 1:
            tmp_shape.push_back(sh)
            tmp_strides.push_back(strides[i])
    if tmp_shape.size() == 0:
        return

    reduced_shape.push_back(tmp_shape[0])
    reduced_strides.push_back(tmp_strides[0])
    index = 0
    for i in range(tmp_shape.size() - 1):
        sh = tmp_shape[i + 1]
        st = tmp_strides[i + 1]
        if tmp_strides[i] == sh * st:
            reduced_shape[index] *= sh
            reduced_strides[index] = st
        else:
            reduced_shape.push_back(sh)
            reduced_strides.push_back(st)
            index += 1


@cython.profile(False)
cpdef vector.vector[Py_ssize_t] get_contiguous_strides(
        vector.vector[Py_ssize_t]& shape, Py_ssize_t itemsize) except *:
    cdef vector.vector[Py_ssize_t] strides
    cdef Py_ssize_t st, sh
    strides.resize(shape.size(), 0)
    st = itemsize
    for i in range(shape.size() - 1, -1, -1):
        strides[i] = st
        sh = shape[i]
        if sh > 1:
            st *= sh
    return strides


@cython.profile(False)
cpdef inline bint get_c_contiguity(
        vector.vector[Py_ssize_t]& shape, vector.vector[Py_ssize_t]& strides,
        Py_ssize_t itemsize) except *:
    cdef vector.vector[Py_ssize_t] r_shape, r_strides
    cpdef Py_ssize_t ndim
    ndim = strides.size()
    if ndim == 0 or (ndim == 1 and strides[0] == itemsize):
        return True
    get_reduced_dims(shape, strides, itemsize, r_shape, r_strides)
    ndim = r_strides.size()
    return ndim == 0 or (ndim == 1 and r_strides[0] == itemsize)


@cython.profile(False)
cpdef vector.vector[Py_ssize_t] infer_unknown_dimension(
        vector.vector[Py_ssize_t]& shape, Py_ssize_t size) except *:
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
cpdef slice complete_slice(slice slc, Py_ssize_t dim):
    cpdef Py_ssize_t start=0, stop=0, step=0
    cpdef bint start_none, stop_none
    if slc.step is None:
        step = 1
    else:
        try:
            step = int(slc.step)
        except TypeError:
            raise IndexError(
                'slice.step must be int or None: {}'.format(slc))

    if step == 0:
        raise ValueError('Slice step must be nonzero.')

    start_none = slc.start is None
    if not start_none:
        try:
            start = int(slc.start)
        except TypeError:
            raise IndexError(
                'slice.start must be int or None: {}'.format(slc))
        if start < 0:
            start += dim

    stop_none = slc.stop is None
    if not stop_none:
        try:
            stop = int(slc.stop)
        except TypeError:
            raise IndexError(
                'slice.stop must be int or None: {}'.format(slc))
        if stop < 0:
            stop += dim

    if step > 0:
        start = 0 if start_none else max(0, min(dim, start))
        stop = dim if stop_none else max(start, min(dim, stop))
    else:
        start = dim - 1 if start_none else max(-1, min(dim - 1, start))
        stop = -1 if stop_none else max(-1, min(start, stop))

    return slice(start, stop, step)

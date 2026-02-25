from libc.string cimport memcpy
from cpython.mem cimport PyMem_Malloc, PyMem_Free

from cupy.cuda cimport function
from cupy._core cimport internal


cdef fused index_t:
    int
    long long


cdef inline int populate_shape_strides(
        index_t val, size_t ndim, bint allow_unsafe,
        void* data, size_t& offset, int itemsize,
        const shape_t& shape, const strides_t& strides) except?-1:
    cdef size_t i

    for i in range(ndim):
        if not allow_unsafe and shape[i] == 0:
            raise RuntimeError(
                f"{i}-th dimension has size zero")
        (<index_t*>(<char*>(data) + offset))[0] = <index_t>shape[i]
        offset += sizeof(index_t)

    for i in range(ndim):
        if not allow_unsafe and strides[i] <= 0:
            raise RuntimeError(
                f"{i}-th dimension has non-positive stride")
        (<index_t*>(<char*>(data) + offset))[0] = (
            <index_t>(strides[i] // itemsize)
        )
        offset += sizeof(index_t)

    return 0


cdef class mdspan(function.CPointer):

    cdef int init(
            self, void* data_ptr, int itemsize,
            const shape_t& shape, const strides_t& strides,
            int index_itemsize, bint allow_unsafe) except?-1:
        cdef size_t ndim = shape.size()
        assert ndim == strides.size()
        assert ndim <= MAX_NDIM

        cdef size_t total_size = \
            sizeof(void*) + ndim * 2 * index_itemsize
        cdef void* data = PyMem_Malloc(total_size)
        if data == NULL:
            raise MemoryError
        self.ptr = <intptr_t>data

        cdef size_t offset = 0
        (<void**>(<char*>(data) + offset))[0] = data_ptr
        offset += sizeof(data_ptr)
        if ndim != 0:
            try:
                if index_itemsize == 4:
                    populate_shape_strides(
                        <int>(0),  # dummy
                        ndim, allow_unsafe, data, offset, itemsize,
                        shape, strides)
                elif index_itemsize == 8:
                    populate_shape_strides(
                        <long long>(0),  # dummy
                        ndim, allow_unsafe, data, offset, itemsize,
                        shape, strides)
                else:
                    raise ValueError(
                        f"Unsupported index_itemsize: {index_itemsize}"
                    )
            except Exception:
                PyMem_Free(data)
                self.ptr = 0
                raise

        assert offset == total_size

    def __cinit__(self):
        self.ptr = 0

    def __dealloc__(self):
        if self.ptr != 0:
            PyMem_Free(<void*>(self.ptr))
            self.ptr = 0


cdef class CArray(function.CPointer):

    cdef _CArray* _allocate(
            self, void* data_ptr, Py_ssize_t element_count,
            size_t ndim) except NULL:
        assert ndim <= MAX_NDIM
        cdef _CArray* data = <_CArray*>PyMem_Malloc(
            sizeof(_CArray) + ndim * 2 * sizeof(Py_ssize_t))
        if data == NULL:
            raise MemoryError
        self.ptr = <intptr_t>data
        self.ndim = ndim
        data.data = data_ptr
        data.size = element_count
        return data

    cdef int init(
            self, void* data_ptr, Py_ssize_t element_count,
            const shape_t& shape, const strides_t& strides) except -1:
        cdef size_t ndim = shape.size()
        assert ndim == strides.size()
        cdef _CArray* data = self._allocate(data_ptr, element_count, ndim)

        if ndim != 0:
            memcpy(data + 1, shape.data(), sizeof(Py_ssize_t) * ndim)
            memcpy(<Py_ssize_t*>(data + 1) + ndim, strides.data(),
                   sizeof(Py_ssize_t) * ndim)
        return 0

    @staticmethod
    cdef CArray create_squashed(
            void* data_ptr, Py_ssize_t element_count,
            const shape_t& shape, const strides_t& strides,
            const vector.vector[cpp_bool]& squashable,
            const vector.vector[int]* output_axes):
        """For a reduction output, output_axes maps the retained suffix of the
        input loop axes to the corresponding axes in shape and strides.
        """
        cdef CArray self = CArray.__new__(CArray)
        cdef size_t ndim = shape.size()
        cdef size_t loop_ndim = squashable.size()
        cdef size_t ax, array_axis, start = 0, curr_ndim = 0, packed_ndim
        assert ndim == strides.size()
        if output_axes == NULL:
            assert loop_ndim <= ndim
            packed_ndim = ndim - loop_ndim
        else:
            assert output_axes[0].size() <= loop_ndim
            start = loop_ndim - output_axes[0].size()
            packed_ndim = 0

        for ax in range(start, loop_ndim):
            if ax == start or not squashable[ax]:
                packed_ndim += 1

        cdef _CArray* data = self._allocate(
            data_ptr, element_count, packed_ndim)
        cdef Py_ssize_t* cshape = <Py_ssize_t*>(data + 1)
        cdef Py_ssize_t* cstrides = cshape + packed_ndim
        for ax in range(start, loop_ndim):
            array_axis = (
                ax if output_axes == NULL else output_axes[0][ax - start])
            if ax == start or not squashable[ax]:
                cshape[curr_ndim] = shape[array_axis]
                cstrides[curr_ndim] = strides[array_axis]
                curr_ndim += 1
            elif shape[array_axis] != 1:
                cshape[curr_ndim - 1] *= shape[array_axis]
                cstrides[curr_ndim - 1] = strides[array_axis]
        if output_axes == NULL:
            for ax in range(loop_ndim, ndim):
                cshape[curr_ndim] = shape[ax]
                cstrides[curr_ndim] = strides[ax]
                curr_ndim += 1
        assert curr_ndim == packed_ndim
        return self

    def __cinit__(self):
        self.ptr = 0

    def __dealloc__(self):
        if self.ptr != 0:
            PyMem_Free(<void*>(self.ptr))
            self.ptr = 0


cdef class CIndexer(function.CPointer):

    @staticmethod
    cdef CIndexer create(const shape_t& shape):
        cdef CIndexer self = CIndexer.__new__(CIndexer)
        cdef size_t ndim = shape.size()
        assert ndim <= MAX_NDIM

        cdef _CIndexer* data = <_CIndexer*>PyMem_Malloc(
            sizeof(_CIndexer) + ndim * 2 * sizeof(Py_ssize_t))
        if data == NULL:
            raise MemoryError
        self.ptr = <intptr_t>data
        self.ndim = ndim
        data.size = internal.prod(shape)

        if ndim != 0:
            # The device fills the index slots when it calls set().
            memcpy(data + 1, shape.data(), sizeof(Py_ssize_t) * ndim)
        return self

    cdef Py_ssize_t size(self) noexcept:
        return (<_CIndexer*>self.ptr).size

    cdef bint _index_32_bits(self) noexcept:
        return self.size() < <Py_ssize_t>(1 << 31)

    def __cinit__(self):
        self.ptr = 0

    def __dealloc__(self):
        if self.ptr != 0:
            PyMem_Free(<void*>(self.ptr))
            self.ptr = 0

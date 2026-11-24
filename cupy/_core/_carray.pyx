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

    @staticmethod
    cdef CArray create(
            void* data_ptr, Py_ssize_t element_count,
            const shape_t& shape, const strides_t& strides):
        cdef CArray self = CArray.__new__(CArray)
        cdef size_t ndim = shape.size()
        assert ndim == strides.size()
        cdef _CArray* data = self._allocate(data_ptr, element_count, ndim)

        if ndim != 0:
            memcpy(data + 1, shape.data(), sizeof(Py_ssize_t) * ndim)
            memcpy(<Py_ssize_t*>(data + 1) + ndim, strides.data(),
                   sizeof(Py_ssize_t) * ndim)
        return self

    @staticmethod
    cdef CArray from_packed_layout(
            void* data_ptr, Py_ssize_t element_count,
            const shape_t& shape, const strides_t& strides,
            const shape_t& loop_shape, const vector.vector[int]& kept_axes,
            int core_ndim):
        """Create a CArray which assumes `loop_shape` is the correct shape
        except for the `core_ndim` (which come from shape).
        """
        cdef CArray self = CArray.__new__(CArray)
        cdef size_t loop_ndim = loop_shape.size()
        cdef size_t packed_ndim = loop_ndim + core_ndim
        cdef size_t ax
        assert loop_ndim == kept_axes.size()
        assert shape.size() == strides.size()
        cdef size_t core_start = shape.size() - core_ndim

        cdef _CArray* data = self._allocate(
            data_ptr, element_count, packed_ndim)
        cdef Py_ssize_t* cshape = <Py_ssize_t*>(data + 1)
        cdef Py_ssize_t* cstrides = cshape + packed_ndim
        for ax in range(loop_ndim):
            cshape[ax] = loop_shape[ax]
            cstrides[ax] = strides[kept_axes[ax]]
        for ax in range(core_ndim):
            cshape[loop_ndim + ax] = shape[core_start + ax]
            cstrides[loop_ndim + ax] = strides[core_start + ax]
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

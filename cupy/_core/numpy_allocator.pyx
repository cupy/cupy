# distutils: language = c++

cimport cython

from libc cimport errno
from libc cimport stdlib
from libc.stdint cimport intptr_t
from libc.string cimport memset, memcpy

cimport cpython
cimport cython

cimport numpy as cnp

from cupy.cuda cimport device
from cupy.cuda cimport memory
from cupy_backends.cuda.api cimport runtime


cdef extern from "numpy/ndarraytypes.h" nogil:
    cpython.PyObject* PyArray_HANDLER(cnp.ndarray arr)


# Similar to, but simplfied from, the NumPy alloc.cpp code
cdef extern from *:
    r"""
    #if defined(linux) || defined(__linux) || defined(__linux__)
    #include <sys/mman.h>
    #include <stdint.h>
    #endif

    static inline void hint_hugepages(void *ptr, size_t size) {
    #ifdef MADV_HUGEPAGE
        if (size < (1U << 22U)) {
            return;  // do nothing for small arrays.
        }
        uintptr_t offset = 4096U - (uintptr_t)ptr % 4096U;
        madvise(ptr + offset, size - offset, MADV_HUGEPAGE);
    #endif
    }
    """
    void hint_hugepages(void *ptr, size_t size) noexcept nogil


IF UNAME_SYSNAME == "Windows":
    cdef extern from "stdlib.h" nogil:
        void * _aligned_malloc(size_t size, size_t alignment)
        void _aligned_free(void * memblock)

    cdef inline void * aligned_alloc(size_t alignment, size_t size) noexcept nogil:  # noqa
        return _aligned_malloc(size, alignment)

    cdef inline void aligned_free(void * ptr) noexcept nogil:
        _aligned_free(ptr)
ELSE:
    cdef extern from * nogil:
        void * aligned_alloc(size_t alignment, size_t size)

    cdef inline void aligned_free(void * ptr) noexcept nogil:
        stdlib.free(ptr)


# CuPy mempool requirement, see ALLOCATION_UNIT_SIZE in cupy/cuda/memory.pyx
cdef const int ALIGNMENT = 512


cdef public void* _calloc(size_t nmemb, size_t size) noexcept nogil:
    errno.errno = 0
    cdef void* buf = aligned_alloc(ALIGNMENT, nmemb * size)
    if buf and errno.errno == 0:
        hint_hugepages(buf, nmemb * size)
        buf = memset(buf, 0, nmemb * size)

    return buf


cdef public void* _malloc(size_t size) noexcept nogil:
    cdef void *buf
    errno.errno = 0
    # TODO: Use madvise hugepages (for larger allocations at least)
    buf = aligned_alloc(ALIGNMENT, size)
    hint_hugepages(buf, size)
    return buf


@cython.cdivision(True)
cdef public void* _realloc(void *ptr, size_t size) noexcept nogil:
    errno.errno = 0
    cdef void* buf = stdlib.realloc(ptr, size)
    cdef void* tmp

    if buf and errno.errno == 0 and <intptr_t>(buf) % ALIGNMENT != 0:
        tmp = buf
        errno.errno = 0
        buf = aligned_alloc(ALIGNMENT, size)
        if buf and errno.errno == 0:
            buf = memcpy(buf, tmp, size)
            stdlib.free(tmp)

    return buf


cdef public void _free(void* ptr) noexcept nogil:
    aligned_free(ptr)

cdef void *_malloc_system(void *ctx, size_t size) noexcept nogil:
    return _malloc(size)

cdef void *_calloc_system(void *ctx, size_t nmemb, size_t size) noexcept nogil:
    return _calloc(nmemb, size)

cdef void *_realloc_system(void *ctx, void* ptr, size_t size) noexcept nogil:
    return _realloc(ptr, size)

cdef void _free_system(void *ctx, void* ptr, size_t size) noexcept nogil:
    _free(ptr)

# Define an allocator for aligned memory (only useful with UMP supported)
cdef cnp.PyDataMem_Handler _aligned_handler
# The init.pyd file uses `char *` rather than `char[127]`, making awkward:
memcpy(_aligned_handler.name, b"cupy_aligned_handler", 21)
_aligned_handler.version = 1
_aligned_handler.allocator.ctx = NULL
_aligned_handler.allocator.malloc = &_malloc_system
_aligned_handler.allocator.calloc = &_calloc_system
_aligned_handler.allocator.realloc = &_realloc_system
_aligned_handler.allocator.free = &_free_system


# Define helpers/functions for a NumPy allocator backed by managed memory.
cdef dict _managed_blocks = {}

cdef void* _malloc_managed(void *ctx, size_t size) noexcept:
    cdef memory.MemoryPointer mem
    try:
        mem = memory.malloc_managed(size)
    except MemoryError as e:
        # don't print out memory error, it adds nothing?
        return NULL

    # TODO: Should managed memory also use hint_hugepages?
    #       (Although if the case, that should be in memory.pyx!)
    _managed_blocks[mem.ptr] = mem
    return <void *>mem.ptr


cdef void* _calloc_managed(void *ctx, size_t nmemb, size_t size) noexcept:
    cdef memory.MemoryPointer mem
    try:
        mem = memory.malloc_managed(nmemb * size)
    except MemoryError as e:
        # don't print out memory error, it adds nothing?
        return NULL
    except:  # noqa: E722
        # noexcept would ensure this, but let's be explicit
        cpython.PyErr_WriteUnraisable("CuPy malloc_managed")
        return NULL

    memset(<void *>mem.ptr, 0, nmemb * size)
    _managed_blocks[mem.ptr] = mem
    return <void *>mem.ptr


cdef void* _realloc_managed(
    void *ctx, void* ptr, size_t size
) noexcept with gil:
    # Note(seberg): It appears NumPy uses realloc sometimes without the GIL
    cdef memory.MemoryPointer old_mem = _managed_blocks[<intptr_t>ptr]
    cdef void *new_ptr

    if size <= old_mem.mem.size:
        return ptr  # just keep using the old allocation.
    else:
        new_ptr = _malloc_managed(ctx, size)
        if new_ptr == NULL:
            return NULL
        memcpy(new_ptr, ptr, old_mem.mem.size)
        del _managed_blocks[old_mem.ptr]  # free old pointer
        return new_ptr


cdef void _free_managed(void *ctx, void* ptr, size_t size) noexcept:
    del _managed_blocks[<intptr_t>ptr]


# Define an allocator using managed memory
cdef cnp.PyDataMem_Handler _managed_handler
# The init.pyd file uses `char *` rather than `char[127]`, making awkward:
memcpy(_managed_handler.name, b"cupy_managed_handler", 21)
_managed_handler.version = 1
_managed_handler.allocator.ctx = NULL
_managed_handler.allocator.malloc = &_malloc_managed
_managed_handler.allocator.calloc = &_calloc_managed
_managed_handler.allocator.realloc = &_realloc_managed
_managed_handler.allocator.free = &_free_managed


cdef object _aligned_capsule = cpython.PyCapsule_New(
    &_aligned_handler, b"mem_handler", NULL)
cdef object _managed_capsule = cpython.PyCapsule_New(
    &_managed_handler, b"mem_handler", NULL)


cdef array_uses_cupy_allocator(cnp.ndarray arr):
    """Check if the array uses the CuPy allocator.

    Note that the alternative (and more general) approach would be to
    check the cuda pointer attributes. If that succeeds, we know the data is
    managed memory.
    """
    cdef cpython.PyObject *handler
    cdef cpython.PyObject *base = arr.base
    while base != NULL and isinstance(<object>base, cnp.ndarray):
        arr = <cnp.ndarray>base
        base = arr.base

    handler = PyArray_HANDLER(arr)  # may be NULL
    if handler == <cpython.PyObject *>_aligned_capsule:
        return "system"
    elif handler == <cpython.PyObject *>_managed_capsule:
        return "managed"
    else:
        return None


@cython.final
cdef class CuPyNumPyAllocator:
    cdef object _handler
    cdef object _prev_handler
    cdef object kind

    def __cinit__(self, kind=None):
        """Initialize the NumPy allcator.

        The allocator object can be used as a context manager or via
        `.use()` globally.
        The typical use should be ``with CuPyNumPyAllocator(): ...``

        Args:
            kind (str): The kind of the allocator, maybe queried as ``.kind``
                after initialization. Values are:
                - "system": Use aligned system memory.
                - "managed": Use managed allocator.
                - None: Uses system memory if the system supports HMM direct
                  memory access (hardware or software support).
        """
        if kind is None:
            # If the system supports direct system memory access (hardware or
            # software) we default to the system allocator.
            if runtime.deviceGetAttribute(
                runtime.cudaDevAttrPageableMemoryAccess,
                device.get_device_id(),
            ) == 1:
                kind = "system"
            else:
                kind = "managed"

        if kind == "system":
            self._handler = _aligned_capsule
        elif kind == "managed":
            self._handler = _managed_capsule
        else:
            raise ValueError(f"Invalid allocator kind: {kind}")

        self.kind = kind

    cpdef use(self):
        if self._prev_handler is not None:
            raise RuntimeError("Cannot use/enter allocator twice.")

        self._prev_handler = cnp.PyDataMem_SetHandler(self._handler)

    cpdef restore(self):
        if self._prev_handler is None:
            raise RuntimeError("Allocator not in use.")

        cnp.PyDataMem_SetHandler(self._prev_handler)
        self._prev_handler = None

    def __enter__(self):
        self.use()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.restore()

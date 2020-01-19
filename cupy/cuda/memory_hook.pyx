import collections
import threading

cdef object _thread_local = threading.local()


cdef class _ThreadLocal:

    cdef object memory_hooks

    def __init__(self):
        self.memory_hooks = None

    @staticmethod
    cdef _ThreadLocal get():
        try:
            tls = _thread_local.tls
        except AttributeError:
            tls = _thread_local.tls = _ThreadLocal()
        return <_ThreadLocal>tls


cpdef bint _has_memory_hooks():
    tls = _ThreadLocal.get()
    return tls.memory_hooks is not None


cpdef get_memory_hooks():
    tls = _ThreadLocal.get()
    if tls.memory_hooks is None:
        tls.memory_hooks = collections.OrderedDict()
    return tls.memory_hooks


class MemoryHook(object):
    """Base class of hooks for Memory allocations.

    :class:`~cupy.cuda.MemoryHook` is an callback object.
    Registered memory hooks are invoked before and after
    memory is allocated from GPU device, and
    memory is retrieved from memory pool, and
    memory is released to memory pool.

    Memory hooks that derive :class:`MemoryHook` are required
    to implement six methods:
    :meth:`~cupy.cuda.MemoryHook.alloc_preprocess`,
    :meth:`~cupy.cuda.MemoryHook.alloc_postprocess`,
    :meth:`~cupy.cuda.MemoryHook.malloc_preprocess`,
    :meth:`~cupy.cuda.MemoryHook.malloc_postprocess`,
    :meth:`~cupy.cuda.MemoryHook.free_preprocess`, and
    :meth:`~cupy.cuda.MemoryHook.free_postprocess`,
    By default, these methods do nothing.

    Specifically, :meth:`~cupy.cuda.MemoryHook.alloc_preprocess`
    (resp. :meth:`~cupy.cuda.MemoryHook.alloc_postprocess`)
    of all memory hooks registered are called before (resp. after)
    memory is allocated from GPU device.

    Likewise, :meth:`~cupy.cuda.MemoryHook.malloc_preprocess`
    (resp. :meth:`~cupy.cuda.MemoryHook.malloc_postprocess`)
    of all memory hooks registered are called before (resp. after)
    memory is retrieved from memory pool.

    Below is a pseudo code to descirbe how malloc and hooks work.
    Please note that :meth:`~cupy.cuda.MemoryHook.alloc_preprocess` and
    :meth:`~cupy.cuda.MemoryHook.alloc_postprocess` are not invoked if a cached
    free chunk is found::

        def malloc(size):
            Call malloc_preprocess of all memory hooks
            Try to find a cached free chunk from memory pool
            if chunk is not found:
                Call alloc_preprocess for all memory hooks
                Invoke actual memory allocation to get a new chunk
                Call alloc_postprocess for all memory hooks
            Call malloc_postprocess for all memory hooks

    Moreover, :meth:`~cupy.cuda.MemoryHook.free_preprocess`
    (resp. :meth:`~cupy.cuda.MemoryHook.free_postprocess`)
    of all memory hooks registered are called before (resp. after)
    memory is released to memory pool.

    Below is a pseudo code to descirbe how free and hooks work::

        def free(ptr):
            Call free_preprocess of all memory hooks
            Push a memory chunk of a given pointer back to memory pool
            Call free_postprocess for all memory hooks

    To register a memory hook, use ``with`` statement. Memory hooks
    are registered to all method calls within ``with`` statement
    and are unregistered at the end of ``with`` statement.

    .. note::

       CuPy stores the dictionary of registered function hooks
       as a thread local object. So, memory hooks registered
       can be different depending on threads.
    """

    name = 'MemoryHook'

    def __enter__(self):
        memory_hooks = get_memory_hooks()
        if self.name in memory_hooks:
            raise KeyError('memory hook %s already exists' % self.name)

        memory_hooks[self.name] = self
        return self

    def __exit__(self, *_):
        del get_memory_hooks()[self.name]

    def alloc_preprocess(self, **kwargs):
        """Callback function invoked before allocating memory from GPU device.

        Keyword Args:
            device_id(int): CUDA device ID
            mem_size(int): Rounded memory bytesize to be allocated
        """
        pass

    def alloc_postprocess(self, **kwargs):
        """Callback function invoked after allocating memory from GPU device.

        Keyword Args:
            device_id(int): CUDA device ID
            mem_size(int): Rounded memory bytesize allocated
            mem_ptr(int): Obtained memory pointer.
                0 if an error occurred in allocation.
        """
        pass

    def malloc_preprocess(self, **kwargs):
        """Callback function invoked before retrieving memory from memory pool.

        Keyword Args:
            device_id(int): CUDA device ID
            size(int): Requested memory bytesize to allocate
            mem_size(int): Rounded memory bytesize to be allocated
        """
        pass

    def malloc_postprocess(self, **kwargs):
        """Callback function invoked after retrieving memory from memory pool.

        Keyword Args:
            device_id(int): CUDA device ID
            size(int): Requested memory bytesize to allocate
            mem_size(int): Rounded memory bytesize allocated
            mem_ptr(int): Obtained memory pointer.
                0 if an error occurred in ``malloc``.
            pmem_id(int): Pooled memory object ID.
                0 if an error occurred in ``malloc``.
        """
        pass

    def free_preprocess(self, **kwargs):
        """Callback function invoked before releasing memory to memory pool.

        Keyword Args:
            device_id(int): CUDA device ID
            mem_size(int): Memory bytesize
            mem_ptr(int): Memory pointer to free
            pmem_id(int): Pooled memory object ID.
        """
        pass

    def free_postprocess(self, **kwargs):
        """Callback function invoked after releasing memory to memory pool.

        Keyword Args:
            device_id(int): CUDA device ID
            mem_size(int): Memory bytesize
            mem_ptr(int): Memory pointer to free
            pmem_id(int): Pooled memory object ID.
        """
        pass

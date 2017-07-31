import collections
import threading

thread_local = threading.local()


def get_memory_hooks():
    if not hasattr(thread_local, 'memory_hooks'):
        thread_local.memory_hooks = collections.OrderedDict()
    return thread_local.memory_hooks


class MemoryHook(object):
    """Base class of hooks for Memory allocations.

    :class:`~cupy.cuda.MemoryHook` is an callback object
    that is registered to :class:`~cupy.cuda.SingleDeviceMemoryPool`.
    Registered memory hooks are invoked before and after
    memory allocation from GPU device is invoked, and
    memory retrieval from memory pool, that is,
    :meth:`~cupy.cuda.SingleDeviceMemoryPool.malloc` is invoked.

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
    memory allocation from GPU device is invoked.

    Likewise, :meth:`~cupy.cuda.MemoryHook.malloc_preprocess`
    (resp. :meth:`~cupy.cuda.MemoryHook.malloc_postprocess`)
    of all memory hooks registered are called before (resp. after)
    memory retrieval from memory pool, that is,
    :meth:`~cupy.cuda.SingleDeviceMemoryPool.malloc` is invoked.

    Moreover, :meth:`~cupy.cuda.MemoryHook.free_preprocess`
    (resp. :meth:`~cupy.cuda.MemoryHook.free_postprocess`)
    of all memory hooks registered are called before (resp. after)
    memory is released to memory pool, that is,
    :meth:`~cupy.cuda.memory.PooledMemory.free` is invoked.

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
            pmem_id(int): PooledMemory object ID.
                0 if an error occurred in ``malloc``.
        """
        pass

    def free_preprocess(self, **kwargs):
        """Callback function invoked before releasing memory to memory pool.

        Keyword Args:
            device_id(int): CUDA device ID
            mem_size(int): Memory bytesize
            mem_ptr(int): Memory pointer to free
            pmem_id(int): PooledMemory object ID.
        """
        pass

    def free_postprocess(self, **kwargs):
        """Callback function invoked after releasing memory to memory pool.

        Keyword Args:
            device_id(int): CUDA device ID
            mem_size(int): Memory bytesize
            mem_ptr(int): Memory pointer to free
            pmem_id(int): PooledMemory object ID.
        """
        pass

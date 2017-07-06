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
    to implement four methods:
    :meth:`~cupy.cuda.MemoryHook.alloc_preprocess`,
    :meth:`~cupy.cuda.MemoryHook.alloc_postprocess`,
    :meth:`~cupy.cuda.MemoryHook.malloc_preprocess`, and
    :meth:`~cupy.cuda.MemoryHook.malloc_postprocess`.
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

    To register a memory hook, use ``with`` statement. Memory hooks
    are registered to all method calls within ``with`` statement
    and are unregistered at the end of ``with`` statement.

    .. note::

       Cupy stores the dictionary of registered function hooks
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

    def alloc_preprocess(self, device_id, rounded_size):
        """Callback function invoked before raw memory allocation.

        Args:
            device_id(int): CUDA device ID
            rounded_size(int): rounded bytesize
        """
        pass

    def alloc_postprocess(self, device_id, rounded_size):
        """Callback function invoked after raw memory allocation.

        Args:
            device_id(int): CUDA device ID
            rounded_size(int): rounded bytesize
        """
        pass

    def malloc_preprocess(self, device_id, size, rounded_size):
        """Callback function invoked before memory retrieval via memory pool.

        Args:
            device_id(int): CUDA device ID
            size(int): bytesize requested by users
            rounded_size(int): bytesize rounded to retrieve from memory pool
        """
        pass

    def malloc_postprocess(self, device_id, size, rounded_size):
        """Callback function invoked after memory retrieval via memory pool.

        Args:
            device_id(int): CUDA device ID
            size(int): bytesize requested by users
            rounded_size(int): bytesize rounded to retrieve from memory pool
        """
        pass

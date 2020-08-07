# distutils: language = c++

from cupy_backends.cuda.api cimport runtime
from cupy.cuda cimport device
from cupy.cuda cimport memory

import threading

from cupy import util
from cupy.cuda import cufft


cdef object _thread_local = threading.local()


cdef class _ThreadLocal:

    cdef list per_device_cufft_cache

    def __init__(self):
        cdef int i
        self.per_device_cufft_cache = [
            None for i in range(runtime.getDeviceCount())]

    @staticmethod
    cdef _ThreadLocal get():
        cdef _ThreadLocal tls
        tls = getattr(_thread_local, 'tls', None)
        if tls is None:
            tls = _ThreadLocal()
            setattr(_thread_local, 'tls', tls)
        return tls


cdef class _Node:
    # Unfortunately cython cdef class cannot be nested, so the node class
    # has to live outside of the linked list...

    # data
    cdef readonly tuple key
    cdef readonly object plan
    cdef readonly Py_ssize_t memsize

    # link
    cdef _Node prev
    cdef _Node next

    def __init__(self, tuple key, plan=None):
        self.key = key
        self.plan = plan
        self.memsize = 0

        cdef memory.MemoryPointer ptr
        cdef int dev, curr_dev
        # work_area could be None for "empty" plans...
        if plan is not None and plan.work_area is not None:
            if isinstance(plan.work_area, list):
                # multi-GPU plan
                curr_dev = runtime.getDevice()
                for dev, ptr in zip(plan.gpus, plan.work_area):
                    if dev == curr_dev:
                        self.memsize = ptr.mem.size
                        break
                else:
                    raise RuntimeError('invalid multi-GPU plan')
            else:
                # single-GPU plan
                self.memsize = <Py_ssize_t>plan.work_area.mem.size

        self.prev = None
        self.next = None

    def __repr__(self):
        cdef str output
        cdef str plan_type
        if isinstance(self.plan, cufft.Plan1d):
            plan_type = 'Plan1d'
        elif isinstance(self.plan, cufft.PlanNd):
            plan_type = 'PlanNd'
        else:
            raise TypeError('unrecognized plan type: {}'.format(
                type(self.plan)))
        output = 'key: {0}, plan type: {1}, memory usage: {2}'.format(
            self.key, plan_type, self.memsize)
        return output


cdef class _LinkedList:
    # link
    cdef _Node head
    cdef _Node tail
    cdef readonly size_t count

    def __init__(self):
        """ A doubly linked list to be used as an LRU cache. """
        self.head = _Node(None, None)
        self.tail = _Node(None, None)
        self.count = 0
        self.head.next = self.tail
        self.tail.prev = self.head

    cdef void remove_node(self, _Node node):
        """ Remove the node from the linked list. """
        cdef _Node p = node.prev
        cdef _Node n = node.next
        p.next = n
        n.prev = p
        self.count -= 1

    cdef void append_node(self, _Node node):
        """ Add a node to the tail of the linked list. """
        cdef _Node t = self.tail
        cdef _Node p = t.prev
        p.next = node
        t.prev = node
        node.prev = p
        node.next = t
        self.count += 1


cdef class PlanCache:
    # total number of plans, regardless of plan type
    # -1: unlimited/ignored, cache size is restricted by "memsize"
    # 0: disable cache
    cdef Py_ssize_t size

    # current number of cached plans
    cdef Py_ssize_t curr_size

    # total amount of memory for all of the cached plans
    # -1: unlimited/ignored, cache size is restricted by "size"
    # 0: disable cache
    cdef Py_ssize_t memsize

    # current amount of memory used by cached plans
    cdef Py_ssize_t curr_memsize

    # whether the cache is enabled (True) or disabled (False)
    cdef bint is_enabled

    # key: all arguments used to construct Plan1d or PlanNd
    # value: the node that holds the plan corresponding to the key
    cdef dict cache

    # for keeping track of least recently used plans
    # lru.head: least recent used
    # lru.tail: most recent used
    cdef _LinkedList lru

    cdef inline void _reset(self):
        self.curr_size = 0
        self.curr_memsize = 0
        self.cache = {}
        self.lru = _LinkedList()

    def __init__(self, Py_ssize_t size=16, Py_ssize_t memsize=-1):
        # TODO(leofang): use stream as part of cache key?
        self._validate_size_memsize(size, memsize)
        self._set_size_memsize(size, memsize)
        self._reset()

    def __getitem__(self, tuple key):
        # no-op if cache is disabled
        if not self.is_enabled:
            assert (self.size == 0 or self.memsize == 0)
            return

        cdef _Node node = self.cache.get(key)
        cdef int dev
        cdef PlanCache cache
        cdef list plans
        if node is not None:
            # hit, move the node to the end
            plan = node.plan
            if plan.gpus is None:
                self.lru.remove_node(node)
                self.lru.append_node(node)
            else:
                plans = []
                for dev in plan.gpus:
                    with device.Device(dev):
                        cache = get_plan_cache()
                        node = cache.cache.get(key)
                        cache.lru.remove_node(node)
                        cache.lru.append_node(node)
                        plans.append(node.plan)
                for item in plans:
                    assert plan is item
            return plan
        else:
            raise KeyError('plan not found for key: {}'.format(key))

    def __setitem__(self, tuple key, plan):
        cdef _Node node = _Node(key, plan)
        cdef _Node unwanted_node

        # no-op if cache is disabled
        if not self.is_enabled:
            assert (self.size == 0 or self.memsize == 0)
            return

        # First, check for the worst case: the plan is too large to fit in
        # the cache. In this case, we leave the cache intact and return early.
        if (node.memsize > self.memsize > 0):
            raise RuntimeError('cannot insert the plan -- perhaps '
                               'cache size/memsize too small?')

        # Now we ensure we have room to insert, check if the key already exists
        unwanted_node = self.cache.get(key)
        if unwanted_node is not None:
            self._remove_plan(unwanted_node)

        # See if the plan can fit in, if not we remove least used ones
        self._eject_until_fit(
            self.size - 1 if self.size != -1 else -1,
            self.memsize - node.memsize if self.memsize != -1 else -1)

        # At this point we ensure we have room to insert
        self._add_node(node)
        self.cache[key] = node

    def __delitem__(self, tuple key):
        # no-op if cache is disabled
        if not self.is_enabled:
            assert (self.size == 0 or self.memsize == 0)
            return

        cdef _Node node = self.cache.get(key)
        if node is not None:
            self._remove_plan(node)
        else:
            raise KeyError('plan not found for key: {}'.format(key))

    cdef void _validate_size_memsize(
            self, Py_ssize_t size, Py_ssize_t memsize) except*:
        if size < -1 or memsize < -1:
            raise ValueError('invalid input')

    cdef void _set_size_memsize(self, Py_ssize_t size, Py_ssize_t memsize):
        self.size = size
        self.memsize = memsize
        self.is_enabled = (size != 0 and memsize != 0)

    cdef void _remove_plan(self, _Node node):
        cdef list gpus = node.plan.gpus
        if gpus is None:
            self._remove_node(node)
            del self.cache[node.key]
        else:
            # collectively remove the plan from all devices' caches
            _remove_multi_gpu_plan(gpus, node.key)

    cdef void _remove_node(self, _Node node):
        """ Remove the node corresponding to the given plan from the list. """
        # update linked list
        self.lru.remove_node(node)

        # update bookkeeping
        self.curr_size -= 1
        self.curr_memsize -= node.memsize

    cdef void _add_node(self, _Node node):
        """ Add a node corresponding to the given plan to the tail of
        the list.
        """
        # update linked list
        self.lru.append_node(node)

        # update bookkeeping
        self.curr_size += 1
        self.curr_memsize += node.memsize

    cpdef set_size(self, Py_ssize_t size):
        self._validate_size_memsize(size, self.memsize)
        self._eject_until_fit(size, self.memsize)
        self._set_size_memsize(size, self.memsize)

    cpdef Py_ssize_t get_size(self):
        return self.size

    cpdef set_memsize(self, Py_ssize_t memsize):
        self._validate_size_memsize(self.size, memsize)
        self._eject_until_fit(self.size, memsize)
        self._set_size_memsize(self.size, memsize)

    cpdef Py_ssize_t get_memsize(self):
        return self.memsize

    cpdef get(self, tuple key, default=None):
        # behaves as if calling dict.get()
        try:
            plan = self[key]
        except KeyError:
            plan = default
        else:
            # if cache is disabled, plan can be None
            if plan is None:
                plan = default
        return plan

    cdef inline void _eject_until_fit(
            self, Py_ssize_t size, Py_ssize_t memsize):
        cdef _Node unwanted_node
        while True:
            if (self.curr_size == 0
                or ((self.curr_size <= size or size == -1)
                    and (self.curr_memsize <= memsize or memsize == -1))):
                break
            else:
                # remove from the front to free up space
                unwanted_node = self.lru.head.next
                if unwanted_node is not self.lru.tail:
                    self._remove_plan(unwanted_node)

    cpdef clear(self):
        self._reset()

    def __repr__(self):
        # we also validate data when the cache information is needed
        assert len(self.cache) == int(self.lru.count) == self.curr_size
        if self.size >= 0:
            assert self.curr_size <= self.size
        if self.memsize >= 0:
            assert self.curr_memsize <= self.memsize

        cdef str output = '-------------- cuFFT plan cache --------------\n'
        output += 'cache enabled? {}\n'.format(self.is_enabled)
        output += 'current / max size: {0} / {1} (counts)\n'.format(
            self.curr_size,
            '(unlimited)' if self.size == -1 else self.size)
        output += 'current / max memsize: {0} / {1} (bytes)\n'.format(
            self.curr_memsize,
            '(unlimited)' if self.memsize == -1 else self.memsize)
        output += '\ncached plans (least used first):\n'

        # TODO(leofang): maybe traverse from the end?
        cdef _Node node = self.lru.head
        cdef size_t count = 0
        while node.next is not self.lru.tail:
            node = node.next
            output += str(node) + '\n'
            assert self.cache[node.key] is node
            count += 1
        assert count == self.lru.count
        return output[:-1]

    cpdef show_info(self):
        print(self)


cpdef PlanCache get_plan_cache():
    """Get the per-thread, per-device plan cache, or create one if not found.
    """
    cdef _ThreadLocal tls = _ThreadLocal.get()
    cdef int dev = runtime.getDevice()
    cdef PlanCache cache = tls.per_device_cufft_cache[dev]
    if cache is None:
        # not found, do a default initialization
        cache = PlanCache()
        tls.per_device_cufft_cache[dev] = cache
    return cache


# TODO(leofang): remove experimental warning when scipy/scipy#12512 is merged
cpdef Py_ssize_t get_plan_cache_size():
    util.experimental('cupy.fft.cache.get_plan_cache_size')
    cdef PlanCache cache = get_plan_cache()
    return cache.get_size()


# TODO(leofang): remove experimental warning when scipy/scipy#12512 is merged
cpdef set_plan_cache_size(size):
    util.experimental('cupy.fft.cache.set_plan_cache_size')
    cdef PlanCache cache = get_plan_cache()
    cache.set_size(size)


# TODO(leofang): remove experimental warning when scipy/scipy#12512 is merged
cpdef Py_ssize_t get_plan_cache_max_memsize():
    util.experimental('cupy.fft.cache.get_plan_cache_max_memsize')
    cdef PlanCache cache = get_plan_cache()
    return cache.get_memsize()


# TODO(leofang): remove experimental warning when scipy/scipy#12512 is merged
cpdef set_plan_cache_max_memsize(size):
    util.experimental('cupy.fft.cache.set_plan_cache_max_memsize')
    cdef PlanCache cache = get_plan_cache()
    cache.set_memsize(size)


# TODO(leofang): remove experimental warning when scipy/scipy#12512 is merged
cpdef clear_plan_cache():
    util.experimental('cupy.fft.cache.clear_plan_cache')
    cdef PlanCache cache = get_plan_cache()
    cache.clear()


cpdef void add_multi_gpu_plan(tuple key, plan) except*:
    cdef int dev
    cdef PlanCache cache
    cdef list insert_ok = []

    try:
        for dev in plan.gpus:
            with device.Device(dev):
                cache = get_plan_cache()
                cache[key] = plan
            insert_ok.append(dev)
    except Exception as e:
        for dev in insert_ok:
            with device.Device(dev):
                cache = get_plan_cache()
                del cache[key]
        raise RuntimeError(
            'Insert succeeded only on devices {0}:\n{1}'.format(insert_ok, e)).with_traceback(e.__traceback__)
    assert len(insert_ok) == len(plan.gpus)


cdef void _remove_multi_gpu_plan(list gpus, tuple key) except*:
    """ Removal of a multi-GPU plan is triggered when any of the participating
    devices removes the plan from its cache.
    """
    cdef int dev
    cdef _Node node
    cdef PlanCache cache

    for dev in gpus:
        with device.Device(dev):
            cache = get_plan_cache()
            node = cache.cache[key]
            cache._remove_node(node)
            del cache.cache[key]

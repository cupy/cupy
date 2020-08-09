# distutils: language = c++

from cupy_backends.cuda.api cimport runtime
from cupy.cuda cimport device
from cupy.cuda cimport memory

import threading

from cupy import util
from cupy.cuda import cufft


#####################################################################
# Internal implementation                                           #
#####################################################################

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


cdef inline Py_ssize_t _get_plan_memsize(plan) except*:
    cdef Py_ssize_t memsize = 0
    cdef memory.MemoryPointer ptr
    cdef int dev, curr_dev

    # work_area could be None for "empty" plans...
    if plan is not None and plan.work_area is not None:
        if isinstance(plan.work_area, list):
            # multi-GPU plan
            curr_dev = runtime.getDevice()
            for dev, ptr in zip(plan.gpus, plan.work_area):
                if dev == curr_dev:
                    memsize = <Py_ssize_t>(ptr.mem.size)
                    break
            else:
                raise RuntimeError('invalid multi-GPU plan')
        else:
            # single-GPU plan
            ptr = plan.work_area
            memsize = <Py_ssize_t>(ptr.mem.size)

    return memsize


cdef class _Node:
    # Unfortunately cython cdef class cannot be nested, so the node class
    # has to live outside of the linked list...

    # data
    cdef readonly tuple key
    cdef readonly object plan
    cdef readonly Py_ssize_t memsize
    cdef readonly list gpus

    # link
    cdef _Node prev
    cdef _Node next

    def __init__(self, tuple key, plan=None):
        self.key = key
        self.plan = plan
        self.memsize = _get_plan_memsize(plan)
        self.gpus = plan.gpus if plan is not None else None

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


#####################################################################
# cuFFT plan cache                                                  #
#####################################################################

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

    # the ID of the device on which the cached plans are allocated
    cdef int dev

    # key: all arguments used to construct Plan1d or PlanNd
    # value: the node that holds the plan corresponding to the key
    cdef dict cache

    # for keeping track of least recently used plans
    # lru.head: least recent used
    # lru.tail: most recent used
    cdef _LinkedList lru

    # ---------------------- Python methods ---------------------- #

    def __init__(self, Py_ssize_t size=16, Py_ssize_t memsize=-1, int dev=-1):
        # TODO(leofang): use stream as part of cache key?
        self._validate_size_memsize(size, memsize)
        self._set_size_memsize(size, memsize)
        self._reset()
        if dev == -1:
            self.dev = runtime.getDevice()
        else:
            self.dev = dev

    def __getitem__(self, tuple key):
        # no-op if cache is disabled
        if not self.is_enabled:
            assert (self.size == 0 or self.memsize == 0)
            return

        cdef _Node node
        cdef int dev
        cdef PlanCache cache
        cdef list gpus

        node = self.cache.get(key)
        if node is not None:
            # hit, move the node to the end
            gpus = node.gpus
            if gpus is None:
                self._move_plan_to_end(key=None, node=node)
            else:
                _remove_append_multi_gpu_plan(gpus, key)
            return node.plan
        else:
            raise KeyError('plan not found for key: {}'.format(key))

    def __setitem__(self, tuple key, plan):
        # no-op if cache is disabled
        if not self.is_enabled:
            assert (self.size == 0 or self.memsize == 0)
            return

        # First, check for the worst case: the plan is too large to fit in
        # the cache. In this case, we leave the cache intact and return early.
        # If we have the budget, then try to squeeze in.
        cdef list gpus = plan.gpus
        if gpus is None:
            self._check_plan_fit(plan)
            self._add_plan(key, plan)
        else:
            # check all device's caches
            _check_multi_gpu_plan_fit(gpus, plan)
            # collectively add the plan to all devices' caches
            _add_multi_gpu_plan(gpus, key, plan)

    def __delitem__(self, tuple key):
        cdef _Node node
        cdef list gpus

        # no-op if cache is disabled
        if not self.is_enabled:
            assert (self.size == 0 or self.memsize == 0)
            return

        node = self.cache.get(key)
        if node is not None:
            gpus = node.gpus
            if gpus is None:
                self._remove_plan(key=None, node=node)
            else:
                _remove_multi_gpu_plan(gpus, key)
        else:
            raise KeyError('plan not found for key: {}'.format(key))

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
        output += '\ncached plans (most recently used first):\n'

        cdef _Node node = self.lru.tail
        cdef size_t count = 0
        while node.prev is not self.lru.head:
            node = node.prev
            output += str(node) + '\n'
            assert self.cache[node.key] is node
            count += 1
        assert count == self.lru.count
        return output[:-1]

    # --------------------- internal helpers --------------------- #

    cdef void _reset(self):
        self.curr_size = 0
        self.curr_memsize = 0
        self.cache = {}
        self.lru = _LinkedList()

    cdef void _validate_size_memsize(
            self, Py_ssize_t size, Py_ssize_t memsize) except*:
        if size < -1 or memsize < -1:
            raise ValueError('invalid input')

    cdef void _set_size_memsize(self, Py_ssize_t size, Py_ssize_t memsize):
        self.size = size
        self.memsize = memsize
        self.is_enabled = (size != 0 and memsize != 0)

    cdef void _check_plan_fit(self, plan) except*:
        cdef Py_ssize_t memsize = _get_plan_memsize(plan)
        if (memsize > self.memsize > 0):
            raise RuntimeError('the plan memsize is too large')

    # The four helpers below (_move_plan_to_end, _add_plan, _remove_plan, and
    # _eject_until_fit) only change the internal state of the current device's
    # cache (self); they are not responsible for other devices'.

    cdef void _move_plan_to_end(self, tuple key=None, _Node node=None) except*:
        # either key is None or node is None
        assert (key is None) == (node is not None)
        if node is None:
            node = self.cache.get(key)

        self.lru.remove_node(node)
        self.lru.append_node(node)

    cdef void _add_plan(self, tuple key, plan) except*:
        cdef _Node node = _Node(key, plan)
        cdef _Node unwanted_node

        # Now we ensure we have room to insert, check if the key already exists
        unwanted_node = self.cache.get(key)
        if unwanted_node is not None:
            self._remove_plan(key=None, node=unwanted_node)

        # See if the plan can fit in, if not we remove least used ones
        self._eject_until_fit(
            self.size - 1 if self.size != -1 else -1,
            self.memsize - node.memsize if self.memsize != -1 else -1)

        # At this point we ensure we have room to insert
        self.lru.append_node(node)
        self.cache[node.key] = node
        self.curr_size += 1
        self.curr_memsize += node.memsize

    cdef void _remove_plan(self, tuple key=None, _Node node=None) except*:
        # either key is None or node is None
        assert (key is None) == (node is not None)
        if node is None:
            node = self.cache.get(key)
        elif key is None:
            key = node.key

        self.lru.remove_node(node)
        del self.cache[key]
        self.curr_size -= 1
        self.curr_memsize -= node.memsize

    cdef void _eject_until_fit(
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
                    self._remove_plan(key=None, node=unwanted_node)

    # -------------- helpers also exposed to Python -------------- #

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

    cpdef clear(self):
        self._reset()

    cpdef show_info(self):
        print(self)


# The three functions below are used to collectively add, remove, or move a
# a multi-GPU plan in all devices' caches (per thread). Therefore, they're
# not PlanCache's methods, which focus on the current (device's) cache. This
# module-level definition has an additional benefit that "cdef inline ..."
# can work.

cdef inline void _add_multi_gpu_plan(list gpus, tuple key, plan) except*:
    cdef int dev
    cdef PlanCache cache
    cdef list insert_ok = []

    try:
        for dev in gpus:
            with device.Device(dev):
                cache = get_plan_cache()
                cache._add_plan(key, plan)
            insert_ok.append(dev)
    except Exception as e:
        # clean up and raise
        _remove_multi_gpu_plan(insert_ok, key)
        x = RuntimeError('Insert succeeded only on devices {0}:\n'
                         '{1}'.format(insert_ok, e))
        raise x.with_traceback(e.__traceback__)
    assert len(insert_ok) == len(gpus)


cdef inline void _remove_multi_gpu_plan(list gpus, tuple key) except*:
    """ Removal of a multi-GPU plan is triggered when any of the participating
    devices removes the plan from its cache.
    """
    cdef int dev
    cdef PlanCache cache

    for dev in gpus:
        with device.Device(dev):
            cache = get_plan_cache()
            cache._remove_plan(key=key)


cdef inline void _remove_append_multi_gpu_plan(list gpus, tuple key) except *:
    cdef int dev
    cdef PlanCache cache

    for dev in gpus:
        with device.Device(dev):
            cache = get_plan_cache()
            cache._move_plan_to_end(key=key)


cdef inline void _check_multi_gpu_plan_fit(list gpus, plan) except*:
    cdef int dev
    cdef PlanCache cache

    try:
        for dev in gpus:
            with device.Device(dev):
                cache = get_plan_cache()
                cache._check_plan_fit(plan)
    except RuntimeError as e:
        e.args = (e.args[0] + ' for device {}'.format(cache.dev),)
        raise e


#####################################################################
# Public API                                                        #
#####################################################################

cpdef inline PlanCache get_plan_cache():
    """Get the per-thread, per-device plan cache, or create one if not found.
    """
    cdef _ThreadLocal tls = _ThreadLocal.get()
    cdef int dev = runtime.getDevice()
    cdef PlanCache cache = tls.per_device_cufft_cache[dev]
    if cache is None:
        # not found, do a default initialization
        cache = PlanCache(dev=dev)
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

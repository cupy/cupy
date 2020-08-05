# distutils: language = c++

from libcpp.list cimport list as cpplist
from libcpp cimport vector

from cupy.cuda import cufft


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
        self.memsize = <Py_ssize_t>plan.work_area.mem.size if plan is not None else 0

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
            raise TypeError('unrecognized plan type: {}'.format(type(self.plan)))
        output = 'key: {0}, plan type: {1}, memory usage: {2}'.format( 
            self.key, plan_type, self.memsize)
        return output


cdef class _LinkedList:
    # link
    cdef _Node head
    cdef _Node tail
    cdef size_t count

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
        cdef _Node p = self.tail.prev
        p.next = node
        self.tail.prev = node
        node.prev = p
        node.next = self.tail
        self.count += 1


cdef class PlanCache:
    # total number of plans, regardless of plan type
    # -1: unlimited, cache size is restricted to "memsize"
    cdef Py_ssize_t size

    # current number of cached plans
    cdef Py_ssize_t curr_size

    # total amount of memory for all of the cached plans
    # -1: unlimited, cache size is restricted to "size"
    cdef Py_ssize_t memsize

    # current amount of memory used by cached plans
    cdef Py_ssize_t curr_memsize

    # key: all arguments used to construct Plan1d or PlanNd
    # value: the node that holds the plan corresponding to the key
    cdef dict cache

    # for keeping track of least recently used plans
    # lru.head: most recent used
    # lru.tail: least recent used
    cdef _LinkedList lru

    cdef inline void _reset(self):
        self.curr_size = 0
        self.curr_memsize = 0
        self.cache = {}
        self.lru = _LinkedList()

    def __init__(self, Py_ssize_t size=4, Py_ssize_t memsize=-1):
        self._validate_size_memsize(size, memsize)
        self._reset()
        self.size = size
        self.memsize = memsize

    def __getitem__(self, tuple key):
        cdef _Node node = self.cache.get(key)
        if node is not None:
            # hit, move the node to the end
            self.lru.remove_node(node)
            self.lru.append_node(node)
            return node.plan
        else:
            raise KeyError

    def __setitem__(self, tuple key, plan):
        cdef _Node node, unwanted_node
        cdef size_t i

        unwanted_node = self.cache.get(key)
        if unwanted_node is not None:
            self._remove_plan(unwanted_node)
            # self.cache will be updated later, so don't del

        node = _Node(key, plan)
        while True:
            if ((self.curr_size + 1 <= self.size or self.size == -1)
                    and (self.curr_memsize + node.memsize <= self.memsize
                         or self.memsize == -1)):
                self._add_plan(node)
                self.cache[key] = node
                break
            else:
                # remove from the front to free up space
                unwanted_node = self.lru.head.next
                if unwanted_node is not self.lru.tail:
                    self._remove_plan(unwanted_node)
                    del self.cache[unwanted_node.key]
                else:
                    # the cache is empty and the plan is too large
                    raise RuntimeError('cannot insert the plan -- '
                                       'cache size/memsize too small?')

    cdef void _validate_size_memsize(
            self, Py_ssize_t size, Py_ssize_t memsize) except*:
        if size == memsize == -1:
            raise ValueError('size and memsize cannot be -1 at the same time')
        if size < -1 or memsize < -1:
            raise ValueError('invalid input')
        if ((size == 0 and memsize not in (-1, 0))
                or (memsize == 0 and size not in (-1, 0))):
            raise ValueError('to disable the cache, both size and memsize '
                             'need to be 0')

    cdef void _remove_plan(self, _Node node):
        """ Remove the node corresponding to the given plan from the list. """
        # update linked list
        self.lru.remove_node(node)

        # update bookkeeping
        self.curr_size -= 1
        self.curr_memsize -= node.memsize
    
    cdef void _add_plan(self, _Node node):
        """ Add a node corresponding to the given plan to the tail of the list. """
        # update linked list
        self.lru.append_node(node)

        # update bookkeeping
        self.curr_size += 1
        self.curr_memsize += node.memsize

    cpdef set_size(self, Py_ssize_t size):
        raise NotImplementedError

    cpdef Py_ssize_t get_size(self):
        return self.size

    cpdef set_memsize(self, Py_ssize_t memsize):
        raise NotImplementedError

    cpdef Py_ssize_t get_memsize(self):
        return self.memsize

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
        output += 'max size: {0} (current size: {1})\n'.format(
            '(unlimited)' if self.size == -1 else self.size,
            self.curr_size)
        output += 'max memsize: {0} (current memsize: {1})\n'.format(
            '(unlimited)' if self.memsize == -1 else self.memsize,
            self.curr_memsize)
        output += '\ncached plans:\n'

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

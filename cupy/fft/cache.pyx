# distutils: language = c++

from libcpp.list cimport list as cpplist
from libcpp cimport vector

from cupy.cuda import cufft


cdef class _Node:
    cdef tuple key
    cdef object plan
    cdef Py_ssize_t memsize
    cdef _Node prev
    cdef _Node next

    def __init__(self, tuple key, plan=None):
        self.key = key
        self.plan = plan
        self.memsize = <Py_ssize_t>plan.work_area.mem.size if plan is not None else 0
        self.prev = None
        self.next = None

    def __repr__(self):
        print(self.key, type(self.plan), self.memsize)


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
    # value: plan
    cdef dict cache

    # head: most recent used
    # tail: least recent used
    cdef _Node head
    cdef _Node tail

    #cdef size_t curr_rank

    def __cinit__(self):
        self.size = 0
        self.curr_size = 0
        self.memsize = 0
        self.curr_memsize = 0
        self.cache = {}
        self.head = None
        self.tail = None

    def __init__(self, Py_ssize_t size=4, Py_ssize_t memsize=-1):
        self._validate_size_memsize(size, memsize)

        self.size = size
        self.memsize = memsize
        self.head = _Node(None, None)
        self.tail = _Node(None, None)
        self.head.next = self.tail
        self.tail.prev = self.head

    def __getitem__(self, tuple key):
        cdef _Node node = self.cache.get(key)
        if node is not None:
            # hit, move the node to the end
            self._remove_node(node)
            self._push_back(node)
            return node.plan
        else:
            raise KeyError

    def __setitem__(self, tuple key, plan):
        cdef _Node node, unwanted_node
        cdef size_t i

        unwanted_node = self.cache.get(key)
        if unwanted_node is not None:
            self._remove_node(unwanted_node)
            # self.cache will be updated later, so don't del

        node = _Node(key, plan)
        while True:
            print("in the loop")
            if ((self.curr_size + 1 < self.size or self.size == -1)
                    and (self.curr_memsize + node.memsize < self.memsize or self.memsize == -1)):
                self._push_back(node)
                self.cache[key] = node
                break
            else:
                print("in else")
                # remove from the front to release space
                unwanted_node = self.head.next
                if unwanted_node is not self.tail:
                    print("removeing...", unwanted_node)
                    self._remove_node(unwanted_node)
                    del self.cache[unwanted_node.key]
                else:
                    # the cache is empty and the plan is too large
                    raise RuntimeError('cannot insert the plan')

    cdef void _validate_size_memsize(
            self, Py_ssize_t size, Py_ssize_t memsize) except*:
        if size == memsize == -1:
            raise ValueError('size and memsize cannot be -1 at the same time')
        if size < -1 or memsize < -1:
            raise ValueError

    cdef void _remove_node(self, _Node node):
        """ Remove the node from the linked list. """
        cdef _Node p = node.prev
        cdef _Node n = node.next

        # update linked list
        p.next = n
        n.prev = p

        # update bookkeeping
        self.curr_size -= 1
        self.curr_memsize -= node.memsize
    
    cdef void _push_back(self, _Node node):
        """ Add a node to the tail of the linked list. """
        cdef _Node p = self.tail.prev

        # update linked list
        p.next = node
        self.tail.prev = node
        node.prev = p
        node.next = self.tail

        # update bookkeeping
        self.curr_size += 1
        self.curr_memsize += node.memsize

    cpdef set_size(self, Py_ssize_t size):
        raise NotImplementedError

    cpdef Py_ssize_t get_size(self):
        return self.size

    cpdef set_memsize(self, Py_ssize_t size):
        raise NotImplementedError

    cpdef Py_ssize_t get_memsize(self):
        return self.memsize

    cpdef clear(self):
        raise NotImplementedError

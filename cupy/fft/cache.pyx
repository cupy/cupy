# distutils: language = c++

from libcpp cimport list as cpplist
from libcpp cimport vector

from cupy.cuda import cufft


cdef class _PlanItem:
    cdef object plan
    cdef size_t rank

    def __init__(self, plan, size_t rank):
        self.plan = plan
        self.rank = rank


cdef class PlanCache:
    # total number of plans, regardless of plan type
    # -1: unlimited, cache size is restricted to "memsize"
    cdef Py_ssize_t size

    # total amount of memory for all of the cached plans
    # -1: unlimited, cache size is restricted to "size"
    cdef Py_ssize_t memsize

    # key: all arguments used to construct Plan1d or PlanNd
    # value: (plan, rank)
    cdef dict cache

    ## head: most recent used
    ## tail: least recent used
    #cdef cpplist.list[_PlanItem*] lru_ranking

    cdef vector.vector[_PlanItem*] lru_ranking

    cdef size_t curr_rank

    def __init__(self, Py_ssize_t size=4, Py_ssize_t memsize=-1):
        self._validate_size_memsize(size, memsize)

        if size >= 0:
            self.lru_ranking.reserve(size)
        elif memsize >= 0:
            # can't infer size here, so do nothing
            pass
        
        self.size = size
        self.memsize = memsize
        self.curr_rank = 0

    def __getitem__(self, tuple key):
        return None

    def __setitem__(self, tuple key, plan):
        cdef _PlanItem rank
        cdef tuple out

        out = self.cache.get(key)
        if out is not None:
            old_plan, rank = out
            self.lru_ranking.erase(&rank)
            del old_plan
        self.curr_rank += 1
        rank = _PlanItem(self.curr_rank)
        self.cache[key] = (plan, rank)
        self.lru_ranking.push_front(&rank)

    cdef _validate_size_memsize(
            self, Py_ssize_t size, Py_ssize_t memsize) except -1:
        if size == memsize == -1:
            raise ValueError('size and memsize cannot be -1 at the same time')
        if size < -1 or memsize < -1:
            raise ValueError

    cpdef set_size(self, Py_ssize_t size):
        pass

    cpdef Py_ssize_t get_size(self):
        return self.size

    cpdef set_memsize(self, Py_ssize_t size):
        pass

    cpdef Py_ssize_t get_memsize(self):
        return self.memsize

    cpdef clear(self):
        pass

from libc.stdint cimport intptr_t


cdef class CPointer:
    cdef void* ptr


cdef class Function:

    cdef:
        public Module module
        public intptr_t ptr

    cpdef linear_launch(
        self, args, size_t gridx, size_t blockx, size_t shared_mem=*, stream=*)


cdef class Module:

    cdef:
        public intptr_t ptr

    cpdef load_file(self, filename)
    cpdef load(self, bytes cubin)
    cpdef get_global_var(self, name)
    cpdef get_function(self, name)
    cpdef get_texref(self, name)


cdef class LinkState:

    cdef:
        public intptr_t ptr

    cpdef add_ptr_data(self, unicode data, unicode name)
    cpdef add_ptr_file(self, unicode path)
    cpdef bytes complete(self)

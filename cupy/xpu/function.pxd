# Backend-neutral declarations of the kernel Module/Function API.
#
# Both backend implementations compile against THIS pxd as the module
# `cupy.xpu.function` (the per-backend pyx source file is selected in
# `install/cupy_builder/features/{cuda,ascend}.py`):
#
#   CUDA/HIP: `cupy/cuda/function.pyx`   (upstream location, upstream code)
#   Ascend:   `cupy/xpu/ascend/function.pyx`  (aclrt binary/launch API)
#
# Cimporters therefore simply do
# `from cupy.xpu.function cimport CPointer, Function, Module` on every
# backend -- no `IF` branching needed at cimport sites.
from libc.stdint cimport intptr_t, uintmax_t


cdef class CPointer:
    cdef:
        void* ptr
        # Ascend only: number of bytes appended to the kernel argument buffer
        # by `aclrtKernelArgsAppend` (CUDA passes an array of pointers to the
        # argument values and needs no size).  The CUDA implementation leaves
        # it at 0.
        size_t size


cdef class Function:

    cdef:
        public Module module
        public intptr_t ptr

    cpdef linear_launch(self, size_t size, args, size_t shared_mem=*,
                        size_t block_max_size=*, stream=*,
                        bint enable_cooperative_groups=*)

cdef class Module:

    cdef:
        # CUDA: CUmodule; Ascend: aclrtBinHandle
        public intptr_t ptr
        # Ascend only: the aclrtBinary produced by aclrtCreateBinary for an
        # in-memory load; kept alive because its lifetime is not documented
        # to end at aclrtBinaryLoad time.  Unused by the CUDA implementation.
        cdef intptr_t _binary
        readonly dict mapping

    cpdef load_file(self, filename)
    cpdef load(self, bytes cubin)
    cpdef get_global_var(self, name)
    cpdef get_function(self, name)
    cpdef _set_mapping(self, dict mapping)


cdef class LinkState:

    cdef:
        public intptr_t ptr

    cpdef add_ptr_data(self, bytes data, unicode name)
    cpdef add_ptr_file(self, unicode path)
    cpdef bytes complete(self)


把下面cython代码(cupy)项目中的Function和Module

```
   cdef class Function:

        """CUDA kernel function."""

        def __init__(self, Module module, str funcname):
            self.module = module  # to keep module loaded
            self.ptr = driver.moduleGetFunction(module.ptr, funcname)

        def __call__(self, tuple grid, tuple block, args, size_t shared_mem=0,
                    stream=None, enable_cooperative_groups=False):
            grid = (grid + (1, 1))[:3]
            block = (block + (1, 1))[:3]
            s = _get_stream(stream)
            _launch(
                self.ptr,
                max(1, grid[0]), max(1, grid[1]), max(1, grid[2]),
                max(1, block[0]), max(1, block[1]), max(1, block[2]),
                args, shared_mem, s, enable_cooperative_groups)

        cpdef linear_launch(self, size_t size, args, size_t shared_mem=0,
                            size_t block_max_size=128, stream=None,
                            bint enable_cooperative_groups=False):
            # TODO(beam2d): Tune it
            cdef size_t gridx = min(
                0x7fffffffUL, (size + block_max_size - 1) // block_max_size)
            cdef size_t blockx = min(block_max_size, size)
            s = _get_stream(stream)
            _launch(
                self.ptr,
                gridx, 1, 1, blockx, 1, 1,
                args,
                shared_mem, s, enable_cooperative_groups)


    cdef class Module:

        """CUDA kernel module."""

        def __init__(self):
            self.ptr = 0
            self.mapping = None

        def __dealloc__(self):
            if self.ptr:
                driver.moduleUnload(self.ptr)
                self.ptr = 0

        cpdef load_file(self, filename):
            if isinstance(filename, bytes):
                filename = filename.decode()
            runtime._ensure_context()
            self.ptr = driver.moduleLoad(filename)

        cpdef load(self, bytes cubin):
            runtime._ensure_context()
            self.ptr = driver.moduleLoadData(cubin)

        cpdef get_global_var(self, name):
            if isinstance(name, bytes):
                name = name.decode()
            return driver.moduleGetGlobal(self.ptr, name)

        cpdef get_function(self, name):
            if isinstance(name, bytes):
                name = name.decode()
            return Function(self, name)

        cpdef _set_mapping(self, dict mapping):
            self.mapping = mapping

```
用ascend runtime API, 比如
```
aclError aclrtLaunchKernel(aclrtFuncHandle funcHandle, uint32_t blockDim, const void *argsData, size_t argsSize, aclrtStream stream);
```
    cdef class LinkState:

        """CUDA link state."""

        def __init__(self):
            runtime._ensure_context()
            self.ptr = driver.linkCreate()

        def __dealloc__(self):
            if self.ptr:
                driver.linkDestroy(self.ptr)
                self.ptr = 0

        cpdef add_ptr_data(self, bytes data, unicode name):
            driver.linkAddData(self.ptr, driver.CU_JIT_INPUT_PTX, data, name)

        cpdef add_ptr_file(self, unicode path):
            driver.linkAddFile(self.ptr, driver.CU_JIT_INPUT_LIBRARY, path)

        cpdef bytes complete(self):
            cubin = driver.linkComplete(self.ptr)
            return cubin
#ifndef INCLUDE_GUARD_CUPY_CUDA_MEMORY_H
#define INCLUDE_GUARD_CUPY_CUDA_MEMORY_H

#include "memory_api.h"  // generated by Cython from cupy.cuda.memory

class cupy_device_allocator {
private:
    int is_init;
    void* (*_malloc)(size_t);
    void (*_free)(void*);

public:
/*
    To use this utility, the lifetime of this allocator instance should
    be as long as your shared library or application. Early destruction
    would lead to segfault!

    If this piece of code is already embedded in a Python application,
    we skip calling Py_Initialize() at construction and Py_Finalize()
    at deconstruction.
*/
    cupy_device_allocator() {
        is_init = Py_IsInitialized();
        if (!is_init) {
            Py_Initialize();
        }
        import_cupy__cuda__memory();
        _malloc = cupy_c_malloc; // defined in cupy.cuda.memory
        _free = cupy_c_free;     // defined in cupy.cuda.memory
    }

    ~cupy_device_allocator() {
        _malloc = nullptr;
        _free = nullptr;
        if (!is_init) {
            Py_Finalize();
        }
    }

    void* malloc(size_t n_bytes) {
        void* ptr = _malloc(n_bytes);
        return ptr;
    }
    
    void free(void* ptr) {
        _free(ptr);
    }
}; 
    
#endif // #ifndef INCLUDE_GUARD_CUPY_CUDA_MEMORY_H

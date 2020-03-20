#ifndef INCLUDE_GUARD_CUPY_CUDA_MEMORY_H
#define INCLUDE_GUARD_CUPY_CUDA_MEMORY_H

#include "memory_api.h"


//class cupy_allocator {
//private:
//    void* memory;
//
//public:
//    typedef char value_type;
//
//    cupy_allocator(void* memory) : memory(memory) {}
//
//    char *allocate(std::ptrdiff_t num_bytes) {
//        return cupy_malloc(memory, num_bytes);
//    }
//
//    void deallocate(char *ptr, size_t n) {
//        cupy_free(memory, ptr);
//    }
//};

class cupy_device_allocator {
private:
    void* (*_malloc)(size_t);
    void (*_free)(void*);

public:
    cupy_device_allocator() {
        Py_Initialize();
        import_cupy__cuda__memory();
        _malloc = cupy_c_malloc;
        _free = cupy_c_free;
    }

    ~cupy_device_allocator() {
        _malloc = nullptr;
        _free = nullptr;
        Py_Finalize();
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

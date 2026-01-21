import os
import sys

import pytest

import cupy
from cupy import testing


code_verify = """
#include <cuda/std/mdspan>

// typedef struct {
//     void* ptr;
//     size_t ext1;
//     size_t ext2;
// } mdspan_view_t;
// 
// 
// // Kernel to verify layout_right (C-order) mdspan arguments
// template<typename T>
// __global__ void verify_mdspan_layout_right(
//     mdspan_view_t arr
// ) {
//     // Only thread 0 prints to avoid cluttered output
//     if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
//         printf("=== layout_right (C-order) mdspan ===\\n");
//         printf("sizeof(mdspan_view_t): %llu\\n", sizeof(arr));
//         printf("view - ptr: %p\\n", reinterpret_cast<mdspan_view_t*>(&arr)->ptr);
//         printf("view2 : %p\\n", *(void**)((char*)(&arr) + 0));
//         printf("view - ext1: %p\\n", reinterpret_cast<mdspan_view_t*>(&arr)->ext1);
//         printf("view - ext2: %p\\n", reinterpret_cast<mdspan_view_t*>(&arr)->ext2);
//     }
// }

// Kernel to verify layout_right (C-order) mdspan arguments

typedef struct {
    void* ptr;
    void* ext1;
    void* ext2;
} mdspan_view_t;

template<typename T>
__global__ void verify_mdspan_layout_right(
    cuda::std::mdspan<T, cuda::std::extents<size_t, cuda::std::dynamic_extent, cuda::std::dynamic_extent>, cuda::std::layout_right> arr_in,
    cuda::std::mdspan<T, cuda::std::extents<size_t, cuda::std::dynamic_extent, cuda::std::dynamic_extent>, cuda::std::layout_right> arr_out
) {
    // Only thread 0 prints to avoid cluttered output
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("=== layout_right (C-order) mdspan ===\\n");
        printf("sizeof(mdspan): %llu\\n", sizeof(arr_in));
        printf("view - ptr: %p\\n", reinterpret_cast<mdspan_view_t*>(&arr_in)->ptr);
        printf("view2 : %p\\n", (void**)((char*)(&arr_in) + 0));
        //printf("view - ext1: %llu\\n", *((size_t*)(reinterpret_cast<mdspan_view_t*>(&arr_in)->ext1)));
        //printf("view - ext2: %llu\\n", *((size_t*)(reinterpret_cast<mdspan_view_t*>(&arr_in)->ext2)));
        printf("view - ext1: %p\\n", reinterpret_cast<mdspan_view_t*>(&arr_in)->ext1);
        printf("view - ext2: %p\\n", reinterpret_cast<mdspan_view_t*>(&arr_in)->ext2);

        printf("Data pointer: %p\\n", arr_in.data_handle());
        printf("Data pointer (actual): %p\\n", (void*)((char*)(&arr_in) + 0));
        printf("Data pointer (actual): %p\\n", addressof(arr_in));
        printf("Extent 0 (rows): %llu\\n", arr_in.extent(0));
        printf("Extent 1 (cols): %llu\\n", arr_in.extent(1));
        printf("Extent 0 (rows) (actual): %llu\\n", (size_t)(*((char*)(&arr_in) + 8)));
        printf("Extent 1 (cols) (actual): %llu\\n", (size_t)(*((char*)(&arr_in) + 16)));
        printf("Size: %zu\\n", arr_in.size());
        
        // For layout_right, strides are implicit but we can query them
        printf("Stride 0: %llu\\n", arr_in.stride(0));
        printf("Stride 1: %llu\\n", arr_in.stride(1));
        printf("Stride 0 (actual): %llu\\n", (size_t)((char*)(&arr_in) + 24));
        printf("Stride 1 (actual): %llu\\n", (size_t)((char*)(&arr_in) + 32));
        
        // Verify memory layout: for layout_right (C-order)
        // stride(0) should equal extent(1), stride(1) should be 1
        printf("Expected stride(0) = extent(1): %s\\n", 
               (arr_in.stride(0) == arr_in.extent(1)) ? "PASS" : "FAIL");
        printf("Expected stride(1) = 1: %s\\n", 
               (arr_in.stride(1) == 1) ? "PASS" : "FAIL");
    }

    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < arr_in.extent(0); i += gridDim.x * blockDim.x) {
        for (size_t j = blockIdx.y * (size_t)blockDim.y + threadIdx.y; j < arr_in.extent(1); j += gridDim.y * blockDim.y) {
            // Simple operation: increment each element by 1
            arr_out(i, j) = arr_in(i, j) + 1;
        }
    }
}
"""


#@pytest.mark.parametrize('index_type', [cupy.int32, cupy.int64])
class TestMdspan:

    def setup_class(self):
        self.mod = cupy.RawModule(
            code=code_verify,
            options=('--std=c++17',),
            name_expressions=['verify_mdspan_layout_right<double>']
        )

    def dtype_to_cpp_type(self, dtype):
        #FIXME: we should already have something like this in cupy
        if dtype == cupy.float32:
            return 'float'
        elif dtype == cupy.float64:
            return 'double'
        elif dtype == cupy.int32:
            return 'int'
        elif dtype == cupy.int64:
            return 'long long'
        else:
            raise ValueError(f'Unsupported dtype: {dtype}')

    #@testing.for_all_dtypes()
    def test_mdspan_layout_strided(self):#, dtype, index_type):
        dtype = cupy.float64
        a = testing.shaped_random((4, 8), dtype=dtype)
        a_mdspan = a.mdspan#index_type=index_type)
        out = cupy.zeros_like(a)
        out_mdspan = out.mdspan

        mod = self.mod
        ker = mod.get_function(
            f'verify_mdspan_layout_right<{self.dtype_to_cpp_type(dtype)}>'
        )

        ker((1,), (4, 8), (a_mdspan, out_mdspan))
        assert cupy.all(out == a + 1)

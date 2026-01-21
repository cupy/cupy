from __future__ import annotations

import pytest

import cupy
from cupy import testing
from cupy._core._scalar import get_typename


code_verify = """
#include <cuda/std/mdspan>
#include <cupy/carray.cuh>
#include <cupy/complex.cuh>

template<typename T, typename IndexType>
__global__ void verify_mdspan_layout_stride(
    cuda::std::mdspan<
        T,
        cuda::std::extents<
            IndexType,
            cuda::std::dynamic_extent,
            cuda::std::dynamic_extent>,
        cuda::std::layout_stride> arr_in,
    cuda::std::mdspan<
        T,
        cuda::std::extents<
            IndexType,
            cuda::std::dynamic_extent,
            cuda::std::dynamic_extent>,
        cuda::std::layout_stride> arr_out
) {
    for (IndexType i = blockIdx.x * (IndexType)blockDim.x + threadIdx.x;
         i < arr_in.extent(0);
         i += gridDim.x * blockDim.x) {
        for (IndexType j = blockIdx.y * (IndexType)blockDim.y + threadIdx.y;
             j < arr_in.extent(1);
             j += gridDim.y * blockDim.y) {
            // Simple operation: increment each element by 1
            arr_out(i, j) = arr_in(i, j) + T(1);
        }
    }
}
"""


# Supported index types for mdspan
MDSPAN_INDEX_TYPES = [cupy.int32, cupy.int64]


@pytest.mark.parametrize('index_type', MDSPAN_INDEX_TYPES)
@pytest.mark.skipif(
    cupy.cuda.runtime.is_hip, reason='libcudacxx not supported in HIP'
)
class TestMdspan:

    def setup_class(self):
        # All type chars from cupy/_core/_dtype.pyx
        all_type_chars = '?bhilqBHILQefdFD'
        dtypes = [cupy.dtype(c).type for c in all_type_chars]

        name_expressions = []
        for dtype in dtypes:
            for index_type in MDSPAN_INDEX_TYPES:
                dtype_str = get_typename(dtype)
                index_str = get_typename(index_type)
                name_expressions.append(
                    f'verify_mdspan_layout_stride<{dtype_str}, {index_str}>'
                )

        self.mod = cupy.RawModule(
            code=code_verify,
            options=('--std=c++17',),
            name_expressions=name_expressions
        )

    # TODO(leofang): it does not seem 'AK' makes any difference here?
    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    def test_mdspan_layout_stride(self, dtype, order, index_type):
        shape = (4, 8)

        a = testing.shaped_random(shape, dtype=dtype, order=order)
        a_mdspan = a.mdspan(index_type=index_type)
        out = cupy.zeros_like(a)
        out_mdspan = out.mdspan(index_type=index_type)

        dtype_str = get_typename(dtype)
        index_type_str = get_typename(index_type)

        ker = self.mod.get_function(
            f'verify_mdspan_layout_stride<{dtype_str}, {index_type_str}>'
        )

        ker((1,), shape, (a_mdspan, out_mdspan))
        assert cupy.all(out == a + cupy.ones(shape, dtype=dtype))

    # TODO(leofang): it does not seem 'AK' makes any difference here?
    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    def test_mdspan_layout_stride_sliced(self, dtype, order, index_type):
        shape = (4, 16)
        slice1 = slice(None, None, 2)
        slice2 = slice(None, None, 3)

        a = testing.shaped_random(shape, dtype=dtype, order=order)
        a_mdspan = a[slice1, slice2].mdspan(index_type=index_type)
        out = cupy.zeros_like(a)
        out_mdspan = out[slice1, slice2].mdspan(index_type=index_type)

        dtype_str = get_typename(dtype)
        index_type_str = get_typename(index_type)

        ker = self.mod.get_function(
            f'verify_mdspan_layout_stride<{dtype_str}, {index_type_str}>'
        )

        ker((1,), shape, (a_mdspan, out_mdspan))
        assert cupy.all(out[slice1, slice2] == \
            a[slice1, slice2] + cupy.ones(shape, dtype=dtype)[slice1, slice2])

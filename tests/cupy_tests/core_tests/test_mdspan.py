from __future__ import annotations

import pytest

import cupy
from cupy import testing
from cupy._core._scalar import get_typename


# Supported index types for mdspan
MDSPAN_INDEX_TYPES = [cupy.int32, cupy.int64]


# TODO(leofang): do we have a better source of all dtypes?
# All type chars from cupy/_core/_dtype.pyx
ALL_TYPE_CHARS = '?bhilqBHILQefdFD'


# Kernel code for 1D mdspan
code_verify_1d = """
#include <cuda/std/mdspan>
#include <cupy/carray.cuh>
#include <cupy/complex.cuh>

template<typename T, typename IndexType>
__global__ void verify_mdspan_1d(
    cuda::std::mdspan<
        T,
        cuda::std::extents<IndexType, cuda::std::dynamic_extent>,
        cuda::std::layout_stride> arr_in,
    cuda::std::mdspan<
        T,
        cuda::std::extents<IndexType, cuda::std::dynamic_extent>,
        cuda::std::layout_stride> arr_out
) {
    for (IndexType i = blockIdx.x * (IndexType)blockDim.x + threadIdx.x;
         i < arr_in.extent(0);
         i += gridDim.x * blockDim.x) {
        arr_out(i) = arr_in(i) + T(1);
    }
}
"""


# Kernel code for 2D mdspan
code_verify_2d = """
#include <cuda/std/mdspan>
#include <cupy/carray.cuh>
#include <cupy/complex.cuh>

template<typename T, typename IndexType>
__global__ void verify_mdspan_2d(
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


# Kernel code for 3D mdspan
code_verify_3d = """
#include <cuda/std/mdspan>
#include <cupy/carray.cuh>
#include <cupy/complex.cuh>

template<typename T, typename IndexType>
__global__ void verify_mdspan_3d(
    cuda::std::mdspan<
        T,
        cuda::std::extents<
            IndexType,
            cuda::std::dynamic_extent,
            cuda::std::dynamic_extent,
            cuda::std::dynamic_extent>,
        cuda::std::layout_stride> arr_in,
    cuda::std::mdspan<
        T,
        cuda::std::extents<
            IndexType,
            cuda::std::dynamic_extent,
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
            for (IndexType k = blockIdx.z * (IndexType)blockDim.z
                                       + threadIdx.z;
                 k < arr_in.extent(2);
                 k += gridDim.z * blockDim.z) {
                arr_out(i, j, k) = arr_in(i, j, k) + T(1);
            }
        }
    }
}
"""


# Kernel with compile-time size validation
code_verify_with_validation = """
#include <cuda/std/mdspan>
#include <cupy/carray.cuh>
#include <cupy/complex.cuh>

template<typename T, typename IndexType>
__global__ void verify_mdspan_with_size_check(
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
    // Compile-time validation of mdspan size
    using mdspan_t = decltype(arr_in);
    static_assert(sizeof(mdspan_t) == sizeof(void*) + 4*sizeof(IndexType),
                  "mdspan size does not match expected layout!");

    // Verify all extents are dynamic
    static_assert(mdspan_t::rank_dynamic() == mdspan_t::rank(),
                  "mdspan must have all dynamic extents!");

    for (IndexType i = blockIdx.x * (IndexType)blockDim.x + threadIdx.x;
         i < arr_in.extent(0);
         i += gridDim.x * blockDim.x) {
        for (IndexType j = blockIdx.y * (IndexType)blockDim.y + threadIdx.y;
             j < arr_in.extent(1);
             j += gridDim.y * blockDim.y) {
            arr_out(i, j) = arr_in(i, j);  // copy
        }
    }
}
"""


@pytest.mark.parametrize('index_type', MDSPAN_INDEX_TYPES)
@pytest.mark.skipif(
    cupy.cuda.runtime.is_hip, reason='libcudacxx not supported in HIP'
)
class TestMdspan1D:
    """Test mdspan with 1D arrays."""

    def setup_class(self):
        dtypes = [cupy.dtype(c).type for c in ALL_TYPE_CHARS]

        name_expressions = []
        for dtype in dtypes:
            for index_type in MDSPAN_INDEX_TYPES:
                dtype_str = get_typename(dtype)
                index_str = get_typename(index_type)
                name_expressions.append(
                    f'verify_mdspan_1d<{dtype_str}, {index_str}>'
                )

        self.mod = cupy.RawModule(
            code=code_verify_1d,
            options=('--std=c++17',),
            name_expressions=name_expressions
        )

    @testing.for_all_dtypes()
    def test_mdspan_1d(self, dtype, index_type):
        """Test 1D arrays with mdspan."""
        a = testing.shaped_random((100,), dtype=dtype)
        a_mdspan = a.mdspan(index_type=index_type)
        out = cupy.zeros_like(a)
        out_mdspan = out.mdspan(index_type=index_type)

        dtype_str = get_typename(dtype)
        index_type_str = get_typename(index_type)

        ker = self.mod.get_function(
            f'verify_mdspan_1d<{dtype_str}, {index_type_str}>'
        )

        ker((1,), (100,), (a_mdspan, out_mdspan))
        testing.assert_array_equal(out, a + cupy.ones(100, dtype=dtype))

    @testing.for_all_dtypes()
    def test_mdspan_1d_sliced(self, dtype, index_type):
        """Test 1D sliced arrays with positive strides."""
        a = testing.shaped_random((100,), dtype=dtype)
        a_sliced = a[::3]  # Every 3rd element
        a_mdspan = a_sliced.mdspan(index_type=index_type)
        out = cupy.zeros_like(a)
        out_sliced = out[::3]
        out_mdspan = out_sliced.mdspan(index_type=index_type)

        dtype_str = get_typename(dtype)
        index_type_str = get_typename(index_type)

        ker = self.mod.get_function(
            f'verify_mdspan_1d<{dtype_str}, {index_type_str}>'
        )

        ker((1,), (34,), (a_mdspan, out_mdspan))
        testing.assert_array_equal(
            out_sliced, a_sliced + cupy.ones_like(a_sliced, dtype=dtype)
        )

    @testing.for_all_dtypes()
    def test_mdspan_negative_stride_1d(self, dtype, index_type):
        """Test 1D array with negative stride (reversed)."""
        a = testing.shaped_random((100,), dtype=dtype)
        a_rev = a[::-1]  # Negative stride

        # Check if strides are actually negative
        assert a_rev.strides[0] < 0, "Expected negative stride"

        a_mdspan = a_rev.mdspan(index_type=index_type)
        out = cupy.zeros_like(a)
        out_rev = out[::-1]
        out_mdspan = out_rev.mdspan(index_type=index_type)

        dtype_str = get_typename(dtype)
        index_type_str = get_typename(index_type)

        ker = self.mod.get_function(
            f'verify_mdspan_1d<{dtype_str}, {index_type_str}>'
        )

        # This tests if mdspan correctly handles negative strides
        ker((1,), (100,), (a_mdspan, out_mdspan))
        testing.assert_array_equal(out_rev, a_rev +
                                   cupy.ones(100, dtype=dtype))


@pytest.mark.parametrize('index_type', MDSPAN_INDEX_TYPES)
@pytest.mark.skipif(
    cupy.cuda.runtime.is_hip, reason='libcudacxx not supported in HIP'
)
class TestMdspan2D:

    def setup_class(self):
        dtypes = [cupy.dtype(c).type for c in ALL_TYPE_CHARS]

        name_expressions = []
        for dtype in dtypes:
            for index_type in MDSPAN_INDEX_TYPES:
                dtype_str = get_typename(dtype)
                index_str = get_typename(index_type)
                name_expressions.append(
                    f'verify_mdspan_2d<{dtype_str}, {index_str}>'
                )

        self.mod = cupy.RawModule(
            code=code_verify_2d,
            options=('--std=c++17',),
            name_expressions=name_expressions
        )

    # TODO(leofang): it does not seem 'AK' makes any difference here?
    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    def test_mdspan_2d(self, dtype, order, index_type):
        shape = (4, 8)

        a = testing.shaped_random(shape, dtype=dtype, order=order)
        a_mdspan = a.mdspan(index_type=index_type)
        out = cupy.zeros_like(a)
        out_mdspan = out.mdspan(index_type=index_type)

        dtype_str = get_typename(dtype)
        index_type_str = get_typename(index_type)

        ker = self.mod.get_function(
            f'verify_mdspan_2d<{dtype_str}, {index_type_str}>'
        )

        ker((1,), shape, (a_mdspan, out_mdspan))
        testing.assert_array_equal(out, a + cupy.ones(shape, dtype=dtype))

    # TODO(leofang): it does not seem 'AK' makes any difference here?
    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    def test_mdspan_2d_sliced(self, dtype, order, index_type):
        shape = (4, 16)
        slice1 = slice(None, None, 2)
        slice2 = slice(None, None, 3)

        a = testing.shaped_random(shape, dtype=dtype, order=order)
        a_sliced = a[slice1, slice2]
        a_mdspan = a_sliced.mdspan(index_type=index_type)
        out = cupy.zeros_like(a)
        out_sliced = out[slice1, slice2]
        out_mdspan = out_sliced.mdspan(index_type=index_type)

        dtype_str = get_typename(dtype)
        index_type_str = get_typename(index_type)

        ker = self.mod.get_function(
            f'verify_mdspan_2d<{dtype_str}, {index_type_str}>'
        )

        ker((1,), shape, (a_mdspan, out_mdspan))
        testing.assert_array_equal(
            out_sliced, a_sliced + cupy.ones_like(a_sliced, dtype=dtype)
        )

    @testing.for_all_dtypes()
    def test_mdspan_negative_stride_2d(self, dtype, index_type):
        """Test 2D array with negative stride in one dimension."""
        shape = (10, 20)
        a = testing.shaped_random(shape, dtype=dtype)
        a_rev = a[::-1, :]  # Negative stride in first dimension

        assert a_rev.strides[0] < 0, "Expected negative stride"

        a_mdspan = a_rev.mdspan(index_type=index_type)
        out = cupy.zeros_like(a)
        out_rev = out[::-1, :]
        out_mdspan = out_rev.mdspan(index_type=index_type)

        dtype_str = get_typename(dtype)
        index_type_str = get_typename(index_type)

        ker = self.mod.get_function(
            f'verify_mdspan_2d<{dtype_str}, {index_type_str}>'
        )

        ker((1,), shape, (a_mdspan, out_mdspan))
        testing.assert_array_equal(
            out_rev, a_rev + cupy.ones_like(a_rev, dtype=dtype)
        )


@pytest.mark.parametrize('index_type', MDSPAN_INDEX_TYPES)
@pytest.mark.skipif(
    cupy.cuda.runtime.is_hip, reason='libcudacxx not supported in HIP'
)
class TestMdspan3D:
    """Test mdspan with 3D arrays."""

    def setup_class(self):
        dtypes = [cupy.dtype(c).type for c in ALL_TYPE_CHARS]

        name_expressions = []
        for dtype in dtypes:
            for index_type in MDSPAN_INDEX_TYPES:
                dtype_str = get_typename(dtype)
                index_str = get_typename(index_type)
                name_expressions.append(
                    f'verify_mdspan_3d<{dtype_str}, {index_str}>'
                )

        self.mod = cupy.RawModule(
            code=code_verify_3d,
            options=('--std=c++17',),
            name_expressions=name_expressions
        )

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    def test_mdspan_3d(self, dtype, order, index_type):
        """Test 3D arrays with mdspan."""
        shape = (4, 5, 6)
        a = testing.shaped_random(shape, dtype=dtype, order=order)
        a_mdspan = a.mdspan(index_type=index_type)
        out = cupy.zeros_like(a)
        out_mdspan = out.mdspan(index_type=index_type)

        dtype_str = get_typename(dtype)
        index_type_str = get_typename(index_type)

        ker = self.mod.get_function(
            f'verify_mdspan_3d<{dtype_str}, {index_type_str}>'
        )

        ker((1, 1, 1), (4, 5, 6), (a_mdspan, out_mdspan))
        testing.assert_array_equal(out, a + cupy.ones(shape, dtype=dtype))

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    def test_mdspan_3d_sliced(self, dtype, order, index_type):
        """Test 3D sliced arrays."""
        shape = (8, 10, 12)
        a = testing.shaped_random(shape, dtype=dtype, order=order)
        a_sliced = a[::2, ::2, ::3]
        a_mdspan = a_sliced.mdspan(index_type=index_type)
        out = cupy.zeros_like(a)
        out_sliced = out[::2, ::2, ::3]
        out_mdspan = out_sliced.mdspan(index_type=index_type)

        dtype_str = get_typename(dtype)
        index_type_str = get_typename(index_type)

        ker = self.mod.get_function(
            f'verify_mdspan_3d<{dtype_str}, {index_type_str}>'
        )

        ker((1, 1, 1), (4, 5, 4), (a_mdspan, out_mdspan))
        testing.assert_array_equal(
            out_sliced, a_sliced + cupy.ones_like(a_sliced, dtype=dtype)
        )


@pytest.mark.parametrize('index_type', MDSPAN_INDEX_TYPES)
@pytest.mark.skipif(
    cupy.cuda.runtime.is_hip, reason='libcudacxx not supported in HIP'
)
class TestMdspanValidation:
    """Test mdspan with compile-time validation."""

    def setup_class(self):
        dtypes = [cupy.dtype(c).type for c in ALL_TYPE_CHARS]

        name_expressions = []
        for dtype in dtypes:
            for index_type in MDSPAN_INDEX_TYPES:
                dtype_str = get_typename(dtype)
                index_str = get_typename(index_type)
                name_expressions.append(
                    f'verify_mdspan_with_size_check<{dtype_str}, {index_str}>'
                )

        self.mod = cupy.RawModule(
            code=code_verify_with_validation,
            options=('--std=c++17',),
            name_expressions=name_expressions
        )

    @testing.for_all_dtypes()
    def test_mdspan_size_validation(self, dtype, index_type):
        """Test that mdspan size matches expected layout."""
        shape = (4, 8)
        a = testing.shaped_random(shape, dtype=dtype)
        a_mdspan = a.mdspan(index_type=index_type)
        out = cupy.zeros_like(a)
        out_mdspan = out.mdspan(index_type=index_type)

        dtype_str = get_typename(dtype)
        index_type_str = get_typename(index_type)

        # This kernel has static_assert for size validation
        ker = self.mod.get_function(
            f'verify_mdspan_with_size_check<{dtype_str}, {index_type_str}>'
        )

        ker((1,), shape, (a_mdspan, out_mdspan))
        testing.assert_array_equal(out, a)

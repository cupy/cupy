from __future__ import annotations

import functools

import pytest

import cupy
from cupy import testing
from cupy._core._scalar import get_typename


# Supported index types for mdspan
MDSPAN_INDEX_TYPES = [cupy.int32, cupy.int64]


# Supported layout types for mdspan
MDSPAN_LAYOUT_TYPES = ['layout_stride', 'layout_right', 'layout_left']


# TODO(leofang): do we have a better source of all dtypes?
# All type chars from cupy/_core/_dtype.pyx
ALL_TYPE_CHARS = '?bhilqBHILQefdFD'


def make_kernel_code(kernel_name, ndim, layout='layout_stride'):
    """Generate mdspan kernel code with specified dimensionality and layout.

    Args:
        kernel_name: Name of the kernel function
        ndim: Number of dimensions (0, 1, 2, or 3)
        layout: Layout policy ('layout_stride', 'layout_right', 'layout_left')
    """
    # Common header
    code = r"""
#include <cuda/std/mdspan>
#include <cupy/carray.cuh>  // TODO(leofang): replace this by fp16 header once available
#include <cupy/complex.cuh>

template<typename T, typename IndexType>
__global__ void {kernel_name}(
    cuda::std::mdspan<
        T,
        cuda::std::extents<{extents}>,
        cuda::std::{layout}> arr_in,
    cuda::std::mdspan<
        T,
        cuda::std::extents<{extents}>,
        cuda::std::{layout}> arr_out
) {{
{body}
}}
"""  # noqa: E501

    # Build extents based on dimensionality
    dyn_ext = ["cuda::std::dynamic_extent"] * ndim
    extents = "IndexType," + ",".join(dyn_ext)
    if ndim == 0:
        body = r"""
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        arr_out() = arr_in() + T(1);
    }"""
    elif ndim == 1:
        body = r"""
    for (IndexType i = blockIdx.x * (IndexType)blockDim.x + threadIdx.x;
         i < arr_in.extent(0);
         i += gridDim.x * blockDim.x) {
        arr_out(i) = arr_in(i) + T(1);
    }"""
    elif ndim == 2:
        body = r"""
    for (IndexType i = blockIdx.x * (IndexType)blockDim.x + threadIdx.x;
         i < arr_in.extent(0);
         i += gridDim.x * blockDim.x) {
        for (IndexType j = blockIdx.y * (IndexType)blockDim.y + threadIdx.y;
             j < arr_in.extent(1);
             j += gridDim.y * blockDim.y) {
            arr_out(i, j) = arr_in(i, j) + T(1);
        }
    }"""
    elif ndim == 3:
        body = r"""
    for (IndexType i = blockIdx.x * (IndexType)blockDim.x + threadIdx.x;
         i < arr_in.extent(0);
         i += gridDim.x * blockDim.x) {
        for (IndexType j = blockIdx.y * (IndexType)blockDim.y + threadIdx.y;
             j < arr_in.extent(1);
             j += gridDim.y * blockDim.y) {
            for (IndexType k = blockIdx.z * (IndexType)blockDim.z + threadIdx.z;
                 k < arr_in.extent(2);
                 k += gridDim.z * blockDim.z) {
                arr_out(i, j, k) = arr_in(i, j, k) + T(1);
            }
        }
    }"""  # noqa: E501
    else:
        raise ValueError(f"Unsupported dimensionality: {ndim}")

    return code.format(
        kernel_name=kernel_name,
        extents=extents,
        layout=layout,
        body=body
    )


# Kernel with compile-time size validation
code_verify_with_validation = r"""
#include <cuda/std/mdspan>
#include <cupy/carray.cuh>  // TODO(leofang): replace this by fp16 header once available
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
"""  # noqa: E501


# Reduce overhead
mod_cache = {}


@pytest.fixture(scope='class', params=MDSPAN_INDEX_TYPES)
def index_type(request):
    return request.param


@pytest.fixture(scope='class', params=MDSPAN_LAYOUT_TYPES)
def layout(request):
    return request.param


@pytest.fixture(scope='class')
def make_kernel_module(layout, index_type):

    def _make_kernel_module(layout, index_type, ndim):
        mod = mod_cache.get((layout, index_type, ndim))
        if mod is not None:
            return mod, layout, index_type

        dtypes = [cupy.dtype(c).type for c in ALL_TYPE_CHARS]
        name_expressions = []
        for dtype in dtypes:
            dtype_str = get_typename(dtype)
            index_str = get_typename(index_type)
            name_expressions.append(
                f'verify_mdspan_{layout}_{ndim}d<{dtype_str}, {index_str}>'
            )
        mod = cupy.RawModule(
            code=make_kernel_code(
                f'verify_mdspan_{layout}_{ndim}d', ndim, layout
            ),
            options=('--std=c++17',),
            name_expressions=name_expressions
        )

        mod = mod_cache.setdefault((layout, index_type, ndim), mod)
        return mod, layout, index_type

    return functools.partial(_make_kernel_module, layout, index_type)


@pytest.mark.skipif(
    cupy.cuda.runtime.is_hip, reason='libcudacxx not supported in HIP'
)
class TestMdspan0D:
    """Test mdspan with 0D arrays (scalars)."""

    @testing.for_all_dtypes()
    def test_mdspan_0d(self, dtype, make_kernel_module):
        mod, layout, index_type = make_kernel_module(0)

        a = cupy.array(42, dtype=dtype)
        a_mdspan = a.mdspan(index_type=index_type)
        out = cupy.zeros_like(a)
        out_mdspan = out.mdspan(index_type=index_type)

        dtype_str = get_typename(dtype)
        index_type_str = get_typename(index_type)
        ker = mod.get_function(
            f'verify_mdspan_{layout}_0d<{dtype_str}, {index_type_str}>'
        )

        ker((1,), (1,), (a_mdspan, out_mdspan))
        testing.assert_array_equal(out, a + dtype(1))

    @testing.for_all_dtypes()
    def test_mdspan_0d_scalar_from_slice(self, dtype, make_kernel_module):
        mod, layout, index_type = make_kernel_module(0)

        # Create 0D array by slicing
        a = testing.shaped_random((5,), dtype=dtype)
        a_scalar = a[2]  # 0D array
        a_mdspan = a_scalar.mdspan(index_type=index_type)
        out = cupy.zeros_like(a_scalar)
        out_mdspan = out.mdspan(index_type=index_type)

        dtype_str = get_typename(dtype)
        index_type_str = get_typename(index_type)
        ker = mod.get_function(
            f'verify_mdspan_{layout}_0d<{dtype_str}, {index_type_str}>'
        )

        ker((1,), (1,), (a_mdspan, out_mdspan))
        testing.assert_array_equal(out, a_scalar + dtype(1))


@pytest.mark.skipif(
    cupy.cuda.runtime.is_hip, reason='libcudacxx not supported in HIP'
)
class TestMdspan1D:
    """Test mdspan with 1D arrays."""

    @testing.for_all_dtypes()
    def test_mdspan_1d(self, dtype, make_kernel_module):
        mod, layout, index_type = make_kernel_module(1)

        a = testing.shaped_random((100,), dtype=dtype)
        a_mdspan = a.mdspan(index_type=index_type)
        out = cupy.zeros_like(a)
        out_mdspan = out.mdspan(index_type=index_type)

        dtype_str = get_typename(dtype)
        index_type_str = get_typename(index_type)
        ker = mod.get_function(
            f'verify_mdspan_{layout}_1d<{dtype_str}, {index_type_str}>'
        )

        ker((1,), (100,), (a_mdspan, out_mdspan))
        testing.assert_array_equal(out, a + cupy.ones(100, dtype=dtype))

    @testing.for_all_dtypes()
    def test_mdspan_1d_sliced(self, dtype, make_kernel_module):
        mod, layout, index_type = make_kernel_module(1)
        if layout != 'layout_stride':
            pytest.skip("Sliced test only applicable for layout_stride")

        a = testing.shaped_random((100,), dtype=dtype)
        a_sliced = a[::3]  # Every 3rd element
        a_mdspan = a_sliced.mdspan(index_type=index_type)
        out = cupy.zeros_like(a)
        out_sliced = out[::3]
        out_mdspan = out_sliced.mdspan(index_type=index_type)

        dtype_str = get_typename(dtype)
        index_type_str = get_typename(index_type)
        ker = mod.get_function(
            f'verify_mdspan_{layout}_1d<{dtype_str}, {index_type_str}>'
        )

        ker((1,), (34,), (a_mdspan, out_mdspan))
        testing.assert_array_equal(
            out_sliced, a_sliced + cupy.ones_like(a_sliced, dtype=dtype)
        )

    @testing.for_all_dtypes()
    def test_mdspan_negative_stride_1d(self, dtype, make_kernel_module):
        mod, layout, index_type = make_kernel_module(1)
        if layout != 'layout_stride':
            pytest.skip(
                "Negative stride test only applicable for layout_stride")

        a = testing.shaped_random((100,), dtype=dtype)
        a_rev = a[::-1]  # Negative stride
        # Check if strides are actually negative
        assert a_rev.strides[0] < 0, "Expected negative stride"

        a_mdspan = a_rev.mdspan(index_type=index_type, allow_unsafe=True)
        out = cupy.zeros_like(a)
        out_rev = out[::-1]
        out_mdspan = out_rev.mdspan(index_type=index_type, allow_unsafe=True)

        dtype_str = get_typename(dtype)
        index_type_str = get_typename(index_type)
        ker = mod.get_function(
            f'verify_mdspan_{layout}_1d<{dtype_str}, {index_type_str}>'
        )

        # This tests if mdspan correctly handles negative strides
        ker((1,), (100,), (a_mdspan, out_mdspan))
        testing.assert_array_equal(out_rev, a_rev +
                                   cupy.ones(100, dtype=dtype))


@pytest.mark.skipif(
    cupy.cuda.runtime.is_hip, reason='libcudacxx not supported in HIP'
)
class TestMdspan2D:

    # TODO(leofang): it does not seem 'AK' makes any difference here?
    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    def test_mdspan_2d(self, dtype, order, make_kernel_module):
        mod, layout, index_type = make_kernel_module(2)
        shape = (4, 8)

        a = testing.shaped_random(shape, dtype=dtype, order=order)
        a_mdspan = a.mdspan(index_type=index_type)
        out = cupy.zeros_like(a)
        out_mdspan = out.mdspan(index_type=index_type)

        dtype_str = get_typename(dtype)
        index_type_str = get_typename(index_type)
        ker = mod.get_function(
            f'verify_mdspan_{layout}_2d<{dtype_str}, {index_type_str}>'
        )

        ker((1,), shape, (a_mdspan, out_mdspan))
        testing.assert_array_equal(out, a + cupy.ones(shape, dtype=dtype))

    # TODO(leofang): it does not seem 'AK' makes any difference here?
    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    def test_mdspan_2d_sliced(self, dtype, order, make_kernel_module):
        mod, layout, index_type = make_kernel_module(2)
        if layout != 'layout_stride':
            pytest.skip("Sliced test only applicable for layout_stride")

        shape = (4, 16)
        a = testing.shaped_random(shape, dtype=dtype, order=order)
        a_sliced = a[::2, ::3]
        a_mdspan = a_sliced.mdspan(index_type=index_type)
        out = cupy.zeros_like(a)
        out_sliced = out[::2, ::3]
        out_mdspan = out_sliced.mdspan(index_type=index_type)

        dtype_str = get_typename(dtype)
        index_type_str = get_typename(index_type)
        ker = mod.get_function(
            f'verify_mdspan_{layout}_2d<{dtype_str}, {index_type_str}>'
        )

        ker((1,), shape, (a_mdspan, out_mdspan))
        testing.assert_array_equal(
            out_sliced, a_sliced + cupy.ones_like(a_sliced, dtype=dtype)
        )

    @testing.for_all_dtypes()
    def test_mdspan_negative_stride_2d(self, dtype, make_kernel_module):
        mod, layout, index_type = make_kernel_module(2)
        if layout != 'layout_stride':
            pytest.skip(
                "Negative stride test only applicable for layout_stride")

        shape = (10, 20)
        a = testing.shaped_random(shape, dtype=dtype)
        a_rev = a[::-1, :]  # Negative stride in first dimension
        assert a_rev.strides[0] < 0, "Expected negative stride"

        a_mdspan = a_rev.mdspan(index_type=index_type, allow_unsafe=True)
        out = cupy.zeros_like(a)
        out_rev = out[::-1, :]
        out_mdspan = out_rev.mdspan(index_type=index_type, allow_unsafe=True)

        dtype_str = get_typename(dtype)
        index_type_str = get_typename(index_type)
        ker = mod.get_function(
            f'verify_mdspan_{layout}_2d<{dtype_str}, {index_type_str}>'
        )

        ker((1,), shape, (a_mdspan, out_mdspan))
        testing.assert_array_equal(
            out_rev, a_rev + cupy.ones_like(a_rev, dtype=dtype)
        )


@pytest.mark.skipif(
    cupy.cuda.runtime.is_hip, reason='libcudacxx not supported in HIP'
)
class TestMdspan3D:
    """Test mdspan with 3D arrays."""

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    def test_mdspan_3d(self, dtype, order, make_kernel_module):
        mod, layout, index_type = make_kernel_module(3)

        shape = (4, 5, 6)
        a = testing.shaped_random(shape, dtype=dtype, order=order)
        a_mdspan = a.mdspan(index_type=index_type)
        out = cupy.zeros_like(a)
        out_mdspan = out.mdspan(index_type=index_type)

        dtype_str = get_typename(dtype)
        index_type_str = get_typename(index_type)
        ker = mod.get_function(
            f'verify_mdspan_{layout}_3d<{dtype_str}, {index_type_str}>'
        )

        ker((1, 1, 1), (4, 5, 6), (a_mdspan, out_mdspan))
        testing.assert_array_equal(out, a + cupy.ones(shape, dtype=dtype))

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    def test_mdspan_3d_sliced(self, dtype, order, make_kernel_module):
        mod, layout, index_type = make_kernel_module(3)
        if layout != 'layout_stride':
            pytest.skip("Sliced test only applicable for layout_stride")

        shape = (8, 10, 12)
        a = testing.shaped_random(shape, dtype=dtype, order=order)
        a_sliced = a[::2, ::2, ::3]
        a_mdspan = a_sliced.mdspan(index_type=index_type)
        out = cupy.zeros_like(a)
        out_sliced = out[::2, ::2, ::3]
        out_mdspan = out_sliced.mdspan(index_type=index_type)

        dtype_str = get_typename(dtype)
        index_type_str = get_typename(index_type)
        ker = mod.get_function(
            f'verify_mdspan_{layout}_3d<{dtype_str}, {index_type_str}>'
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


@pytest.mark.skipif(
    cupy.cuda.runtime.is_hip, reason='libcudacxx not supported in HIP'
)
class TestMdspanBroadcast:
    """Test mdspan with broadcasted arrays (zero strides)."""

    @testing.for_all_dtypes()
    def test_mdspan_broadcast_1d_zero_stride(self, dtype, make_kernel_module):
        mod, layout, index_type = make_kernel_module(1)
        if layout != 'layout_stride':
            pytest.skip("Broadcast test only applicable for layout_stride")

        # Create broadcasted array with zero stride: [1] -> [1, 1, 1, ...]
        shape = (10,)
        a = cupy.array([42], dtype=dtype)
        a_broadcast = cupy.broadcast_to(a, shape)

        # Verify it has zero stride
        assert a_broadcast.strides[0] == 0

        a_mdspan = a_broadcast.mdspan(
            index_type=index_type, allow_unsafe=True)
        out = cupy.zeros(shape, dtype=dtype)
        out_mdspan = out.mdspan(index_type=index_type)

        dtype_str = get_typename(dtype)
        index_type_str = get_typename(index_type)
        ker = mod.get_function(
            f'verify_mdspan_{layout}_1d<{dtype_str}, {index_type_str}>'
        )

        ker((1,), shape, (a_mdspan, out_mdspan))
        # All elements should be a[0] + 1
        expected = cupy.full(shape, a[0] + dtype(1), dtype=dtype)
        testing.assert_array_equal(out, expected)

    @testing.for_all_dtypes()
    def test_mdspan_broadcast_2d_zero_stride_axis0(
            self, dtype, make_kernel_module):
        mod, layout, index_type = make_kernel_module(2)
        if layout != 'layout_stride':
            pytest.skip("Broadcast test only applicable for layout_stride")

        # Broadcast along axis 0: (1, 5) -> (3, 5) with stride[0] = 0
        shape = (3, 5)
        a = testing.shaped_random((1, 5), dtype=dtype)
        a_broadcast = cupy.broadcast_to(a, shape)

        # Verify zero stride in axis 0
        assert a_broadcast.strides[0] == 0

        a_mdspan = a_broadcast.mdspan(
            index_type=index_type, allow_unsafe=True)
        out = cupy.zeros(shape, dtype=dtype)
        out_mdspan = out.mdspan(index_type=index_type)

        dtype_str = get_typename(dtype)
        index_type_str = get_typename(index_type)
        ker = mod.get_function(
            f'verify_mdspan_{layout}_2d<{dtype_str}, {index_type_str}>'
        )

        ker((1,), shape, (a_mdspan, out_mdspan))
        expected = a_broadcast + cupy.ones(shape, dtype=dtype)
        testing.assert_array_equal(out, expected)

    @testing.for_all_dtypes()
    def test_mdspan_broadcast_2d_zero_stride_axis1(
            self, dtype, make_kernel_module):
        mod, layout, index_type = make_kernel_module(2)
        if layout != 'layout_stride':
            pytest.skip("Broadcast test only applicable for layout_stride")

        # Broadcast along axis 1: (4, 1) -> (4, 6) with stride[1] = 0
        shape = (4, 6)
        a = testing.shaped_random((4, 1), dtype=dtype)
        a_broadcast = cupy.broadcast_to(a, shape)
        # Verify zero stride in axis 1
        assert a_broadcast.strides[1] == 0

        a_mdspan = a_broadcast.mdspan(
            index_type=index_type, allow_unsafe=True)
        out = cupy.zeros(shape, dtype=dtype)
        out_mdspan = out.mdspan(index_type=index_type)

        dtype_str = get_typename(dtype)
        index_type_str = get_typename(index_type)
        ker = mod.get_function(
            f'verify_mdspan_{layout}_2d<{dtype_str}, {index_type_str}>'
        )

        ker((1,), shape, (a_mdspan, out_mdspan))
        expected = a_broadcast + cupy.ones(shape, dtype=dtype)
        testing.assert_array_equal(out, expected)

    @testing.for_all_dtypes()
    def test_mdspan_broadcast_2d_both_axes(
            self, dtype, make_kernel_module):
        mod, layout, index_type = make_kernel_module(2)
        if layout != 'layout_stride':
            pytest.skip("Broadcast test only applicable for layout_stride")

        # Broadcast from scalar: (1, 1) -> (3, 4) with both strides = 0
        shape = (3, 4)
        a = cupy.array([[42]], dtype=dtype)
        a_broadcast = cupy.broadcast_to(a, shape)

        # Verify both strides are zero
        assert a_broadcast.strides[0] == 0
        assert a_broadcast.strides[1] == 0

        a_mdspan = a_broadcast.mdspan(
            index_type=index_type, allow_unsafe=True)
        out = cupy.zeros(shape, dtype=dtype)
        out_mdspan = out.mdspan(index_type=index_type)

        dtype_str = get_typename(dtype)
        index_type_str = get_typename(index_type)
        ker = mod.get_function(
            f'verify_mdspan_{layout}_2d<{dtype_str}, {index_type_str}>'
        )

        ker((1,), shape, (a_mdspan, out_mdspan))
        expected = cupy.full(shape, a[0, 0] + dtype(1), dtype=dtype)
        testing.assert_array_equal(out, expected)


@pytest.mark.skipif(
    cupy.cuda.runtime.is_hip, reason='libcudacxx not supported in HIP'
)
class TestMdspanValidationUnsafe:
    """Test mdspan validation with allow_unsafe parameter."""

    def test_zero_size_dimension_rejected(self):
        """Test that zero-size dimensions raise RuntimeError by default."""
        # 1D case
        a = cupy.empty((0,), dtype=cupy.float32)
        with pytest.raises(RuntimeError, match="0-th dimension has size zero"):
            a.mdspan(index_type=cupy.int64)

        # 2D case - first dimension zero
        a = cupy.empty((0, 5), dtype=cupy.float32)
        with pytest.raises(RuntimeError, match="0-th dimension has size zero"):
            a.mdspan(index_type=cupy.int64)

        # 2D case - second dimension zero
        a = cupy.empty((5, 0), dtype=cupy.float32)
        with pytest.raises(RuntimeError, match="1-th dimension has size zero"):
            a.mdspan(index_type=cupy.int64)

    def test_zero_size_dimension_allowed_with_flag(self):
        """Test that zero-size dimensions work with allow_unsafe=True."""
        a = cupy.empty((0,), dtype=cupy.float32)
        mdspan = a.mdspan(index_type=cupy.int64, allow_unsafe=True)
        assert mdspan is not None

        a = cupy.empty((0, 5), dtype=cupy.float32)
        mdspan = a.mdspan(index_type=cupy.int64, allow_unsafe=True)
        assert mdspan is not None

        a = cupy.empty((5, 0), dtype=cupy.float32)
        mdspan = a.mdspan(index_type=cupy.int64, allow_unsafe=True)
        assert mdspan is not None

    def test_zero_stride_rejected(self):
        """Test that zero strides raise RuntimeError by default."""
        # 1D broadcast
        a = cupy.array([42], dtype=cupy.float32)
        a_broadcast = cupy.broadcast_to(a, (10,))
        assert a_broadcast.strides[0] == 0

        with pytest.raises(
                RuntimeError, match="0-th dimension has non-positive stride"):
            a_broadcast.mdspan(index_type=cupy.int64)

        # 2D broadcast along axis 0
        a = cupy.array([[1, 2, 3]], dtype=cupy.float32)
        a_broadcast = cupy.broadcast_to(a, (5, 3))
        assert a_broadcast.strides[0] == 0

        with pytest.raises(
                RuntimeError, match="0-th dimension has non-positive stride"):
            a_broadcast.mdspan(index_type=cupy.int64)

    def test_zero_stride_allowed_with_flag(self):
        """Test that zero strides work with allow_unsafe=True."""
        a = cupy.array([42], dtype=cupy.float32)
        a_broadcast = cupy.broadcast_to(a, (10,))

        mdspan = a_broadcast.mdspan(index_type=cupy.int64, allow_unsafe=True)
        assert mdspan is not None

        # 2D case
        a = cupy.array([[1, 2, 3]], dtype=cupy.float32)
        a_broadcast = cupy.broadcast_to(a, (5, 3))

        mdspan = a_broadcast.mdspan(index_type=cupy.int64, allow_unsafe=True)
        assert mdspan is not None

    def test_negative_stride_rejected(self):
        """Test that negative strides raise RuntimeError by default."""
        # 1D reversed
        a = cupy.arange(10, dtype=cupy.float32)
        a_rev = a[::-1]
        assert a_rev.strides[0] < 0

        with pytest.raises(
                RuntimeError, match="0-th dimension has non-positive stride"):
            a_rev.mdspan(index_type=cupy.int64)

        # 2D reversed first dimension
        a = cupy.arange(20, dtype=cupy.float32).reshape(4, 5)
        a_rev = a[::-1, :]
        assert a_rev.strides[0] < 0

        with pytest.raises(
                RuntimeError, match="0-th dimension has non-positive stride"):
            a_rev.mdspan(index_type=cupy.int64)

    def test_negative_stride_allowed_with_flag(self):
        """Test that negative strides work with allow_unsafe=True."""
        a = cupy.arange(10, dtype=cupy.float32)
        a_rev = a[::-1]

        mdspan = a_rev.mdspan(index_type=cupy.int64, allow_unsafe=True)
        assert mdspan is not None

        # 2D case
        a = cupy.arange(20, dtype=cupy.float32).reshape(4, 5)
        a_rev = a[::-1, :]

        mdspan = a_rev.mdspan(index_type=cupy.int64, allow_unsafe=True)
        assert mdspan is not None

    def test_validation_with_int32_index(self):
        """Test validation works with int32 index type."""
        a = cupy.empty((0,), dtype=cupy.float32)
        with pytest.raises(RuntimeError, match="has size zero"):
            a.mdspan(index_type=cupy.int32)

        # Should work with flag
        mdspan = a.mdspan(index_type=cupy.int32, allow_unsafe=True)
        assert mdspan is not None

        # Test zero stride
        a = cupy.broadcast_to(cupy.array([1]), (10,))
        with pytest.raises(RuntimeError, match="non-positive stride"):
            a.mdspan(index_type=cupy.int32)

        mdspan = a.mdspan(index_type=cupy.int32, allow_unsafe=True)
        assert mdspan is not None

    def test_validation_with_int64_index(self):
        """Test validation works with int64 index type."""
        a = cupy.empty((0,), dtype=cupy.float32)
        with pytest.raises(RuntimeError, match="has size zero"):
            a.mdspan(index_type=cupy.int64)

        # Should work with flag
        mdspan = a.mdspan(index_type=cupy.int64, allow_unsafe=True)
        assert mdspan is not None

        # Test negative stride
        a = cupy.arange(10)[::-1]
        with pytest.raises(RuntimeError, match="non-positive stride"):
            a.mdspan(index_type=cupy.int64)

        mdspan = a.mdspan(index_type=cupy.int64, allow_unsafe=True)
        assert mdspan is not None

    def test_multiple_zero_dimensions(self):
        """Test multiple zero-size dimensions."""
        a = cupy.empty((0, 0, 5), dtype=cupy.float32)
        # Should fail on first zero dimension
        with pytest.raises(RuntimeError, match="0-th dimension has size zero"):
            a.mdspan(index_type=cupy.int64)

        # Should work with flag
        mdspan = a.mdspan(index_type=cupy.int64, allow_unsafe=True)
        assert mdspan is not None

    def test_default_is_safe(self):
        """Verify that default behavior is safe (allow_unsafe=False)."""
        # Normal arrays should work without flag
        a = cupy.arange(10, dtype=cupy.float32)
        mdspan = a.mdspan(index_type=cupy.int64)  # No allow_unsafe needed
        assert mdspan is not None

        # 2D contiguous array
        a = cupy.arange(20, dtype=cupy.float32).reshape(4, 5)
        mdspan = a.mdspan(index_type=cupy.int64)
        assert mdspan is not None

        # But unsafe arrays should fail without flag
        a_broadcast = cupy.broadcast_to(cupy.array([1]), (10,))
        with pytest.raises(RuntimeError):
            # Should fail without allow_unsafe=True
            a_broadcast.mdspan(index_type=cupy.int64)


class TestMdspanInt32Validation:
    """Test validation of int32 index type against array dimensions/strides."""

    def test_int32_valid_dimensions(self):
        # Small array - should work fine
        a = cupy.arange(1000, dtype=cupy.float32)
        mdspan = a.mdspan(index_type=cupy.int32)
        assert mdspan is not None

        # Multi-dimensional small array
        a = cupy.arange(10000, dtype=cupy.float32).reshape(100, 100)
        mdspan = a.mdspan(index_type=cupy.int32)
        assert mdspan is not None

    @testing.slow
    def test_int32_dimension_overflow(self):
        a = cupy.empty(2**31, dtype=cupy.int8)
        with pytest.raises(ValueError, match="exceeds int32 maximum"):
            a.mdspan(index_type=cupy.int32)

        # Should work with int64
        mdspan = a.mdspan(index_type=cupy.int64)
        assert mdspan is not None

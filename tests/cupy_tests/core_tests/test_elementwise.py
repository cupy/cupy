from __future__ import annotations

import numpy
import pytest

import cupy
from cupy import _core
from cupy import cuda
from cupy import testing


class TestElementwise:

    def check_copy(self, dtype, src_id, dst_id):
        with cuda.Device(src_id):
            src = testing.shaped_arange((2, 3, 4), dtype=dtype)
        with cuda.Device(dst_id):
            dst = cupy.empty((2, 3, 4), dtype=dtype)
        _core.elementwise_copy(src, dst)
        testing.assert_allclose(src, dst)

    @testing.for_all_dtypes()
    def test_copy(self, dtype):
        device_id = cuda.Device().id
        self.check_copy(dtype, device_id, device_id)

    @testing.multi_gpu(2)
    @testing.for_all_dtypes()
    def test_copy_multigpu_nopeer(self, dtype):
        if cuda.runtime.deviceCanAccessPeer(0, 1) == 1:
            pytest.skip('peer access is available')
        with pytest.raises(ValueError):
            self.check_copy(dtype, 0, 1)

    @testing.multi_gpu(2)
    @testing.for_all_dtypes()
    def test_copy_multigpu_peer(self, dtype):
        if cuda.runtime.deviceCanAccessPeer(0, 1) != 1:
            pytest.skip('peer access is unavailable')
        with pytest.warns(cupy._util.PerformanceWarning):
            self.check_copy(dtype, 0, 1)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_copy_zero_sized_array1(self, xp, dtype, order):
        src = xp.empty((0,), dtype=dtype)
        res = xp.copy(src, order=order)
        assert src is not res
        return res

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_copy_zero_sized_array2(self, xp, dtype, order):
        src = xp.empty((1, 0, 2), dtype=dtype)
        res = xp.copy(src, order=order)
        assert src is not res
        return res

    @testing.for_orders('CFAK')
    def test_copy_orders(self, order):
        a = cupy.empty((2, 3, 4))
        b = cupy.copy(a, order)

        a_cpu = numpy.empty((2, 3, 4))
        b_cpu = numpy.copy(a_cpu, order)

        assert b.strides == b_cpu.strides


class TestElementwiseInvalidShape:

    def test_invalid_shape(self):
        with pytest.raises(ValueError, match='Out shape is mismatched'):
            f = cupy.ElementwiseKernel('T x', 'T y', 'y += x')
            x = cupy.arange(12).reshape(3, 4)
            y = cupy.arange(4)
            f(x, y)


class TestElementwiseInvalidArgument:

    def test_invalid_kernel_name(self):
        with pytest.raises(ValueError, match='Invalid kernel name'):
            cupy.ElementwiseKernel('T x', '', '', '1')


class TestElementwiseType:

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal(accept_error=OverflowError)
    def test_large_int_upper_1(self, xp, dtype):
        a = xp.array([0], dtype=xp.int8)
        b = xp.iinfo(dtype).max
        return a + b

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal(accept_error=OverflowError)
    def test_large_int_upper_2(self, xp, dtype):
        a = xp.array([1], dtype=xp.int8)
        b = xp.iinfo(dtype).max - 1
        return a + b

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_large_int_upper_3(self, xp, dtype):
        a = xp.array([xp.iinfo(dtype).max], dtype=dtype)
        b = xp.int8(0)
        return a + b

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_large_int_upper_4(self, xp, dtype):
        a = xp.array([xp.iinfo(dtype).max - 1], dtype=dtype)
        b = xp.int8(1)
        return a + b

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal(accept_error=OverflowError)
    def test_large_int_lower_1(self, xp, dtype):
        a = xp.array([0], dtype=xp.int8)
        b = xp.iinfo(dtype).min
        return a + b

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal(accept_error=OverflowError)
    def test_large_int_lower_2(self, xp, dtype):
        a = xp.array([-1], dtype=xp.int8)
        b = xp.iinfo(dtype).min + 1
        return a + b

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_large_int_lower_3(self, xp, dtype):
        a = xp.array([xp.iinfo(dtype).min], dtype=dtype)
        b = xp.int8(0)
        return a + b

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_large_int_lower_4(self, xp, dtype):
        a = xp.array([xp.iinfo(dtype).min + 1], dtype=dtype)
        b = xp.int8(-1)
        return a + b


def _make_test_kernel(
        in_shapes, out_shapes, no_return=False, return_tuple=False):
    # Creates a toy kernel for testing gufunc-like capabilities of
    # ElementwiseKernel. The core calculation takes the sum of the values
    # in a core slice for each argument, and multiples these sums together.
    # The output core slices are then filled with this product of sums
    # multiplied elementwise by an ndarray of the appropriate core shape
    # whose value at a given element is one more than the sum of indices
    # associated to that element. ``in_shapes`` and ``out_shapes`` contain
    # tuples containing info from the input and output parts of a gufunc
    # signature.
    in_params = [f'T{shape} in{i}' for i, shape in enumerate(in_shapes)]
    out_params = [f'T{shape} out{i}' for i, shape in enumerate(out_shapes)]

    in_params_str = ','.join(in_params)
    out_params_str = ','.join(out_params)

    operation = ''

    for i, shape in enumerate(in_shapes):
        if shape != '()':
            operation += f'auto in{i}_mdspan = in{i}.as_mdspan(); '

    for i, shape in enumerate(out_shapes):
        if shape != '()':
            operation += f'auto out{i}_mdspan = out{i}.as_mdspan(); '

    operation += 'T result = 1; '

    for i, shape in enumerate(in_shapes):
        operation += f'T sum{i} = 0; '

        if shape == '()':
            operation += f'sum{i} += in{i}; '
        else:
            rank = shape.count(',') + 1
            param = f'in{i}_mdspan'
            for dim in range(rank):
                operation += (
                    f'for (int i{dim} = 0; i{dim} < {param}.extent({dim});'
                    f' ++i{dim}) {{ '
                )
            indices = ', '.join(f'i{d}' for d in range(rank))
            operation += f'sum{i} += {param}({indices}); '
            for _ in range(rank):
                operation += '} '

        operation += f'result *= sum{i}; '

    for i, shape in enumerate(out_shapes):
        if shape == '()':
            operation += f'out{i} = result; '
        else:
            rank = shape.count(',') + 1
            param = f'out{i}_mdspan'

            for dim in range(rank):
                operation += (
                    f'for (int j{dim} = 0; j{dim} < {param}.extent({dim});'
                    f' ++j{dim}) {{ '
                )
            indices = ', '.join(f'j{d}' for d in range(rank))
            index_sum = ' + '.join(f'T(j{d})' for d in range(rank))
            operation += (
                f'{param}({indices}) = result * ({index_sum} + T(1)); ')

            for _ in range(rank):
                operation += '} '

    return cupy.ElementwiseKernel(
        in_params=in_params_str,
        out_params=out_params_str,
        operation=operation,
        options=("--std=c++17",),
        return_tuple=return_tuple,
        no_return=no_return,
    )


def _reference_func(*args, out_shape, in_core_ndims, out_core_ndim):
    # Compute reference values for the toy kernel described above.
    val = 1
    for arg, ndim in zip(args, in_core_ndims):
        if ndim == 0:
            val = val * arg
        else:
            axis = tuple(range(-ndim, 0))
            val = val * cupy.sum(arg, axis=axis)

    if not out_shape:
        return val

    expanded_val = val
    for _ in range(out_core_ndim):
        expanded_val = cupy.expand_dims(expanded_val, axis=-1)

    if out_core_ndim > 0:
        core_shape = out_shape[-out_core_ndim:]
        index_sum = sum(cupy.ogrid[tuple(slice(d) for d in core_shape)])
        return expanded_val * (index_sum + 1)

    return cupy.broadcast_to(expanded_val, out_shape)


class TestElementwiseGUFuncLike:
    def test_scalar(self):
        # '(),()->()'
        kern = _make_test_kernel(('()', '()'), ('()',))
        in0 = cupy.random.uniform(size=(1, 10))
        in1 = cupy.random.uniform(size=(10, 1))
        out_shape = (10, 10)
        actual = kern(in0, in1)
        desired = _reference_func(
            in0, in1, out_shape=out_shape, in_core_ndims=(0, 0),
            out_core_ndim=0)
        testing.assert_allclose(actual, desired)

    def test_reduction(self):
        # '(i)->()'
        kern = _make_test_kernel(('(i)',), ('()',))
        in0 = cupy.random.uniform(size=(30, 20, 100))
        out_shape = (30, 20)
        actual = kern(in0)
        desired = _reference_func(
            in0, out_shape=out_shape, in_core_ndims=(1,),
            out_core_ndim=0)
        testing.assert_allclose(actual, desired)

    def test_matmul_like(self):
        # '(m,n),(n,p)->(m,p)'
        kern = _make_test_kernel(('(m,n)', '(n,p)'), ('(m,p)',))
        in0 = cupy.random.uniform(size=(1, 10, 30, 20))
        in1 = cupy.random.uniform(size=(2, 10, 20, 50))
        out_shape = (2, 10, 30, 50)
        actual = kern(in0, in1)
        desired = _reference_func(
            in0, in1, out_shape=out_shape, in_core_ndims=(2, 2),
            out_core_ndim=2)
        testing.assert_allclose(actual, desired)

    def test_frozen_dims(self):
        # '(3),(3)->(3)'
        kern = _make_test_kernel(('(3)', '(3)'), ('(3)',))
        in0 = cupy.random.uniform(size=(100, 3))
        in1 = cupy.random.uniform(size=(100, 3))
        out_shape = (100, 3)
        actual = kern(in0, in1)
        desired = _reference_func(
            in0, in1, out_shape=out_shape, in_core_ndims=(1, 1),
            out_core_ndim=1)
        testing.assert_allclose(actual, desired)

    def test_pdist_like(self):
        # '(n, d)->(n * (n - 1) // 2)'
        kern = _make_test_kernel(('(n, d)',), ('(n * (n - 1) // 2)',))
        in0 = cupy.random.uniform(size=(100, 6, 10))
        out_shape = (100, 15)
        actual = kern(in0)
        desired = _reference_func(
            in0, out_shape=out_shape, in_core_ndims=(2,), out_core_ndim=1)
        testing.assert_allclose(actual, desired)

    def test_multiple_outputs(self):
        # '(m,n),(n,p)->(m**2+p**2,2*n**2),(m*n*n*p)'
        kern = _make_test_kernel(
            ('(m,n)', '(n,p)'), ('(m**2+p**2, 2*n**2)', '(m*n*n*p)',))
        in0 = cupy.random.uniform(size=(10, 1, 3, 2))
        in1 = cupy.random.uniform(size=(1, 10, 2, 4))
        out_shapes = ((10, 10, 25, 8), (10, 10, 48,))
        actual0, actual1 = kern(in0, in1)
        desired0, desired1 = (
            _reference_func(
                in0, in1, out_shape=out_shape, in_core_ndims=(2, 2),
                out_core_ndim=ndim)
            for out_shape, ndim in zip(out_shapes, (2, 1))
        )
        testing.assert_allclose(actual0, desired0)
        testing.assert_allclose(actual1, desired1)

    def test_with_preallocated_out(self):
        # '(m,n),(n,p)->(m**2+p**2,2*n**2),(m*n*n*p)'
        kern = _make_test_kernel(
            ('(m,n)', '(n,p)'), ('(m**2+p**2, 2*n**2)', '(m*n*n*p)',))
        in0 = cupy.random.uniform(size=(10, 1, 3, 2))
        in1 = cupy.random.uniform(size=(1, 10, 2, 4))
        out_shapes = ((10, 10, 25, 8), (10, 10, 48,))
        out0, out1 = (cupy.empty(shape) for shape in out_shapes)
        actual0, actual1 = kern(in0, in1, out0, out1)
        assert actual0 is out0
        assert actual1 is out1
        desired0, desired1 = (
            _reference_func(
                in0, in1, out_shape=out_shape, in_core_ndims=(2, 2),
                out_core_ndim=ndim)
            for out_shape, ndim in zip(out_shapes, (2, 1))
        )
        testing.assert_allclose(actual0, desired0)
        testing.assert_allclose(actual1, desired1)

    def test_return_tuple(self):
        # '(i)->()'
        kern = _make_test_kernel(('(i)',), ('()',), return_tuple=True)
        in0 = cupy.random.uniform(size=(30, 20, 100))
        out_shape = (30, 20)
        actual = kern(in0)
        assert isinstance(actual, tuple)
        assert len(actual) == 1
        desired = _reference_func(
            in0, out_shape=out_shape, in_core_ndims=(1,),
            out_core_ndim=0)
        testing.assert_allclose(actual[0], desired)

    def test_no_return(self):
        # '(n, d)->(n * (n - 1) // 2)'
        kern = _make_test_kernel(
            ('(n, d)',), ('(n * (n - 1) // 2)',), no_return=True)
        in0 = cupy.random.uniform(size=(100, 6, 10))
        out_shape = (100, 15)
        actual = cupy.empty(out_shape)
        result = kern(in0, actual)
        assert result is None
        desired = _reference_func(
            in0, out_shape=out_shape, in_core_ndims=(2,), out_core_ndim=1)
        testing.assert_allclose(actual, desired)

    def test_indeterminate_out_shape(self):
        # '(i)->(j)'
        kern = _make_test_kernel(('(i)',), ('(j)',))
        in0 = cupy.random.uniform(size=(100, 10))
        out_shape = (100, 20)
        actual = cupy.empty(out_shape)
        _ = kern(in0, actual)
        desired = _reference_func(
            in0, out_shape, in_core_ndims=(1,), out_core_ndim=1)
        testing.assert_allclose(actual, desired)

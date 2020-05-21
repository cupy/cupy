import threading
import unittest

import mock

import cupy
from cupy import testing
from cupy_tests.core_tests.fusion_tests import fusion_utils


class FusionTestBase(unittest.TestCase):
    def generate_inputs(self, xp, nargs, dtype):
        inputs = [
            testing.shaped_random((3, 4), xp, dtype, scale=10, seed=seed)
            for seed in range(nargs)
        ]
        return inputs, {}

    def dtype_combination(self, xp, dtype1, dtype2):
        x = testing.shaped_random((3, 4), xp, dtype1, scale=10, seed=0)
        y = testing.shaped_random((3, 4), xp, dtype2, scale=10, seed=1)
        return (x, y), {}


@testing.gpu
class TestFusionOutArg(unittest.TestCase):

    def generate_inputs(self, xp, dtype1, dtype2):
        x = testing.shaped_random((3, 4), xp, dtype1, scale=10, seed=0)
        y = testing.shaped_random((3, 4), xp, dtype1, scale=10, seed=1)
        z = testing.shaped_random((3, 4), xp, dtype2, scale=10, seed=2)
        return (x, y, z), {}

    @testing.for_all_dtypes_combination(
        names=('dtype1', 'dtype2'), full=True, no_complex=True)
    @fusion_utils.check_fusion(accept_error=TypeError)
    def test_outarg_add(self, xp, dtype1, dtype2):
        def func(x, y, z):
            xp.add(x, y, out=z)
            return z

        return func


@testing.gpu
class TestFusionInplaceUpdate(FusionTestBase):

    @testing.for_all_dtypes(no_bool=True)
    @fusion_utils.check_fusion(generate_inputs_args=(5,))
    def test_outarg_mixed(self, xp, dtype):
        def func(x, y, z, u, v):
            xp.add(x, y, out=z)
            xp.subtract(z, x, out=u)
            xp.multiply(z, x, out=v)
            return u

        return func

    @testing.for_all_dtypes(no_bool=True)
    @fusion_utils.check_fusion(generate_inputs_args=(2,))
    def test_iadd_multiple_times(self, xp, dtype):
        def func(x, y):
            x += y
            x += y
            x += y
            return x

        return func


@testing.gpu
class TestFusionTuple(FusionTestBase):

    @testing.for_all_dtypes(no_complex=True)
    @fusion_utils.check_fusion(generate_inputs_args=(3,))
    def test_tuple(self, xp, dtype):
        def func(x, y, z):
            w = x * y + z
            (x, w) = (w, x)
            return z * w + y + x

        return func

    @testing.for_all_dtypes(no_complex=True)
    @fusion_utils.check_fusion(generate_inputs_args=(3,))
    def test_return_tuple(self, xp, dtype):
        def func(x, y, z):
            return x + y, y + z, z * x

        return func

    @testing.for_all_dtypes(no_complex=True)
    @fusion_utils.check_fusion(generate_inputs_args=(3,))
    def test_multiple_outputdifferent_type_same_ufunc(self, xp, dtype):
        def func(x, y, z):
            x = x.astype('int32')
            y = x.astype('float32')
            return x + y, y + z, z + x

        return func

    @testing.for_all_dtypes(no_complex=True)
    @fusion_utils.check_fusion(generate_inputs_args=(1,))
    def test_return_empty_tuple(self, xp, dtype):
        def func(x):
            return ()

        return func

    @testing.for_all_dtypes(no_complex=True)
    @fusion_utils.check_fusion(generate_inputs_args=(1,))
    def test_return_singleton_tuple(self, xp, dtype):
        def func(x):
            return (x,)

        return func


@testing.gpu
class TestFusionReduction(FusionTestBase):

    @testing.for_all_dtypes(no_bool=True)
    @fusion_utils.check_fusion(generate_inputs_args=(1,))
    def test_reduction_premap(self, xp, dtype):
        def func(x):
            return xp.sum(xp.sqrt(x) + 10)

        return func

    @testing.for_all_dtypes(no_bool=True)
    @fusion_utils.check_fusion(generate_inputs_args=(1,))
    def test_reduction_postmap(self, xp, dtype):
        def func(x):
            return xp.sqrt(xp.sum(x)) + 10

        return func

    @testing.for_all_dtypes_combination(
        names=('dtype1', 'dtype2'), no_bool=True)
    @fusion_utils.check_fusion(generate_inputs_name='dtype_combination')
    def test_reduction_pairwise_premap(self, xp, dtype1, dtype2):
        def func(x, y):
            return xp.sum(xp.sqrt(xp.abs(x - y)))

        return func

    @testing.for_all_dtypes(no_bool=True)
    @fusion_utils.check_fusion(generate_inputs_args=(1,))
    def test_sum_axis_0(self, xp, dtype):
        def func(x):
            return xp.sum(x, axis=0)

        return func

    @testing.for_all_dtypes(no_bool=True)
    @fusion_utils.check_fusion(generate_inputs_args=(1,))
    def test_sum_axis_1(self, xp, dtype):
        def func(x):
            return xp.sum(x, axis=1)

        return func


@testing.gpu
class TestFusionDecorator(unittest.TestCase):
    def test_without_paren(self):
        @cupy.fuse
        def func_wo_paren(x):
            """Fuse without parentheses"""
            return x + x

        self.assertEqual(func_wo_paren.__name__, 'func_wo_paren')
        self.assertEqual(func_wo_paren.__doc__, 'Fuse without parentheses')

    def test_with_paren(self):
        @cupy.fuse()
        def func_w_paren(x):
            """Fuse with parentheses"""
            return x + x

        self.assertEqual(func_w_paren.__name__, 'func_w_paren')
        self.assertEqual(func_w_paren.__doc__, 'Fuse with parentheses')


@testing.gpu
class TestFusionKernelName(unittest.TestCase):

    def check(self, xp, func, expected_name, is_elementwise):
        a = xp.arange(0, 12, dtype='d').reshape(3, 4)
        b = xp.arange(5, 17, dtype='d').reshape(3, 4)
        c = xp.arange(13, 25, dtype='d').reshape(3, 4)

        # Test kernel name (with mock)
        if xp is cupy:
            target = (
                cupy.ElementwiseKernel if is_elementwise
                else cupy.ReductionKernel)
            target_full_name = '{}.{}'.format(
                target.__module__, target.__name__)

            with mock.patch(target_full_name) as kernel:
                func(a, b, c)
                kernel.assert_called_once()
                self.assertEqual(kernel.call_args[1]['name'], expected_name)

        # Test there's no error in computation (without mock)
        return func(a, b, c)

    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_elementwise(self, xp):
        def func(a, b, c):
            @cupy.fuse()
            def func_a1(x, y, z):
                return (x + y) * z

            return func_a1(a, b, c)

        return self.check(xp, func, 'func_a1', True)

    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_elementwise_with_name(self, xp):
        def func(a, b, c):
            @cupy.fuse(kernel_name='abc')
            def func_a1(x, y, z):
                return (x + y) * z

            return func_a1(a, b, c)

        return self.check(xp, func, 'abc', True)

    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_reduction_premap(self, xp):
        def func(a, b, c):
            @cupy.fuse()
            def func_a1(x, y, z):
                return xp.sum((x + y) * z)

            return func_a1(a, b, c)

        return self.check(xp, func, 'func_a1', False)

    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_reduction_postmap(self, xp):
        def func(a, b, c):
            @cupy.fuse()
            def func_a1(x):
                return xp.sqrt(xp.sum(x) + 10)

            return func_a1(a)

        return self.check(xp, func, 'func_a1', False)

    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_reduction_01(self, xp):
        def func(a, b, c):
            @cupy.fuse()
            def func_a1(x, y, z):
                return xp.sqrt(xp.prod(x + y * z, axis=1) + 10)

            return func_a1(a, b, c)

        return self.check(xp, func, 'func_a1', False)

    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_reduction_with_name(self, xp):
        def func(a, b, c):
            @cupy.fuse(kernel_name='abc')
            def func_a1(x, y, z):
                return xp.sum((x + y) * z)

            return func_a1(a, b, c)

        return self.check(xp, func, 'abc', False)


@testing.gpu
class TestFusionScalar(FusionTestBase):

    def generate_inputs(self, xp, dtype1, dtype2):
        array = testing.shaped_random((3, 4), xp, dtype1, scale=10, seed=0)
        return (array,), {}

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @fusion_utils.check_fusion()
    def test_python_scalar(self, xp, dtype1, dtype2):
        def func(array):
            py_scalar = dtype2(1).item()
            return array + py_scalar

        return func

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @fusion_utils.check_fusion()
    def test_numpy_scalar(self, xp, dtype1, dtype2):
        def func(array):
            np_scalar = dtype2(1)
            return array + np_scalar

        return func

    def python_scalar_param(self, xp, dtype1, dtype2):
        array = testing.shaped_random((3, 4), xp, dtype1, scale=10, seed=0)
        py_scalar = dtype2(1).item()
        return (array, py_scalar), {}

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @fusion_utils.check_fusion(generate_inputs_name='python_scalar_param')
    def test_python_scalar_param(self, xp, dtype1, dtype2):
        def func(array, py_scalar):
            return array + py_scalar

        return func

    def numpy_scalar_param(self, xp, dtype1, dtype2):
        array = testing.shaped_random((3, 4), xp, dtype1, scale=10, seed=0)
        np_scalar = dtype2(1)
        return (array, np_scalar), {}

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @fusion_utils.check_fusion(generate_inputs_name='numpy_scalar_param')
    def test_numpy_scalar_param(self, xp, dtype1, dtype2):
        def func(array, np_scalar):
            return array + np_scalar

        return func

    def numpy_scalar_params_binop(self, xp, dtype1, dtype2):
        scalar1 = dtype1(1)
        scalar2 = dtype2(1)
        array = testing.shaped_random((3, 4), xp, 'int64', scale=10, seed=0)
        return (scalar1, scalar2, array), {}

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @fusion_utils.check_fusion(
        generate_inputs_name='numpy_scalar_params_binop')
    def test_numpy_scalar_params_binop(self, xp, dtype1, dtype2):
        def func(scalar1, scalar2, array):
            dtype = (scalar1 + scalar2).dtype
            return array.astype(dtype)

        return func

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @fusion_utils.check_fusion(
        generate_inputs_name='numpy_scalar_params_binop')
    def test_scalar_inplace_update(self, xp, dtype1, dtype2):
        def func(scalar1, scalar2, array):
            scalar1_copy = scalar1
            scalar1 += scalar2
            return array + scalar1 + scalar1_copy

        return func

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @fusion_utils.check_fusion(generate_inputs_name='numpy_scalar_param')
    def test_scalar_inplace_update_with_array(self, xp, dtype1, dtype2):
        def func(array, scalar):
            scalar += array
            return scalar

        return func


@testing.gpu
class TestFusionBroadcast(unittest.TestCase):

    def generate_inputs(self, xp, dtype):
        x = testing.shaped_random((3, 4), xp, dtype, scale=10, seed=0)
        y = testing.shaped_random((4,), xp, dtype, scale=10, seed=1)
        return (x, y), {}

    @testing.for_all_dtypes()
    @fusion_utils.check_fusion()
    def test_broadcast(self, xp, dtype):
        def func(x, y):
            x += y
            return x

        return func

    @testing.for_all_dtypes()
    @fusion_utils.check_fusion(accept_error=ValueError)
    def test_broadcast_error(self, xp, dtype):
        def func(x, y):
            y += x
            return y

        return func


@testing.gpu
class TestFusionNoneParams(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_python_none_parameter(self, xp, dtype):
        @cupy.fuse()
        def f(x, y, z):
            if y is None:
                return x * z
            return x + y + z

        x = testing.shaped_arange((10,), xp, dtype)
        y = testing.shaped_arange((10,), xp, dtype)
        z = testing.shaped_arange((10,), xp, dtype)
        return f(x, None, z) + f(x, y, z)


@testing.gpu
class TestFusionReturnsConstantValue(FusionTestBase):

    @testing.for_all_dtypes()
    @fusion_utils.check_fusion(generate_inputs_args=(1,))
    def test_pass(self, xp, dtype):
        def func(x):
            pass

        return func

    @testing.for_all_dtypes(no_bool=True)
    @fusion_utils.check_fusion(generate_inputs_args=(1,))
    def test_no_return_value(self, xp, dtype):
        def func(x):
            x += 1

        return func


class TestFusionComposition(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_composition(self, xp, dtype):
        @cupy.fuse()
        def f(x, y):
            return x - y * 2, x + y

        @cupy.fuse()
        def g(x, y, z):
            a, b = f(x + z, z - x * 3)
            c, d = f(x - y, y - z)
            return a + b * c - d

        @cupy.fuse()
        def h(x, y):
            a, b = f(x + y * 2, y * 3)
            return a - b * g(x - 2, x - 3, -y)

        x = testing.shaped_arange((3, 3), xp, dtype)
        y = testing.shaped_arange((3, 3), xp, dtype)
        return h(x, y)


class TestFusionCompile(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_clear_cache(self, xp, dtype):
        @cupy.fuse()
        def f(x, y):
            return x - y * 2

        x = testing.shaped_arange((3, 3), xp, dtype)
        y = testing.shaped_arange((3, 3), xp, dtype)
        f.clear_cache()
        return f(x, y)


@testing.gpu
class TestFusionGetArrayModule(FusionTestBase):

    @testing.for_all_dtypes()
    @fusion_utils.check_fusion(generate_inputs_args=(1,))
    def test_get_array_module(self, xp, dtype):
        def func(x):
            assert xp == cupy.get_array_module(x)
            return x

        return func


class TestFusionThread(unittest.TestCase):

    def test_thread(self):
        x = testing.shaped_arange((3, 3), cupy, cupy.int64)
        y = testing.shaped_arange((3, 3), cupy, cupy.int64)
        out = [None]

        @cupy.fuse()
        def f(x, y):
            return x + y * 2

        def _target(x, y):
            cupy.cuda.Device(0).use()
            out[0] = f(x, y)

        t = threading.Thread(target=_target, args=(x, y))
        t.daemon = True
        t.start()
        t.join()
        assert (out[0] == f(x, y)).all()

    @testing.numpy_cupy_array_equal()
    def test_thread_multiple_dtypes(self, xp):
        x1 = testing.shaped_arange((3, 3), xp, xp.int64)
        y1 = testing.shaped_arange((3, 3), xp, xp.int64)
        x2 = x1.astype(xp.float64)
        y2 = y1.astype(xp.float64)
        threads = [None] * 100
        out = [None] * 100

        @cupy.fuse()
        def f(x, y):
            return x + y * 2

        def _target(tid, x, y):
            if xp is cupy:
                xp.cuda.Device(0).use()
            out[tid] = f(x, y).astype(xp.int64)

        def run_thread(tid):
            x, y = (x1, y1) if tid % 2 == 0 else (x2, y2)
            t = threading.Thread(target=_target, args=(tid, x, y))
            threads[tid] = t
            t.daemon = True
            t.start()

        for tid in range(0, 50):
            run_thread(tid)

        for tid in range(0, 50):
            threads[tid].join()

        for tid in range(50, 100):
            run_thread(tid)

        for tid in range(50, 100):
            threads[tid].join()

        return xp.concatenate(out)

import os
import tempfile
import unittest
from unittest import mock

import pytest

import cupy
from cupy import testing
from cupy._core import _accelerator


try:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        import cupyx.optimizing
        import cupyx.optimizing._optimize
        import cupy._core._optimize_config
except ImportError:
    pass


@testing.gpu
@testing.with_requires('optuna')
class TestOptimize(unittest.TestCase):

    def setUp(self):
        cupy._core._optimize_config._clear_all_contexts_cache()

    def test_optimize_reduction_kernel(self):
        my_sum = cupy.ReductionKernel(
            'T x', 'T out', 'x', 'a + b', 'out = a', '0', 'my_sum')
        x = testing.shaped_arange((3, 4), cupy)
        y1 = my_sum(x, axis=1)
        with cupyx.optimizing.optimize():
            y2 = my_sum(x, axis=1)
        testing.assert_array_equal(y1, y2)

    def test_optimize_cache(self):
        target = cupyx.optimizing._optimize._optimize
        target_full_name = '{}.{}'.format(target.__module__, target.__name__)

        with mock.patch(target_full_name) as optimize_impl:
            my_sum = cupy.ReductionKernel(
                'T x', 'T out', 'x', 'a + b', 'out = a', '0', 'my_sum')
            my_sum_ = cupy.ReductionKernel(
                'T x', 'T out', 'x', 'a + b', 'out = a', '0', 'my_sum_')
            x = testing.shaped_arange((3, 4), cupy)
            x_ = testing.shaped_arange((3, 4), cupy)
            y = testing.shaped_arange((4, 4), cupy)
            z = testing.shaped_arange((3, 4), cupy)[::-1]
            assert x.strides == y.strides
            assert x.shape == z.shape

            with cupyx.optimizing.optimize():
                my_sum(x, axis=1)
                assert optimize_impl.call_count == 1
                my_sum(x, axis=1)
                assert optimize_impl.call_count == 1
                my_sum(x, axis=0)
                assert optimize_impl.call_count == 2
                my_sum(x_, axis=1)
                assert optimize_impl.call_count == 2
                my_sum(y, axis=1)
                assert optimize_impl.call_count == 3
                my_sum(z, axis=1)
                assert optimize_impl.call_count == 4
                my_sum_(x, axis=1)
                assert optimize_impl.call_count == 5

            with cupyx.optimizing.optimize(key='new_key'):
                my_sum(x, axis=1)
                assert optimize_impl.call_count == 6

            with cupyx.optimizing.optimize(key=None):
                my_sum(x, axis=1)
                assert optimize_impl.call_count == 6
                my_sum(x)
                assert optimize_impl.call_count == 7

    @testing.multi_gpu(2)
    def test_optimize_cache_multi_gpus(self):
        target = cupyx.optimizing._optimize._optimize
        target_full_name = '{}.{}'.format(target.__module__, target.__name__)

        with mock.patch(target_full_name) as optimize_impl:
            my_sum = cupy.ReductionKernel(
                'T x', 'T out', 'x', 'a + b', 'out = a', '0', 'my_sum')

            with cupyx.optimizing.optimize():
                with cupy.cuda.Device(0):
                    x = testing.shaped_arange((3, 4), cupy)
                    my_sum(x, axis=1)
                    assert optimize_impl.call_count == 1

                with cupy.cuda.Device(1):
                    x = testing.shaped_arange((3, 4), cupy)
                    my_sum(x, axis=1)
                    assert optimize_impl.call_count == 2

    def test_optimize_pickle(self):
        my_sum = cupy.ReductionKernel(
            'T x', 'T out', 'x', 'a + b', 'out = a', '0', 'my_sum')
        x = testing.shaped_arange((3, 4), cupy)

        with tempfile.TemporaryDirectory() as directory:
            filepath = directory + '/optimize_params'

            with cupyx.optimizing.optimize() as context:
                my_sum(x, axis=1)
                params_map = context._params_map
                context.save(filepath)

            cupy._core._optimize_config._clear_all_contexts_cache()

            with cupyx.optimizing.optimize() as context:
                assert params_map.keys() != context._params_map.keys()
                context.load(filepath)
                assert params_map.keys() == context._params_map.keys()

            with cupyx.optimizing.optimize(key='other_key') as context:
                with pytest.raises(ValueError):
                    context.load(filepath)

    def test_optimize_autosave(self):
        with tempfile.TemporaryDirectory() as directory:
            filepath = directory + '/optimize_params'

            # non-existing file, readonly=True
            with testing.assert_warns(UserWarning):
                with cupyx.optimizing.optimize(path=filepath, readonly=True):
                    cupy.sum(cupy.arange(2))

            # non-existing file, readonly=False
            with cupyx.optimizing.optimize(path=filepath, readonly=False):
                cupy.sum(cupy.arange(4))
            filesize = os.stat(filepath).st_size
            assert 0 < filesize

            # existing file, readonly=True
            with cupyx.optimizing.optimize(path=filepath, readonly=True):
                cupy.sum(cupy.arange(6))
            assert filesize == os.stat(filepath).st_size

            # existing file, readonly=False
            with cupyx.optimizing.optimize(path=filepath, readonly=False):
                cupy.sum(cupy.arange(8))
            assert filesize < os.stat(filepath).st_size


# TODO(leofang): check the optimizer is not applicable to the cutensor backend?
@testing.parameterize(*testing.product({
    'backend': ([], ['cub'])
}))
@testing.gpu
@testing.with_requires('optuna')
class TestOptimizeBackends(unittest.TestCase):
    """This class tests if optuna is in effect for create_reduction_func()"""

    def setUp(self):
        cupy._core._optimize_config._clear_all_contexts_cache()
        self.old_reductions = _accelerator.get_reduction_accelerators()
        _accelerator.set_reduction_accelerators(self.backend)

        # avoid shadowed by the cub module
        self.old_routines = _accelerator.get_routine_accelerators()
        _accelerator.set_routine_accelerators([])

        self.x = testing.shaped_arange((3, 4), cupy, dtype=cupy.float32)

    def tearDown(self):
        _accelerator.set_routine_accelerators(self.old_routines)
        _accelerator.set_reduction_accelerators(self.old_reductions)

    def test_optimize1(self):
        # Ensure the optimizer is run 3 times for all backends.
        func = 'cupyx.optimizing._optimize._optimize'
        times_called = 3

        # Setting "wraps" is necessary to avoid compilation errors.
        with testing.AssertFunctionIsCalled(
                func, times_called=times_called,
                wraps=cupyx.optimizing._optimize._optimize):
            with cupyx.optimizing.optimize():
                self.x.sum()
            with cupyx.optimizing.optimize():
                self.x.sum(axis=1)
            with cupyx.optimizing.optimize():
                self.x.sum(axis=0)  # CUB falls back to the simple reduction

    def test_optimize2(self):
        # Ensure the CUB optimizer is not run when the CUB kernel is not used.
        func = 'cupy._core._cub_reduction._get_cub_optimized_params'
        times_called = 2 if ('cub' in self.backend) else 0

        # Setting "wraps" is necessary to avoid errors being silently ignored.
        with testing.AssertFunctionIsCalled(
                func, times_called=times_called,
                wraps=cupy._core._cub_reduction._get_cub_optimized_params):
            with cupyx.optimizing.optimize():
                self.x.sum()
            with cupyx.optimizing.optimize():
                self.x.sum(axis=1)
            with cupyx.optimizing.optimize():
                self.x.sum(axis=0)  # CUB optimizer not used

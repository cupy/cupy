import tempfile
import unittest
from unittest import mock

import pytest

import cupy
from cupy import testing


try:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        import cupyx.optimizing
        import cupyx.optimizing._optimize
        import cupy.core._optimize_config
except ImportError:
    pass


@testing.gpu
@testing.with_requires('optuna')
class TestOptimize(unittest.TestCase):

    def setUp(self):
        cupy.core._optimize_config._clear_all_contexts_cache()

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

            cupy.core._optimize_config._clear_all_contexts_cache()

            with cupyx.optimizing.optimize() as context:
                assert params_map.keys() != context._params_map.keys()
                context.load(filepath)
                assert params_map.keys() == context._params_map.keys()

            with cupyx.optimizing.optimize(key='other_key') as context:
                with pytest.raises(ValueError):
                    context.load(filepath)

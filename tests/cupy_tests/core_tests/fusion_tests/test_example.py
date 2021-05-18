import unittest
import pytest

import cupy
from cupy import testing
from cupy_tests.core_tests.fusion_tests import fusion_utils


@testing.gpu
@testing.slow
@pytest.mark.skipif(
    cupy.cuda.runtime.is_hip, reason='HIP does not support this')
class TestFusionExample(unittest.TestCase):
    def generate_inputs(self, xp):
        shape = (8, 64, 112, 112)
        _, chan, _, _ = shape
        x = testing.shaped_random(shape, xp, 'float32', scale=10, seed=0)
        gamma = xp.ones(chan)
        beta = xp.zeros(chan)
        running_mean = xp.zeros(chan)
        running_var = xp.ones(chan)
        size = x.size // gamma.size
        adjust = size / max(size - 1., 1.)
        return (x, gamma, beta, running_mean, running_var, size, adjust), {}

    @fusion_utils.check_fusion()
    def test_batchnorm(self, xp):
        def batchnorm(x, gamma, beta, running_mean, running_var, size, adjust):
            decay = 0.9
            eps = 2e-5
            expander = (None, slice(None), None, None)

            gamma = gamma[expander]
            beta = beta[expander]
            mean = xp.sum(x, axis=(0, 2, 3)) / size
            diff = x - mean[expander]
            var = xp.sum(diff * diff, axis=(0, 2, 3)) / size
            inv_std = 1. / xp.sqrt(var + eps)
            y = gamma * diff * inv_std[expander] + beta

            running_mean *= decay
            running_mean += (1 - decay) * mean
            running_var *= decay
            running_var += (1 - decay) * adjust * var
            return y

        return batchnorm

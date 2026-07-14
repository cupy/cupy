from __future__ import annotations

import numpy
import pytest

import cupy
from cupy import cuda
from cupy import testing


# Factories that build a brand-new array, keyed by name. Each takes the
# ``device`` argument under test.
_FACTORIES = {
    'empty': lambda device: cupy.empty((2, 3), device=device),
    'zeros': lambda device: cupy.zeros((2, 3), device=device),
    'ones': lambda device: cupy.ones((2, 3), device=device),
    'full': lambda device: cupy.full((2, 3), 7, device=device),
    'eye': lambda device: cupy.eye(3, device=device),
    'identity': lambda device: cupy.identity(3, device=device),
    'arange': lambda device: cupy.arange(6, device=device),
    'linspace': lambda device: cupy.linspace(0, 1, 5, device=device),
    'array': lambda device: cupy.array([1, 2, 3], device=device),
    'asarray': lambda device: cupy.asarray([1, 2, 3], device=device),
    'asanyarray': lambda device: cupy.asanyarray([1, 2, 3], device=device),
}

_LIKE_FACTORIES = {
    'empty_like': lambda a, device: cupy.empty_like(a, device=device),
    'zeros_like': lambda a, device: cupy.zeros_like(a, device=device),
    'ones_like': lambda a, device: cupy.ones_like(a, device=device),
    'full_like': lambda a, device: cupy.full_like(a, 7, device=device),
}


class TestDeviceArgument:

    @pytest.mark.parametrize('name', list(_FACTORIES))
    @testing.multi_gpu(2)
    def test_device_int(self, name):
        cuda.runtime.setDevice(0)
        arr = _FACTORIES[name](1)
        assert arr.device.id == 1
        # The current device is restored after the call.
        assert cuda.runtime.getDevice() == 0

    @pytest.mark.parametrize('name', list(_FACTORIES))
    @testing.multi_gpu(2)
    def test_device_object(self, name):
        cuda.runtime.setDevice(0)
        arr = _FACTORIES[name](cuda.Device(1))
        assert arr.device.id == 1
        assert cuda.runtime.getDevice() == 0

    @pytest.mark.parametrize('name', list(_FACTORIES))
    @testing.multi_gpu(2)
    def test_device_overrides_context(self, name):
        # An explicit ``device=`` takes precedence over an active context.
        with cuda.Device(1):
            arr = _FACTORIES[name](0)
            assert arr.device.id == 0
            assert cuda.runtime.getDevice() == 1

    @pytest.mark.parametrize('name', list(_FACTORIES))
    def test_device_none_uses_current(self, name):
        arr = _FACTORIES[name](None)
        assert arr.device.id == cuda.runtime.getDevice()

    @pytest.mark.parametrize('name', list(_LIKE_FACTORIES))
    @testing.multi_gpu(2)
    def test_like_device_int(self, name):
        cuda.runtime.setDevice(0)
        a = cupy.arange(6).reshape(2, 3)
        arr = _LIKE_FACTORIES[name](a, 1)
        assert arr.device.id == 1
        assert cuda.runtime.getDevice() == 0

    @testing.multi_gpu(2)
    def test_values_on_target_device(self):
        cuda.runtime.setDevice(0)
        o = cupy.ones((3,), device=1)
        f = cupy.full((3,), 7, device=1)
        e = cupy.eye(2, device=1)
        a = cupy.arange(4, device=1)
        ls = cupy.linspace(0, 1, 5, device=1)
        with cuda.Device(1):
            numpy.testing.assert_array_equal(cupy.asnumpy(o), numpy.ones(3))
            numpy.testing.assert_array_equal(cupy.asnumpy(f), numpy.full(3, 7))
            numpy.testing.assert_array_equal(cupy.asnumpy(e), numpy.eye(2))
            numpy.testing.assert_array_equal(cupy.asnumpy(a), numpy.arange(4))
            numpy.testing.assert_allclose(
                cupy.asnumpy(ls), numpy.linspace(0, 1, 5))

    @testing.multi_gpu(2)
    def test_asarray_cross_device(self):
        with cuda.Device(0):
            src = cupy.arange(4)
        dst = cupy.asarray(src, device=1)
        assert dst.device.id == 1
        assert src.device.id == 0
        with cuda.Device(1):
            numpy.testing.assert_array_equal(
                cupy.asnumpy(dst), numpy.arange(4))

    @pytest.mark.parametrize('bad', ['cpu', 'cuda:0', 1.0, True])
    def test_invalid_device_type(self, bad):
        with pytest.raises(TypeError):
            cupy.zeros(3, device=bad)

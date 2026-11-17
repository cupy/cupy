from __future__ import annotations
import pytest

import cupy as cp
from cupy.cuda import runtime


pytestmark = pytest.mark.skipif(
    not cp.is_available() or not runtime.is_hip,
    reason='HIP runtime required',
)


def test_cuda_alias_enums_defined():
    names = [
        'cudaDevAttrConcurrentKernels',
        'cudaDevAttrMaxBlockDimX',
        'cudaDevAttrWarpSize',
        'cudaDevAttrMaxTexture2DWidth',
        'cudaDevAttrMaxTexture3DDepth',
    ]
    for n in names:
        assert hasattr(runtime, n), f'missing alias: {n}'
        val = getattr(runtime, n)
        assert isinstance(
            val, int), f'alias {n} should be an int, got {type(val)}'


HIP_VERSION = runtime.runtimeGetVersion()


@pytest.mark.skipif(HIP_VERSION < 40300000, reason='Requires HIP >= 4.3.0')
def test_hip_device_attribute_enums_defined_in_backend():
    # Validate hipDeviceAttribute* constants exist at the backend module
    import cupy_backends.cuda.api._runtime_enum as e
    names = [
        'hipDeviceAttributeCudaCompatibleBegin',
        'hipDeviceAttributeAmdSpecificBegin',
        'hipDeviceAttributeEccEnabled',
        'hipDeviceAttributeMaxTexture3DWidth',
    ]
    for n in names:
        assert hasattr(e, n), f'missing HIP enum: {n}'
        val = getattr(e, n)
        assert isinstance(
            val, int), f'HIP enum {n} should be an int, got {type(val)}'


def test_device_attributes_match_device_properties():
    # The enum values must match the HIP header, so an attribute query
    # returns the same number as the device properties struct.
    props = runtime.getDeviceProperties(0)
    pairs = [
        ('cudaDevAttrMultiProcessorCount', 'multiProcessorCount'),
        ('cudaDevAttrMaxThreadsPerBlock', 'maxThreadsPerBlock'),
        ('cudaDevAttrWarpSize', 'warpSize'),
        ('cudaDevAttrL2CacheSize', 'l2CacheSize'),
        ('cudaDevAttrMaxRegistersPerBlock', 'regsPerBlock'),
    ]
    for attr, prop in pairs:
        got = runtime.deviceGetAttribute(getattr(runtime, attr), 0)
        assert got == props[prop], f'{attr}={got}, {prop}={props[prop]}'

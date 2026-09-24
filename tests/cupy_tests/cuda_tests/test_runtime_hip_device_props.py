from __future__ import annotations

import pytest

from cupy.cuda import runtime


@pytest.mark.skipif(not runtime.is_hip, reason="HIP backend required")
@pytest.mark.skipif(runtime.runtimeGetVersion() < 60100000,
                    reason="Test requires HIP runtime >= 6.1.0")
def test_hip_device_props_r0600_keys_present():
    props = runtime.getDeviceProperties(0)

    # Minimal presence checks for fields added in HIP >= 6.1 (R0600)
    expected_keys = [
        'uuid',
        'luidDeviceNodeMask',
        'sparseHipArraySupported',
        'gpuDirectRDMASupported',
        'memoryPoolsSupported',
        'deferredMappingHipArraySupported',
        'unifiedFunctionPointers',
    ]
    missing = [k for k in expected_keys if k not in props]
    assert not missing, f"Missing expected device properties: {missing}"

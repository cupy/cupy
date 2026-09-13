"""pytest fixtures for tests/ascend.

These tests must pass on a machine WITHOUT an NPU (verification ceiling L3):
device-dependent cases are skipped via the ``has_npu`` fixture.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def has_npu() -> bool:
    """True only when a real Ascend device can allocate memory."""
    try:
        import cupy
        cupy.zeros(1)
        return True
    except Exception:
        return False


class StubArray:
    """Duck-typed stand-in for a cupy ndarray (no device needed).

    Mirrors the attributes :class:`cupy.backends.ascend.triton_bridge.
    CuPyTensorAdapter` consumes: ``data.ptr``, ``shape``, ``dtype``,
    ``itemsize`` and byte-unit ``_strides``.
    """

    def __init__(self, ptr=0x1234, shape=(4,), byte_strides=(4,), dtype='f',
                 itemsize=4, size=None):
        self.data = type('Mem', (), {'ptr': ptr})()
        self.shape = shape
        self._strides = byte_strides
        self.dtype = type('Dtype', (), {'char': dtype})()
        self.itemsize = itemsize
        self.size = size if size is not None else shape[0]


@pytest.fixture
def stub_array():
    return StubArray

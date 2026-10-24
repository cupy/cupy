"""Tests for the triton-ascend bridge (cupy/backends/ascend/triton_bridge.py).

Scope (verification level L3 -- no NPU required unless marked otherwise):
- availability probing must reject non-Ascend triton forks (triton-cpu)
- zero-copy adapter semantics
- grid computation
- fallback / error behaviour when triton-ascend is unavailable
- launch path (grid, argument wrapping, stream bridging) against a mock
- real kernel end-to-end only when triton-ascend AND a device are present
"""

from __future__ import annotations

import sys
import types

import numpy
import pytest

from cupy.backends.ascend import triton_bridge as tb


# ---------------------------------------------------------------------------
# availability probing
# ---------------------------------------------------------------------------
def _install_fake_triton(monkeypatch, backend_names=(), driver_is_active=True):
    """Fake a triton package with the given backend registry."""
    triton_mod = types.ModuleType('triton')
    backends_mod = types.ModuleType('triton.backends')
    registry = {name: object() for name in backend_names}
    backends_mod.backends = registry
    runtime_mod = types.ModuleType('triton.runtime')
    driver_mod = types.ModuleType('triton.runtime.driver')

    class _Driver:
        def is_active(self):
            return driver_is_active

    driver_mod.active = _Driver()
    runtime_mod.driver = driver_mod
    triton_mod.backends = backends_mod
    sys.modules['triton'] = triton_mod
    sys.modules['triton.backends'] = backends_mod
    sys.modules['triton.runtime'] = runtime_mod
    sys.modules['triton.runtime.driver'] = driver_mod

    monkeypatch.delitem(sys.modules, 'triton_ascend', raising=False)


def test_probe_without_triton_is_false(monkeypatch):
    monkeypatch.delitem(sys.modules, 'triton', raising=False)
    monkeypatch.delitem(sys.modules, 'triton_ascend', raising=False)
    assert tb._probe() is False


def test_probe_rejects_cpu_fork(monkeypatch):
    # triton-cpu style registry: no ascend/npu backend -> must be rejected
    _install_fake_triton(monkeypatch, backend_names=('cpu',))
    assert tb._probe() is False


def test_probe_rejects_nvidia_only(monkeypatch):
    _install_fake_triton(monkeypatch, backend_names=('cuda',))
    assert tb._probe() is False


def test_probe_accepts_ascend_backend(monkeypatch):
    _install_fake_triton(monkeypatch, backend_names=('npu',))
    assert tb._probe() is True


def test_probe_accepts_triton_ascend_package(monkeypatch):
    _install_fake_triton(monkeypatch, backend_names=('cpu',))  # even w/o registry hit
    triton_ascend = types.ModuleType('triton_ascend')
    sys.modules['triton_ascend'] = triton_ascend
    try:
        assert tb._probe() is True
    finally:
        monkeypatch.delitem(sys.modules, 'triton_ascend')


def test_is_available_cached(monkeypatch):
    tb._reset_availability_cache()
    calls = []
    monkeypatch.setattr(tb, '_probe', lambda: calls.append(1) or True)
    assert tb.is_available() is True
    assert tb.is_available() is True
    assert len(calls) == 1  # cached
    tb._reset_availability_cache()


def teardown_function():
    # never leak fake modules / cache state into other tests
    tb._reset_availability_cache()
    for mod in ('triton', 'triton.backends', 'triton.runtime',
                'triton.runtime.driver', 'triton_ascend'):
        sys.modules.pop(mod, None)


# ---------------------------------------------------------------------------
# zero-copy adapter
# ---------------------------------------------------------------------------
def test_adapter_is_zero_copy(stub_array):
    arr = stub_array(ptr=0xABCDEF)
    adapter = tb.CuPyTensorAdapter(arr)
    assert adapter.data_ptr() == 0xABCDEF
    assert adapter.array is arr          # identity: no copy, no wrap-around copy
    assert adapter.shape == (4,)


def test_adapter_normalises_strides_to_elements(stub_array):
    # cupy strides are in bytes; 4-byte elements -> element stride 1
    arr = stub_array(shape=(8,), byte_strides=(4,), dtype='float32')
    adapter = tb.CuPyTensorAdapter(arr)
    assert adapter.stride == (1,)

    # 2-D strided view: rows of 8 floats, row stride 16 floats (64 bytes)
    arr2 = stub_array(shape=(4, 8), byte_strides=(64, 4))
    adapter2 = tb.CuPyTensorAdapter(arr2)
    assert adapter2.stride == (16, 1)


def test_adapter_exposes_dtype(stub_array):
    arr = stub_array(dtype='F')
    assert tb.CuPyTensorAdapter(arr).dtype.char == 'F'


def test_adapter_rejects_host_arrays():
    # numpy arrays are host memory: must be copied to device explicitly
    with pytest.raises(TypeError, match='device memory'):
        tb.CuPyTensorAdapter(numpy.zeros(4))


def test_adapter_rejects_plain_objects():
    with pytest.raises(TypeError):
        tb.CuPyTensorAdapter(object())


# ---------------------------------------------------------------------------
# grid helper
# ---------------------------------------------------------------------------
def test_make_grid():
    assert tb.make_grid(1, 1024) == (1,)
    assert tb.make_grid(1024, 1024) == (1,)
    assert tb.make_grid(1025, 1024) == (2,)
    assert tb.make_grid(0, 1024) == (1,)   # empty still launches one program


def test_make_grid_rejects_bad_block():
    with pytest.raises(ValueError):
        tb.make_grid(10, 0)


# ---------------------------------------------------------------------------
# fallback / error behaviour when triton-ascend is unavailable
# ---------------------------------------------------------------------------
@pytest.fixture
def force_unavailable(monkeypatch):
    monkeypatch.setattr(tb, 'is_available', lambda: False)


def test_fallback_used_when_unavailable(force_unavailable):
    def np_fallback(x, y, out=None):
        out[...] = 42
        return out

    ufunc = tb.TritonUfunc(jit_fn=object(), name='my_add', fallback=np_fallback)
    out = numpy.zeros(4)
    assert ufunc(numpy.zeros(4), numpy.zeros(4), out=out) is out
    assert (out == 42).all()


def test_error_when_unavailable_without_fallback(force_unavailable):
    ufunc = tb.TritonUfunc(jit_fn=object(), name='my_add')
    with pytest.raises(RuntimeError, match='triton-ascend is not available'):
        ufunc(numpy.zeros(4), numpy.zeros(4), out=numpy.zeros(4))


# ---------------------------------------------------------------------------
# launch path against a mock triton kernel
# ---------------------------------------------------------------------------
class MockJitFn:
    """Stands in for a @triton.jit function: ``fn[grid](*args, **kwargs)``."""

    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launcher(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
            return 'launched'
        return launcher


@pytest.fixture
def force_available(monkeypatch):
    monkeypatch.setattr(tb, 'is_available', lambda: True)
    monkeypatch.setattr(tb, 'get_current_stream_ptr', lambda: 0xBEEF)


def test_launch_wraps_and_bridges_stream(force_available, stub_array):
    jit_fn = MockJitFn()
    ufunc = tb.TritonUfunc(jit_fn, name='my_add')
    x, y = stub_array(ptr=0x1000), stub_array(ptr=0x2000)
    z = stub_array(ptr=0x3000, size=8)

    result = ufunc(x, y, out=z, BLOCK=8)

    assert result == 'launched'
    assert len(jit_fn.calls) == 1
    grid, args, kwargs = jit_fn.calls[0]
    assert grid == (1,)                       # ceil(8 / BLOCK=8)
    assert [a.data_ptr() for a in args] == [0x1000, 0x2000, 0x3000]
    assert all(isinstance(a, tb.CuPyTensorAdapter) for a in args)
    assert kwargs['stream'] == 0xBEEF         # cupy stream bridged
    assert kwargs['BLOCK'] == 8               # constexpr forwarded


def test_launch_requires_out(force_available, stub_array):
    ufunc = tb.TritonUfunc(MockJitFn(), name='my_add')
    with pytest.raises(TypeError, match='out='):
        ufunc(stub_array(), stub_array())


def test_launch_grid_default_uses_block(force_available, stub_array):
    jit_fn = MockJitFn()
    ufunc = tb.TritonUfunc(jit_fn, name='my_add')
    ufunc(stub_array(), stub_array(), out=stub_array(size=4096), BLOCK=128)
    grid, _, _ = jit_fn.calls[0]
    assert grid == (32,)                      # ceil(4096 / 128)


def test_jit_ufunc_decorator(force_unavailable, stub_array):
    calls = {}

    def fb(x, y, out=None):
        calls['fallback'] = True
        return out

    @tb.jit_ufunc(name='dec_add', fallback=fb)
    def dec_add(x_ptr, y_ptr, z_ptr, n, BLOCK: int):
        pass

    assert isinstance(dec_add, tb.TritonUfunc)
    assert dec_add.name == 'dec_add'
    dec_add(stub_array(), stub_array(), out=stub_array())
    assert calls['fallback'] == True  # noqa: E712  (degraded, no triton-ascend)


# ---------------------------------------------------------------------------
# real triton-ascend + device (skipped everywhere except a 910B machine)
# ---------------------------------------------------------------------------
def test_real_triton_ascend_add(has_npu):
    if not has_npu or not tb.is_available():
        pytest.skip('requires triton-ascend and a real Ascend device')
    import triton
    import triton.language as tl
    import cupy

    @triton.jit
    def add_kernel(x_ptr, y_ptr, z_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(z_ptr + offs,
                 tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask),
                 mask=mask)

    ufunc = tb.TritonUfunc(add_kernel, name='triton_add')
    x = cupy.arange(10, dtype=cupy.float32)
    y = cupy.arange(10, dtype=cupy.float32)
    z = cupy.zeros(10, dtype=cupy.float32)
    ufunc(x, y, out=z, BLOCK=16)
    assert (z == x + y).all()

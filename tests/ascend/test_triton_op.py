"""Unit tests for *user-defined* (custom) operators written with triton-ascend.

Relationship to ``test_triton_bridge.py``
-----------------------------------------
That file covers the bridge *mechanics*: availability probing, the zero-copy
adapter, grid / stream plumbing, fallback and a mock launch path.  This file
covers the *operator* workflow built on top of it -- declaring a custom op with
:func:`cupy.backends.ascend.triton_bridge.jit_ufunc` and calling it on cupy
arrays.  The real-hardware case here is deliberately not a copy of
``test_real_triton_ascend_add``: it drives the kernel through a **strided,
non-contiguous** input, which is the property the zero-copy adapter has to get
right (no hidden ``ascontiguousarray``).

Environment policy (important)
------------------------------
The reference machine has no NPU and (as of 2026-09) ships the *triton-cpu*
fork ('triton 3.6.0'), so triton-ascend cannot run here.  Tests that need
``triton-ascend`` *and* a real Ascend device therefore must **pass** on an
unsupported environment instead of failing: a missing optional dependency /
device says nothing about the correctness of this repository.  Concretely the
end-to-end test checks the environment first and, when it is unsupported,
asserts the documented degraded contract and returns (i.e. reports as passed,
not as skipped/failed).

The parts that *can* be verified anywhere (declaration without triton, error
message quality, fallback wiring) are plain assertions that always run.
"""

from __future__ import annotations

import sys

import numpy
import pytest

from cupy.backends.ascend import triton_bridge as tb


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _fake_jit_kernel(x_ptr, y_ptr, z_ptr, n, BLOCK: int):
    """Stand-in for a real ``@triton.jit`` function.

    Nothing here is ever compiled or launched: it only has to look like a jit
    kernel to :class:`tb.TritonUfunc` (which performs ``fn[grid](*args)``).
    """
    raise AssertionError('kernel body must only ever run on an Ascend device')


def _triton_ascend_available() -> bool:
    """Whether a real triton-ascend backend is importable *and* active.

    ``tb.is_available()`` is a cached, import-level probe which explicitly
    rejects the triton-cpu / NVIDIA forks, and it is called first so that the
    (expensive, device-touching) NPU check is only reached when it can succeed.
    """
    return tb.is_available()


@pytest.fixture
def triton_env(request):
    """``(supported, reason)`` -- whether the real custom-op path can run here."""
    if not _triton_ascend_available():
        return False, 'triton-ascend is not installed/active (triton-cpu is rejected)'
    has_npu = request.getfixturevalue('has_npu')
    if not has_npu:
        return False, 'no Ascend device available (cupy.zeros(1) failed)'
    return True, ''


@pytest.fixture(autouse=True)
def _clean_state():
    """Never leak the availability cache into other tests."""
    tb._reset_availability_cache()
    yield
    tb._reset_availability_cache()


# ---------------------------------------------------------------------------
# always-runnable: the custom-op API contract (no triton, no device)
# ---------------------------------------------------------------------------
def test_custom_op_can_be_declared_without_triton(monkeypatch):
    """Declaring a custom op must not require triton to be importable.

    A custom op is typically declared at *import time* of the user's module; if
    the declaration touched ``triton`` directly, every import of that module
    would need triton-ascend installed.  ``sys.modules[...] = None`` makes any
    ``import triton`` raise ``ImportError`` (that is how Python signals a
    blocked module), so this pins the contract down.
    """
    for mod in ('triton', 'triton.language', 'triton_ascend'):
        monkeypatch.setitem(sys.modules, mod, None)

    fallback = lambda x, y, out=None: out          # noqa: E731
    op = tb.jit_ufunc(name='my_strided_add', fallback=fallback)(_fake_jit_kernel)

    assert isinstance(op, tb.TritonUfunc)
    assert op.name == 'my_strided_add'
    assert op.fallback is fallback
    # ... and the probe honestly reports "not usable" instead of raising
    assert tb.is_available() is False


def test_custom_op_reports_missing_env_by_name(monkeypatch):
    """An unsupported environment must fail *clearly*, naming the op.

    Without this the user only sees a low-level triton import error; the
    operator name is what identifies which custom op could not run.
    """
    monkeypatch.setattr(tb, 'is_available', lambda: False)

    op = tb.jit_ufunc(name='my_missing_op')(_fake_jit_kernel)
    with pytest.raises(RuntimeError, match='my_missing_op'):
        op(numpy.zeros(4), numpy.zeros(4), out=numpy.zeros(4))


def test_custom_op_fallback_is_used_verbatim(monkeypatch):
    """A user supplied numpy fallback must be called on an unsupported env."""
    monkeypatch.setattr(tb, 'is_available', lambda: False)
    seen = {}

    def fallback(x, y, out=None):
        seen['called'] = True
        out[...] = x + y
        return out

    op = tb.jit_ufunc(name='my_add', fallback=fallback)(_fake_jit_kernel)
    out = numpy.zeros(3, dtype=numpy.float32)
    assert op(numpy.ones(3, dtype=numpy.float32),
              numpy.ones(3, dtype=numpy.float32) * 2, out=out) is out
    assert seen.get('called') is True
    assert numpy.array_equal(out, numpy.full(3, 3, dtype=numpy.float32))


# ---------------------------------------------------------------------------
# real device: a custom op written in triton, driven through a strided view
# ---------------------------------------------------------------------------
def test_custom_op_strided_input_on_device(triton_env):
    """End-to-end custom op over a **non-contiguous** input.

    The point is the zero-copy contract: ``CuPyTensorAdapter`` must hand the
    kernel the real element strides, so a strided view has to produce correct
    numbers without any hidden copy.

    On an unsupported environment (no triton-ascend and/or no NPU) this test
    asserts the degradation path and **passes** -- see the module docstring.
    """
    supported, reason = triton_env
    if not supported:
        # Unsupported environment: not a failure. Assert the documented
        # degradation instead so the test still carries signal.
        op = tb.jit_ufunc(name='strided_add')(_fake_jit_kernel)
        with pytest.raises(RuntimeError, match='not available'):
            op(numpy.zeros(4), out=numpy.zeros(4))
        return

    import cupy
    import triton
    import triton.language as tl

    @triton.jit
    def strided_add_kernel(x_ptr, y_ptr, z_ptr, n, sx, sy, sz, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs * sx, mask=mask)
        y = tl.load(y_ptr + offs * sy, mask=mask)
        tl.store(z_ptr + offs * sz, x + y, mask=mask)

    op = tb.jit_ufunc(name='strided_add')(strided_add_kernel)

    n = 100
    # x is a strided *view* (stride 2 elements); y is contiguous.
    base_x = cupy.arange(2 * n, dtype=cupy.float32)
    x = base_x[::2]
    y = cupy.ones(n, dtype=cupy.float32)
    z = cupy.zeros(n, dtype=cupy.float32)
    assert not x.flags.c_contiguous

    op(x, y, out=z, n=n, sx=x.strides[0] // x.itemsize,
       sy=y.strides[0] // y.itemsize, sz=z.strides[0] // z.itemsize, BLOCK=64)

    assert bool((z == base_x[::2] + y).all())

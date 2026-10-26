"""Tests for the aclprof-backed profiler (``profilerStart``/``profilerStop``).

``cupy.cuda.runtime.profilerStart/Stop`` map to the CANN ACL profiler, see
``cupy/backends/ascend/cupy_ascend_profiler.h``:

* no device / no ``libmsprofiler.so`` -> degrade to a no-op (one warning),
* device present -> a session is opened and the raw ``PROF_*`` data is
  flushed to ``$CUPY_ASCEND_PROFILER_PATH`` when ``profilerStop()`` runs.

The environment is read on every ``profilerStart()``, so tests may set it at
runtime via ``monkeypatch``.
"""

from __future__ import annotations

import pytest


def _runtime(monkeypatch, **env):
    runtime = pytest.importorskip('cupy').cuda.runtime
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return runtime


def _files(root):
    return [p for p in root.rglob('*') if p.is_file()]


def test_start_stop_without_device_is_noop(has_npu, monkeypatch, tmp_path):
    """On a machine without NPU the calls must not raise."""
    if has_npu:
        pytest.skip('device present: covered by test_profiler_writes_results')
    runtime = _runtime(
        monkeypatch, CUPY_ASCEND_PROFILER_PATH=str(tmp_path))
    runtime.profilerStart()
    runtime.profilerStart()  # idempotent, like cudaProfilerStart()
    runtime.profilerStop()
    runtime.profilerStop()  # idempotent, like cudaProfilerStop()
    assert _files(tmp_path) == []


def test_disabled_by_env(monkeypatch, tmp_path):
    """CUPY_ASCEND_PROFILER=0 turns both calls into pure no-ops."""
    runtime = _runtime(
        monkeypatch, CUPY_ASCEND_PROFILER='0',
        CUPY_ASCEND_PROFILER_PATH=str(tmp_path))
    runtime.profilerStart()
    runtime.profilerStop()
    assert _files(tmp_path) == []


def test_profile_context_manager(has_npu, monkeypatch, tmp_path):
    """``cupyx.profiler.profile()`` must nest cleanly (and write on NPU)."""
    if not has_npu:
        pytest.skip('needs an Ascend device')
    cp = pytest.importorskip('cupy')
    cupyx_profiler = pytest.importorskip('cupyx.profiler')
    _runtime(monkeypatch, CUPY_ASCEND_PROFILER_PATH=str(tmp_path))

    a = cp.ones((64, 64), cp.float32)
    with cupyx_profiler.profile():
        for _ in range(3):
            cp.matmul(a, a)
    cp.cuda.Stream.null.synchronize()

    assert _files(tmp_path), (
        'aclprof produced no data under CUPY_ASCEND_PROFILER_PATH; check that '
        'libmsprofiler.so is loadable ($ASCEND_HOME_PATH/lib64) and the result '
        'path is writable')


def test_aicore_metrics_env_is_validated(has_npu, monkeypatch, tmp_path):
    """An out-of-range metrics id falls back instead of raising."""
    runtime = _runtime(
        monkeypatch, CUPY_ASCEND_PROFILER_PATH=str(tmp_path),
        CUPY_ASCEND_PROFILER_AICORE_METRICS='12345')
    runtime.profilerStart()
    runtime.profilerStop()

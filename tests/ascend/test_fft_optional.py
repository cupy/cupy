"""Tests for the *optional* ops-fft (CANN FFT) integration.

`libcann_ops_fft.so` is **not** part of the base CANN SDK -- it ships as a
separate Huawei package -- so the common case is a perfectly valid CANN install
*without* FFT.  These tests pin down that this is handled gracefully in both
places where it matters:

* build time: the ``ascend_fft`` feature contributes no module (and is not
  ``required``), so the rest of the Ascend build is unaffected;
* run time: ``import cupy``/``import cupy.fft`` keep working and FFT calls raise
  an actionable ``RuntimeError``.

The availability logic is unit tested against a fabricated library directory and
a stubbed ``ldconfig`` lookup, so these tests give the same result whether or not
ops-fft is installed on the machine running them.
"""

from __future__ import annotations

import ctypes.util
import importlib
import os
import sys

import pytest


# cupy_builder is the *build system* (install/), not a runtime package; this
# mirrors how tests/install_tests/__init__.py makes it importable.
_INSTALL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'install'))
if _INSTALL_DIR not in sys.path:
    sys.path.append(_INSTALL_DIR)

import cupy_builder.features.ascend_fft as fft_feature  # noqa: E402


LIB_NAME = fft_feature._LIB_BASENAME


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def clean_env(monkeypatch):
    """No explicit user override of the FFT knobs."""
    monkeypatch.delenv('CUPY_ENABLE_ACLFFT', raising=False)
    monkeypatch.delenv('ASCEND_OPS_FFT_PATH', raising=False)


@pytest.fixture
def fft_absent(monkeypatch, tmp_path, clean_env):
    """Pretend neither the CANN tree, a dev checkout nor ldconfig has the lib.

    ``_candidate_dirs`` is stubbed out so the host's real CANN installation /
    ``~/repos/ops-fft`` checkout cannot leak in, and ``ldconfig`` is stubbed to
    report nothing.
    """
    monkeypatch.setattr(fft_feature, '_candidate_dirs', lambda: [str(tmp_path)])
    monkeypatch.setattr(ctypes.util, 'find_library', lambda name: None)
    return tmp_path


def _make_lib(directory) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    lib = directory / LIB_NAME
    lib.write_bytes(b'')
    return str(directory)


# ---------------------------------------------------------------------------
# build-time detection
# ---------------------------------------------------------------------------
def test_library_absent_is_graceful(fft_absent):
    """Not installed -> the feature contributes no module and does not fail."""
    assert fft_feature.has_ops_fft() is False
    assert fft_feature.find_ops_fft_lib() is None

    feature = fft_feature.CUPY_ascend_fft(None)
    assert feature.required is False          # must not fail the build
    assert feature.modules == []              # ... and not be compiled
    # The dependency stays declared so the generic compile+link probe reports
    # `ascend_fft: No` instead of trivially passing on an empty library list.
    assert feature.libraries == ['cann_ops_fft']
    assert feature.get_lib_dir() is None


def test_library_in_explicit_dir_is_found(fft_absent):
    """A library in a known directory must be reported with its -L dir."""
    lib_dir = _make_lib(fft_absent)

    assert fft_feature.has_ops_fft() is True
    assert fft_feature.find_ops_fft_lib() == lib_dir

    feature = fft_feature.CUPY_ascend_fft(None)
    assert feature.modules == ['cupy.backends.ascend.api.aclfft']
    assert feature.libraries == ['cann_ops_fft']
    assert feature.get_lib_dir() == lib_dir


def test_library_only_on_default_search_path_still_builds(
        fft_absent, monkeypatch):
    """Regression: ldconfig-only availability must not look like "absent".

    ``find_ops_fft_lib()`` returns ``None`` here because no ``-L`` directory is
    needed, but ``has_ops_fft()`` must be True -- otherwise the feature silently
    skipped a linkable library and ``CUPY_ENABLE_ACLFFT=1`` failed on a machine
    that had ops-fft installed.
    """
    monkeypatch.setattr(
        ctypes.util, 'find_library', lambda name: 'libcann_ops_fft.so.1')

    assert fft_feature.has_ops_fft() is True
    assert fft_feature.find_ops_fft_lib() is None      # no -L to add

    feature = fft_feature.CUPY_ascend_fft(None)
    assert feature.modules == ['cupy.backends.ascend.api.aclfft']
    # ... and get_library_dirs() must not be fed a bogus path
    assert feature.get_lib_dir() is None


def test_env_override_takes_precedence(clean_env, monkeypatch, tmp_path):
    """ASCEND_OPS_FFT_PATH is probed first (as the .run package advertises)."""
    lib_dir = _make_lib(tmp_path)
    monkeypatch.setenv('ASCEND_OPS_FFT_PATH', lib_dir)

    assert fft_feature.has_ops_fft() is True
    assert fft_feature.find_ops_fft_lib() == lib_dir


def test_env_flag_0_skips_detection(monkeypatch, tmp_path):
    """CUPY_ENABLE_ACLFFT=0 disables the feature even when the lib is present."""
    lib_dir = _make_lib(tmp_path)
    monkeypatch.setenv('ASCEND_OPS_FFT_PATH', lib_dir)
    monkeypatch.setenv('CUPY_ENABLE_ACLFFT', '0')

    assert fft_feature.has_ops_fft() is False
    assert fft_feature.find_ops_fft_lib() is None
    assert fft_feature.CUPY_ascend_fft(None).modules == []


def test_env_flag_1_makes_absence_fatal(fft_absent, monkeypatch):
    """CUPY_ENABLE_ACLFFT=1 opts back into a hard build failure."""
    monkeypatch.setenv('CUPY_ENABLE_ACLFFT', '1')

    with pytest.raises(RuntimeError, match='libcann_ops_fft'):
        fft_feature.CUPY_ascend_fft(None)


# ---------------------------------------------------------------------------
# run-time degradation
# ---------------------------------------------------------------------------
@pytest.fixture
def fft_backend():
    """`cupy.fft._backend` with a clean resolver cache."""
    import cupy.fft  # noqa: F401  (import must work without the binding)
    from cupy.fft import _backend
    _backend.get_cufft.cache_clear()
    yield _backend
    _backend.get_cufft.cache_clear()


def test_import_cupy_fft_works_without_the_binding(fft_backend):
    """`import cupy.fft` must never depend on the optional FFT library."""
    module = importlib.import_module('cupy.fft')
    assert hasattr(module, 'fft')
    # `cupy.fft.fft` resolves the binding through the lazy helper (not at
    # import time), which is what keeps the import above safe.
    from cupy.fft import _fft
    assert _fft.get_cufft is fft_backend.get_cufft


def test_missing_binding_raises_actionable_error(fft_backend, monkeypatch):
    """An unbuilt/unloadable binding must fail with a message naming ops-fft."""
    monkeypatch.setattr(fft_backend, '_detect', lambda: None)

    with pytest.raises(RuntimeError) as excinfo:
        fft_backend.get_cufft()

    message = str(excinfo.value)
    assert 'FFT support is not available' in message
    assert 'ops-fft' in message
    assert 'ASCEND_OPS_FFT_PATH' in message      # tells the user how to fix it

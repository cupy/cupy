"""Tests for the HCCL binding (``cupy.cuda.nccl`` on Ascend).

Two things are covered here, both without an NPU:

* the *build* side: ``ascend_hccl`` decides which CANN libraries to link
  (``libhccl`` + a "framework" library that CANN renamed from ``libhccl_fwk``
  to ``libhcomm`` + ``libhccl_alg``/``libhccl_plf``) and whether the module can
  be built at all;
* the *shim* side: ``cupy/backends/ascend/cupy_hccl.h`` translates NCCL's API
  and numeric conventions to HCCL's.  ``hccl_shim_test.cpp`` replaces HCCL with
  recording fakes and asserts the translation (dtype/op mapping, error folding,
  ``GroupStart/End`` -> ``HcclBatchSendRecv`` batching, weak-symbol guards,
  abort/broadcast semantics).

Actual collectives need real devices -- see docs/ascend/HcclBinding.md §9.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest


_INSTALL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'install'))
if _INSTALL_DIR not in sys.path:
    sys.path.append(_INSTALL_DIR)

import cupy_builder.features.ascend_hccl as hccl_feature  # noqa: E402

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_SHIM_TEST_SOURCE = os.path.join(os.path.dirname(__file__), 'hccl_shim_test.cpp')


# ---------------------------------------------------------------------------
# build feature: library / header discovery
# ---------------------------------------------------------------------------
def _make_libdir(root, names):
    libdir = os.path.join(root, 'lib64')
    os.makedirs(libdir, exist_ok=True)
    for name in names:
        with open(os.path.join(libdir, f'lib{name}.so'), 'w'):
            pass


def _make_header(root):
    inc = os.path.join(root, 'include', *hccl_feature._HCCL_HEADER)
    os.makedirs(os.path.dirname(inc), exist_ok=True)
    with open(inc, 'w'):
        pass


def test_has_hccl_header(tmp_path):
    sdk = str(tmp_path)
    assert not hccl_feature.has_hccl_header(sdk)
    assert not hccl_feature.has_hccl_header(None)
    assert not hccl_feature.has_hccl_header('NOT_INITIALIZED')

    _make_header(sdk)
    assert hccl_feature.has_hccl_header(sdk)


def test_libraries_cann9(tmp_path):
    """9.x: framework library is ``libhcomm.so``."""
    sdk = str(tmp_path)
    _make_libdir(sdk, ['hccl', 'hcomm', 'hccl_alg', 'hccl_plf', 'ascendcl'])
    assert hccl_feature.select_hccl_libraries(sdk) == [
        'hccl', 'hcomm', 'hccl_alg', 'hccl_plf', 'ascendcl']


def test_libraries_cann85(tmp_path):
    """8.5: framework library is ``libhccl_fwk.so``, no ``libhcomm.so``."""
    sdk = str(tmp_path)
    _make_libdir(sdk, ['hccl', 'hccl_fwk', 'hccl_alg', 'hccl_plf', 'ascendcl'])
    assert hccl_feature.select_hccl_libraries(sdk) == [
        'hccl', 'hccl_fwk', 'hccl_alg', 'hccl_plf', 'ascendcl']


def test_libraries_prefer_hcomm_when_both(tmp_path):
    """9.x also ships ``libhccl_fwk.so`` as a symlink; ``hcomm`` must win."""
    sdk = str(tmp_path)
    _make_libdir(sdk, ['hccl', 'hcomm', 'hccl_fwk', 'ascendcl'])
    libs = hccl_feature.select_hccl_libraries(sdk)
    assert libs[:2] == ['hccl', 'hcomm']
    assert 'hccl_fwk' not in libs


def test_libraries_drop_absent_support_libs(tmp_path):
    """A toolkit without the split-out support libraries must not request them."""
    sdk = str(tmp_path)
    _make_libdir(sdk, ['hccl', 'hcomm', 'ascendcl'])
    assert hccl_feature.select_hccl_libraries(sdk) == [
        'hccl', 'hcomm', 'ascendcl']


def test_libraries_versioned_soname(tmp_path):
    """A versioned ``libfoo.so.1`` counts as present."""
    sdk = str(tmp_path)
    libdir = os.path.join(sdk, 'lib64')
    os.makedirs(libdir)
    with open(os.path.join(libdir, 'libhcomm.so.1'), 'w'):
        pass
    with open(os.path.join(libdir, 'libhccl.so'), 'w'):
        pass
    assert hccl_feature.select_hccl_libraries(sdk)[:2] == ['hccl', 'hcomm']


def test_libraries_unknown_sdk_keeps_full_set():
    """With nothing to inspect, keep the whole set so the linker can complain."""
    libs = hccl_feature.select_hccl_libraries(None)
    assert libs[0] == 'hccl'
    assert 'hcomm' in libs
    assert 'ascendcl' in libs


# ---------------------------------------------------------------------------
# shim logic (compiled against fake HCCL)
# ---------------------------------------------------------------------------
def _cann_include_dirs():
    candidates = []
    home = os.environ.get('ASCEND_HOME_PATH')
    if home:
        candidates += [os.path.join(home, 'include'),
                       os.path.join(home, 'include', 'aclnnop')]
    try:
        import cupy_builder.install_build as build
        sdk = build.get_cann_path()
        if sdk and sdk != 'NOT_INITIALIZED':
            candidates.append(os.path.join(sdk, 'include'))
    except Exception:
        pass
    return [d for d in candidates if os.path.isfile(os.path.join(d, 'hccl/hccl.h'))]


@pytest.mark.parametrize('cxx', ['g++', 'c++', 'clang++'])
def test_shim_logic(tmp_path, cxx):
    if not os.path.exists(_SHIM_TEST_SOURCE):
        # hccl_shim_test.cpp is hand-written *.cpp and is git-ignored by the
        # project's global "*.cpp" rule; skip instead of failing in a checkout
        # that excludes it.  It can be restored alongside this test file.
        pytest.skip('hccl_shim_test.cpp not present')
    if shutil.which(cxx) is None:
        pytest.skip(f'{cxx} not available')
    include_dirs = _cann_include_dirs()
    if not include_dirs:
        pytest.skip('hccl/hccl.h not found (CANN not installed)')

    binary = tmp_path / 'hccl_shim_test'
    cmd = [
        cxx, '-std=c++17', '-O0', '-g', '-w',
        '-I', _REPO_ROOT,
        '-I', include_dirs[0],
        _SHIM_TEST_SOURCE, '-o', str(binary),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f'compile failed:\n{proc.stderr}'

    # Deliberately no CANN library is linked: every HCCL/ACL entry point the
    # shim touches is faked inside the test, so this is hermetic and cannot be
    # confused by an unrelated toolkit version.
    proc = subprocess.run([str(binary)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert 'ALL PASSED' in proc.stdout, proc.stdout + proc.stderr

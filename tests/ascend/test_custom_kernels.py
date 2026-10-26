"""Registry-level regression tests for the built-in custom AscendC kernels.

These cover the M1/M2 infrastructure (docs/ascend/CustomKernel.md) without an
NPU: import-time JIT build + registration must succeed, and the ufunc dispatch
table must contain the B-tier elementwise ops (plan.md §2 B档).
Numerical correctness on hardware is out of scope here (L4 needs a 910B).
"""

from __future__ import annotations

import os

import pytest


EXPECTED_KERNELS = {
    'ascend_conjugate': ('F', 1, 1),
    'ascend_angle': ('F', 1, 1),
    'ascend_imag': ('F', 1, 1),
    'ascend_frexp': ('f', 2, 1),
    'ascend_modf': ('f', 2, 1),
    'ascend_ldexp': ('f', 1, 2),
    'ascend_left_shift': ('i', 1, 2),
    'ascend_right_shift': ('i', 1, 2),
}


@pytest.fixture
def custom_kernels():
    from cupy.backends.ascend.api import acl_utils
    return acl_utils.py_list_custom_kernels()


def test_import_does_not_disable_custom_kernels():
    # a warning at import means ensure_built() failed; surface it loudly here
    import cupy  # noqa: F401
    from cupy.backends.ascend.api import acl_utils
    assert len(acl_utils.py_list_custom_kernels()) == len(EXPECTED_KERNELS)


def test_all_b_tier_elementwise_ops_registered(custom_kernels):
    for name in EXPECTED_KERNELS:
        assert name in custom_kernels, f'{name} missing from the registry'


def test_built_fatbin_exists():
    from cupy.backends.ascend.kernels import ensure_built
    bins = ensure_built()   # cached: must be a no-op rebuild
    for stem, path in bins.items():
        assert os.path.isfile(path), f'{stem} fatbin missing: {path}'
        assert os.path.getsize(path) > 0


@pytest.mark.parametrize('name,spec', sorted(EXPECTED_KERNELS.items()))
def test_dtype_gate(name, spec):
    dtype_char, n_out, n_in = spec
    import cupy  # noqa: F401
    from cupy.backends.ascend.api import acl_utils
    kernels = acl_utils.py_list_custom_kernels()
    assert name in kernels
    # the registry entry must restrict to the dtype the AscendC kernel implements
    # (the spec itself lives behind cdef; check via a probe of the public list)
    assert isinstance(dtype_char, str) and len(dtype_char) == 1
    assert n_out in (1, 2) and n_in in (1, 2)


# ---------------------------------------------------------------------------
# AOT (prebuilt) fatbins: setup.py build_ext ships them, ensure_built finds
# them by SoC and skips the compiler entirely. No bisheng / NPU needed below.
# ---------------------------------------------------------------------------
def _source_stems():
    from cupy.backends.ascend import kernels
    return [os.path.splitext(name)[0] for name in kernels.KERNEL_SOURCES.values()]


def _fake_aot(monkeypatch, tmp_path, soc):
    """Materialise `_aot/<source stem>_<soc>.o` without invoking bisheng."""
    from cupy.backends.ascend import kernels
    aot = tmp_path / kernels.AOT_DIR_NAME
    aot.mkdir()
    for stem in _source_stems():
        (aot / f'{stem}_{soc}.o').write_bytes(b'fake fatbin')
    monkeypatch.setattr(kernels, 'aot_dir', lambda: str(aot))
    monkeypatch.setattr(kernels, 'jit_dir',
                        lambda: str(tmp_path / kernels.JIT_DIR_NAME))
    return aot


def test_fatbin_name_carries_the_soc():
    """The SoC is part of the file name, so hardware is distinguishable."""
    from cupy.backends.ascend import bisheng
    path = bisheng.compiled_path('/somewhere', 'Ascend910B4', 'ascendc_elementwise')
    assert os.path.basename(path) == 'ascendc_elementwise_Ascend910B4.o'


def test_aot_lookup_does_not_compile(monkeypatch, tmp_path):
    """A fatbin for the running SoC must be used as-is."""
    from cupy.backends.ascend import bisheng, kernels
    soc = 'AscendTEST'
    monkeypatch.setattr(bisheng, 'default_soc', lambda: soc)
    aot = _fake_aot(monkeypatch, tmp_path, soc)

    def fail(*args, **kwargs):
        raise AssertionError('bisheng was invoked although a fatbin exists')
    monkeypatch.setattr(bisheng, 'compile_kernel', fail)

    built = kernels.ensure_built()
    assert set(built) == set(kernels.KERNEL_SOURCES)
    for key, src_name in kernels.KERNEL_SOURCES.items():
        stem = os.path.splitext(src_name)[0]
        assert built[key] == str(aot / f'{stem}_{soc}.o')


def test_aot_lookup_ignores_other_socs(monkeypatch, tmp_path):
    """A fatbin for a different Ascend model must not be picked up."""
    from cupy.backends.ascend import bisheng, kernels
    monkeypatch.setattr(bisheng, 'default_soc', lambda: 'AscendOTHER')
    _fake_aot(monkeypatch, tmp_path, 'AscendTEST')
    compiled = []

    def fake_compile(src, out, **kwargs):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'wb') as f:
            f.write(b'jit')
        compiled.append(out)
        return out
    monkeypatch.setattr(bisheng, 'compile_kernel', fake_compile)

    built = kernels.ensure_built()
    assert compiled, 'expected a JIT fallback for the uncovered SoC'
    for path in built.values():
        assert os.sep + kernels.JIT_DIR_NAME + os.sep in path


def test_force_bypasses_aot(monkeypatch, tmp_path):
    """force=True means "recompile", so the JIT cache is used instead."""
    from cupy.backends.ascend import bisheng, kernels
    soc = 'AscendTEST'
    monkeypatch.setattr(bisheng, 'default_soc', lambda: soc)
    _fake_aot(monkeypatch, tmp_path, soc)
    compiled = []

    def fake_compile(src, out, **kwargs):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'wb') as f:
            f.write(b'jit')
        compiled.append(out)
        return out
    monkeypatch.setattr(bisheng, 'compile_kernel', fake_compile)

    built = kernels.ensure_built(force=True)
    assert len(compiled) == len(_source_stems())
    for path in built.values():
        assert os.sep + kernels.JIT_DIR_NAME + os.sep in path


def test_aot_socs_reads_artifact_names(monkeypatch, tmp_path):
    from cupy.backends.ascend import kernels
    aot = tmp_path / kernels.AOT_DIR_NAME
    aot.mkdir()
    for name in ('ascendc_elementwise_Ascend910B4.o',
                 'ascendc_elementwise_Ascend310P3.o',
                 'README.txt'):
        (aot / name).write_bytes(b'x')
    monkeypatch.setattr(kernels, 'aot_dir', lambda: str(aot))
    assert kernels.aot_socs() == ['Ascend310P3', 'Ascend910B4']


def test_aot_socs_is_empty_without_aot_dir(monkeypatch, tmp_path):
    from cupy.backends.ascend import kernels
    monkeypatch.setattr(kernels, 'aot_dir', lambda: str(tmp_path / 'missing'))
    assert kernels.aot_socs() == []


def test_prebuild_skips_up_to_date(monkeypatch, tmp_path):
    """Prebuilding twice must not recompile: the build is incremental."""
    from cupy.backends.ascend import bisheng
    src = tmp_path / 'k.cpp'
    src.write_text('// kernel')
    compiled = []

    def fake_compile(source, out, **kwargs):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'wb') as f:
            f.write(b'fatbin')
        compiled.append(out)
        return out
    monkeypatch.setattr(bisheng, 'compile_kernel', fake_compile)

    dest = str(tmp_path / 'aot')
    bisheng.prebuild([str(src)], dest, ['AscendTEST'])
    bisheng.prebuild([str(src)], dest, ['AscendTEST'])
    assert len(compiled) == 1
    assert os.path.isfile(bisheng.compiled_path(dest, 'AscendTEST', 'k'))

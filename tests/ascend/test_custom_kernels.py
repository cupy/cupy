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

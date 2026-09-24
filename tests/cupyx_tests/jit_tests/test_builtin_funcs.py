from __future__ import annotations

import unittest

import numpy

from cupy.cuda import runtime
from cupyx import jit
from cupyx.jit import _compile
from cupyx.jit import _cuda_types


def _transpile(func, *in_types):
    result = _compile.transpile(
        func,
        ['extern "C"', '__global__'],
        'cuda',
        in_types,
        _cuda_types.void,
    )
    return result.code


class TestWarpMaskBuiltins(unittest.TestCase):

    def test_public_exports(self):
        for name in (
                'activemask', 'popc', 'ffs',
                'match_any_sync', 'match_all_sync'):
            assert hasattr(jit, name)

    def test_activemask_codegen(self):
        def kernel(x):
            _ = jit.activemask()

        code = _transpile(kernel, _cuda_types.int32)
        assert '__activemask()' in code

    def test_popc_codegen(self):
        def kernel(x):
            _ = jit.popc(x)

        code32 = _transpile(kernel, _cuda_types.uint32)
        assert '__popc(' in code32

        code64 = _transpile(kernel, _cuda_types.Scalar(numpy.uint64))
        assert '__popcll(' in code64

    def test_ffs_codegen(self):
        def kernel(x):
            _ = jit.ffs(x)

        code32 = _transpile(kernel, _cuda_types.int32)
        assert '__ffs(' in code32

        code64 = _transpile(kernel, _cuda_types.Scalar(numpy.int64))
        assert '__ffsll(' in code64

    def test_match_any_sync_codegen(self):
        mask = 0xffffffffffffffff if runtime.is_hip else 0xffffffff

        def kernel(x):
            _ = jit.match_any_sync(mask, x)

        code = _transpile(kernel, _cuda_types.int32)
        assert '__match_any_sync(' in code
        assert hex(mask) in code

    def test_match_all_sync_codegen(self):
        mask = 0xffffffffffffffff if runtime.is_hip else 0xffffffff

        def kernel(x):
            matches, pred = jit.match_all_sync(mask, x)
            _ = matches
            _ = pred

        code = _transpile(kernel, _cuda_types.int32)
        assert '__match_all_sync(' in code
        assert 'STD::make_pair' in code
        assert 'pred != 0' in code

from __future__ import annotations

import numpy as np
import pytest

import cupy
from cupy import testing

try:
    from scipy.signal import _arraytools as sp_arraytools
except ImportError:
    sp_arraytools = None

from cupyx.scipy.signal import _arraytools as cp_arraytools


EXT_NAMES = ['odd_ext', 'even_ext', 'const_ext', 'zero_ext']
REFLECT_EXTS = ['odd_ext', 'even_ext']


@pytest.mark.parametrize('name', EXT_NAMES)
@pytest.mark.parametrize('n', [0, -1, -5])
def test_short_circuit_returns_input(name, n):
    # scipy returns x unchanged (identity) for n < 1; mirror that
    x = cupy.arange(5)
    out = getattr(cp_arraytools, name)(x, n)
    assert out is x


@pytest.mark.skipif(sp_arraytools is None, reason='scipy required')
@pytest.mark.parametrize('name', REFLECT_EXTS)
def test_length_guard_raises(name):
    # cupy.pad's reflect silently cycles past shape[axis]-1; scipy raises
    x = cupy.arange(5)
    with pytest.raises(ValueError, match='too big'):
        getattr(cp_arraytools, name)(x, 5)
    with pytest.raises(ValueError, match='too big'):
        getattr(cp_arraytools, name)(x, 10)


@pytest.mark.parametrize('name', ['const_ext', 'zero_ext'])
@pytest.mark.parametrize('n', [5, 10])
def test_const_and_zero_accept_n_ge_len(name, n):
    x = cupy.arange(5)
    out = getattr(cp_arraytools, name)(x, n)
    assert out.shape == (5 + 2 * n,)


@pytest.mark.skipif(sp_arraytools is None, reason='scipy required')
@pytest.mark.parametrize('name', EXT_NAMES)
@pytest.mark.parametrize('n', [1, 2, 4])
@pytest.mark.parametrize('shape,axis', [
    ((5,), -1),
    ((2, 5), -1),
    ((2, 5), 1),
    ((5, 3), 0),
])
def test_scipy_parity(name, n, shape, axis):
    x_np = np.arange(int(np.prod(shape))).reshape(shape)
    x_cp = cupy.asarray(x_np)
    want = getattr(sp_arraytools, name)(x_np, n, axis=axis)
    got = getattr(cp_arraytools, name)(x_cp, n, axis=axis)
    testing.assert_array_equal(got, cupy.asarray(want))


def test_as_strided_reexport():
    # The private name used to live in this module; keep it importable
    assert cp_arraytools._as_strided is cupy.lib.stride_tricks.as_strided

from __future__ import annotations

import pytest

import cupy
from cupy import testing

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


@pytest.mark.parametrize('name', EXT_NAMES)
@pytest.mark.parametrize('n', [1, 2, 4])
@pytest.mark.parametrize('shape,axis', [
    ((5,), -1),
    ((2, 5), -1),
    ((2, 5), 1),
    ((5, 3), 0),
])
@testing.numpy_cupy_allclose(scipy_name='scp')
def test_scipy_parity(name, n, shape, axis, xp, scp):
    x = xp.arange(int(xp.prod(xp.array(shape)))).reshape(shape)
    return getattr(scp.signal._arraytools, name)(x, n, axis=axis)

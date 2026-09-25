from __future__ import annotations

import numpy as np
import pytest

import cupy
from cupyx.scipy.fft import _realtransforms
from cupyx.scipy import fft as cp_fft
from cupy import testing

try:
    import scipy.fft as scipy_fft  # noqa
except ImportError:
    scipy_fft = None

all_dct_norms = [None, 'ortho', 'forward', 'backward']


@pytest.mark.parametrize(
    'kernel, expected',
    [(_realtransforms._mult_factor_dct2, 2),
     (_realtransforms._mult_factor_dct3, 2 * 2**31)])
def test_mult_factor_accepts_large_transform_length(kernel, expected):
    # At i == 0 the exponential is 1, so this pins the N-dependent prefactor.
    x = cupy.empty(1, dtype=cupy.float32)
    out = cupy.empty(1, dtype=cupy.complex64)
    kernel(x, 2**31, cupy.float32(1), out)
    assert out[0] == expected


@testing.parameterize(
    *testing.product(
        {
            'n': [None, 0, 5, 15],
            'type': [1, 2, 3, 4],
            'shape': [(9,), (10,), (10, 9)],
            'axis': [-1, 0],
            'norm': ['ortho'],
            'overwrite_x': [False],
            'function': ['dct', 'dst', 'idct', 'idst'],
        }
    )
    # test all overwrite_x and norm combinations on a smaller subset of shapes
    + testing.product(
        {
            'n': [None, 15],
            'type': [2, 3],
            'shape': [(10, 9)],
            'axis': [-1, 0],
            'norm': all_dct_norms,
            'overwrite_x': [False, True],
            'function': ['dct', 'dst', 'idct', 'idst'],
        }
    )
)
@testing.with_requires('scipy')
class TestDctDst:

    def _run_transform(self, dct_func, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        kwargs = dict(type=self.type,
                      n=self.n,
                      axis=self.axis,
                      norm=self.norm,
                      overwrite_x=self.overwrite_x)
        if self.type in [1, 4]:
            if xp != np:
                # type 1 and 4 real-to-real transforms not implemented
                with pytest.raises(NotImplementedError):
                    dct_func(x, **kwargs)
            return xp.zeros([])
        out = dct_func(x, **kwargs)
        if not self.overwrite_x:
            testing.assert_array_equal(x, x_orig)
        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(
        scipy_name='scp', rtol=1e-4, atol=1e-5, accept_error=ValueError,
        contiguous_check=False
    )
    def test_dct(self, xp, scp, dtype):
        fft_func = getattr(scp.fft, self.function)
        return self._run_transform(fft_func, xp, dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-5, accept_error=ValueError,
                                 contiguous_check=False)
    def test_dct_backend(self, xp, dtype):
        backend = 'scipy' if xp is np else cp_fft
        with scipy_fft.set_backend(backend):
            fft_func = getattr(scipy_fft, self.function)
            return self._run_transform(fft_func, xp, dtype)


@testing.parameterize(
    *(
        # 2D cases
        testing.product(
            {
                'shape': [(3, 4)],
                'type': [2, 3],
                # Note: non-integer s or s == 0 will cause a ValueError
                's': [None, (1, 5), (-1, -1), (0, 5), (1.5, 2.5)],
                'axes': [None, (-2, -1), (-1, -2), (0,)],
                'norm': ['ortho'],
                'overwrite_x': [False],
                'function': ['dctn', 'dstn', 'idctn', 'idstn'],
            }
        )
        # 3D cases
        + testing.product(
            {
                'shape': [(2, 3, 4)],
                'type': [2, 3],
                # Note: len(s) < ndim is allowed
                #       len(s) > ndim raises a ValueError
                's': [None, (1, 5), (1, 4, 10), (2, 2, 2, 2)],
                'axes': [None, (-2, -1), (-1, -2, -3)],
                'norm': ['ortho'],
                'overwrite_x': [False],
                'function': ['dctn', 'dstn', 'idctn', 'idstn'],
            }
        )
        # 4D cases
        + testing.product(
            {
                'shape': [(2, 3, 4, 5)],
                'type': [2, 3],
                's': [None],
                'axes': [None],
                'norm': all_dct_norms,
                'overwrite_x': [True, False],
                'function': ['dctn', 'dstn', 'idctn', 'idstn'],
            }
        )
    )
)
@testing.with_requires('scipy')
class TestDctnDstn:

    def _run_transform(self, dct_func, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        x_orig = x.copy()
        out = dct_func(
            x,
            type=self.type,
            s=self.s,
            axes=self.axes,
            norm=self.norm,
            overwrite_x=self.overwrite_x,
        )
        if not self.overwrite_x:
            testing.assert_array_equal(x, x_orig)
        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(
        scipy_name='scp', rtol=1e-4, atol=1e-5, accept_error=ValueError,
        contiguous_check=False
    )
    def test_dctn(self, xp, scp, dtype):
        fft_func = getattr(scp.fft, self.function)
        return self._run_transform(fft_func, xp, dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-5, accept_error=ValueError,
                                 contiguous_check=False)
    def test_dctn_backend(self, xp, dtype):
        backend = 'scipy' if xp is np else cp_fft
        with scipy_fft.set_backend(backend):
            fft_func = getattr(scipy_fft, self.function)
            return self._run_transform(fft_func, xp, dtype)

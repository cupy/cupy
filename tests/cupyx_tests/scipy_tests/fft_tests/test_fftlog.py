import numpy as np

import cupy
import cupyx.scipy.fft as cp_fft
from cupy import testing

try:
    # scipy.fft is available since scipy v1.4.0+
    import scipy.fft as scipy_fft  # noqa
except ImportError:
    scipy_fft = None

try:
    # fht, ifht, fhtoffset are available in scipy v1.7.0+
    from scipy.fft import fhtoffset
except ImportError:
    fhtoffset = None

atol = {cupy.float64: 1e-10, 'default': 1e-5}
rtol = {cupy.float64: 1e-10, 'default': 1e-5}


@testing.parameterize(*testing.product({
    'mu': [0.3, 0.5, -1.2],
    'shape': [(9,), (10,), (10, 9), (8, 10)],
    'offset': [0.0, 0.1, 'optimal'],
    'bias': [0.0, 0.8, -0.5],
    'function': ['fht', 'ifht'],
}))
@testing.gpu
class TestFftlog:

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=rtol, atol=atol, scipy_name='scp')
    @testing.with_requires('scipy>=1.7.0')
    def test_fht(self, xp, scp, dtype):

        # test function, analytical Hankel transform is of the same form
        def f(r, mu):
            return r**(mu + 1) * xp.exp(-r**2 / 2)

        shape = self.shape
        r = xp.logspace(-4, 4, shape[-1])
        dln = xp.log(r[1]/r[0])

        if len(shape) == 2:
            r = xp.stack([r] * shape[0], axis=0)

        mu = self.mu
        bias = self.bias
        if self.offset == 'optimal':
            offset = fhtoffset(dln, mu, bias)
        else:
            offset = self.offset

        a = f(r, mu).astype(dtype)
        func = getattr(scp.fft, self.function)
        return func(a, dln, mu, offset=offset, bias=bias)


@testing.parameterize(*testing.product({
    'function': ['fht', 'ifht'],
}))
@testing.gpu
class TestFftlogScipyBackend:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-5, accept_error=ValueError,
                                 contiguous_check=False)
    @testing.with_requires('scipy>=1.9.0')
    def test_dct_backend(self, xp, dtype):
        backend = 'scipy' if xp is np else cp_fft
        with scipy_fft.set_backend(backend):
            fft_func = getattr(scipy_fft, self.function)
            r = xp.logspace(-2, 2, 10)
            dln = xp.log(r[1]/r[0])
            return fft_func(r, dln, mu=0.5)

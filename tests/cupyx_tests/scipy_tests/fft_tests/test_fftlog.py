try:
    # fht, ifht, fhtoffset are available in scipy v1.7.0+
    import scipy.fft  # noqa
    from scipy.fft import fhtoffset
except ImportError:
    fhtoffset = None

import cupy
import cupyx.scipy.fft  # noqa
from cupy import testing

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

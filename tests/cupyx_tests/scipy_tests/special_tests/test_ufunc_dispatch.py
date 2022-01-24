import numpy
import cupy
import cupyx.scipy.special

from cupy import testing
import pytest

try:
    import scipy.special

    scipy_ufuncs = {
        f
        for f in scipy.special.__all__
        if isinstance(getattr(scipy.special, f), numpy.ufunc)
    }
    cupyx_scipy_ufuncs = {
        f
        for f in dir(cupyx.scipy.special)
        if isinstance(getattr(cupyx.scipy.special, f), cupy.ufunc)
    }
except ImportError:
    scipy_ufuncs = set()
    cupyx_scipy_ufuncs = set()


@testing.gpu
@testing.with_requires("scipy")
@pytest.mark.parametrize("ufunc", sorted(cupyx_scipy_ufuncs & scipy_ufuncs))
class TestUfunc:
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_dispatch(self, xp, ufunc):
        ufunc = getattr(scipy.special, ufunc)
        # some ufunc (like sph_harm) do not work with float inputs
        # therefore we retrieve the types from the ufunc itself
        types = ufunc.types[0]
        args = [
            cupy.testing.shaped_random((5,), xp, dtype=types[i])
            for i in range(ufunc.nin)
        ]
        return ufunc(*args)

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
    def _should_skip(self, f):
        if f.startswith("gammainc"):
            if (
                cupy.cuda.runtime.is_hip
                and cupy.cuda.runtime.runtimeGetVersion() < 5_00_00000
            ):
                pytest.skip('ROCm/HIP fails in ROCm 4.x')

    @testing.numpy_cupy_allclose(atol=1e-4)
    def test_dispatch(self, xp, ufunc):
        self._should_skip(ufunc)
        ufunc = getattr(scipy.special, ufunc)
        # some ufunc (like sph_harm) do not work with float inputs
        # therefore we retrieve the types from the ufunc itself
        if ufunc.__name__ in ['bdtr', 'bdtrc', 'bdtri']:
            # Make sure non-deprecated types are used.
            # (avoids DeprecationWarning from SciPy >=1.7)
            types = 'dld->d'
        else:
            types = ufunc.types[0]
        args = [
            cupy.testing.shaped_random((5,), xp, dtype=types[i])
            for i in range(ufunc.nin)
        ]
        return ufunc(*args)
